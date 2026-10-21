"""Tests of the inference export (sparc.vocoders.export) and the sparc-export CLI helpers (CPU, synthetic weights, no network)."""

import csv
import json

import numpy as np
import pytest
import torch
from hydra.utils import instantiate

from sparc.cli.export_vocoder import parse_args
from sparc.vocoders import export as ex
from sparc.vocoders.constants import HOP, N_FEATURES, SPEAKER_RAW_DIM
from sparc.vocoders.eval.vocoder_loader import compose_vocoder_config
from sparc.vocoders.models.speaker import SpeakerFFN
from sparc.vocoders.training.callbacks import fixed_torch_rng

SMALL = {
    "hifigan": ["vocoder.generator.channels=32"],
    "vocos": ["vocoder.generator.dim=32", "vocoder.generator.intermediate_dim=96", "vocoder.generator.num_layers=2"],
    "ddsp": [
        "vocoder.generator.channels=24",
        "vocoder.generator.n_harmonics=12",
        "vocoder.generator.n_noise_bands=17",
        "vocoder.generator.post_filter_taps=65",
        "vocoder.generator.head_hidden=24",
        "vocoder.generator.up_channels=[16,12]",
    ],
}
FRAMES = 20


def make_stats() -> dict:
    rng = np.random.default_rng(0)
    return {
        "ema_mean": [0.1 * i for i in range(12)],
        "ema_std": [1.0 + 0.1 * i for i in range(12)],
        "logf0_mean": 5.0,
        "logf0_std": 0.3,
        "loud_log_mean": -5.0,
        "loud_log_std": 1.5,
        "per_mean": 0.5,
        "per_std": 0.4,
        "spk_l0_mean": rng.normal(size=SPEAKER_RAW_DIM).tolist(),
        "spk_l0_std": (1.0 + rng.random(SPEAKER_RAW_DIM)).tolist(),
        "spk_l6_mean": rng.normal(size=SPEAKER_RAW_DIM).tolist(),
        "spk_l6_std": (1.0 + rng.random(SPEAKER_RAW_DIM)).tolist(),
        "unrelated": 1.0,
    }


def make_inputs(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(2, N_FEATURES, FRAMES, generator=g)
    feats[:, 12] = 100.0 + 50.0 * torch.rand(2, FRAMES, generator=g)
    feats[:, 13] = 0.01 + 0.05 * torch.rand(2, FRAMES, generator=g)
    feats[:, 14] = torch.rand(2, FRAMES, generator=g)
    return feats, torch.randn(2, SPEAKER_RAW_DIM, generator=g)


def make_checkpoint(tmp_path, vocoder: str, stats: dict):
    """A training-style checkpoint (generator, speaker FFN, a discriminator entry, counters) and its composed config."""
    cfg = compose_vocoder_config("main_800k", vocoder, SMALL[vocoder])
    torch.manual_seed(1)
    generator = instantiate(cfg.vocoder.generator, stats=stats, _convert_="all")
    layer = cfg.speaker.layer
    speaker = SpeakerFFN(dropout=cfg.speaker.dropout, mean=stats[f"spk_{layer}_mean"], std=stats[f"spk_{layer}_std"])
    state = {f"generator.{k}": v for k, v in generator.state_dict().items()}
    state.update({f"speaker.{k}": v for k, v in speaker.state_dict().items()})
    state["discriminators.mpd.weight"] = torch.randn(4, 4)
    path = tmp_path / f"{vocoder}.ckpt"
    torch.save({"state_dict": state, "vocoder_counters": {"g_step": 1234, "d_step": 1200}}, path)
    return path, cfg, generator.eval(), speaker.eval()


@pytest.fixture
def inversion(tmp_path):
    path = tmp_path / "head.npz"
    np.savez(path, weight=np.ones((12, 1024), np.float32), bias=np.zeros(12, np.float32))
    return path


@pytest.mark.parametrize("vocoder", ["hifigan", "vocos", "ddsp"])
def test_export_round_trip_is_exact(tmp_path, vocoder, inversion):
    stats = make_stats()
    ckpt, cfg, generator, speaker = make_checkpoint(tmp_path, vocoder, stats)
    out = ex.export_vocoder(ckpt, cfg, tmp_path / "export", stats=stats, extractor={"model_name": "en+"}, inversion_npz=inversion)
    names = {name for name, _ in ex.list_files(out)}
    assert {ex.WEIGHTS_FILE, ex.CONFIG_FILE, ex.STATS_FILE, ex.CARD_FILE, f"{ex.INVERSION_DIR}/head.npz"} <= names

    from safetensors.torch import load_file

    keys = set(load_file(str(out / ex.WEIGHTS_FILE)))
    assert keys and all(k.startswith(("generator.", "speaker.")) for k in keys)
    stored_stats = json.loads((out / ex.STATS_FILE).read_text())
    assert "unrelated" not in stored_stats and "spk_l0_mean" not in stored_stats  # only the layer the model reads
    config = json.loads((out / ex.CONFIG_FILE).read_text())
    assert config["training"]["g_step"] == 1234 and config["vocoder"] == vocoder
    assert config["inputs"]["inversion_head_sha256"] == ex.file_sha256(inversion)

    model = ex.load_exported(out)
    feats, spk_raw = make_inputs()
    with torch.no_grad(), fixed_torch_rng(None):
        expected = generator(feats, speaker(spk_raw))
    with torch.no_grad(), fixed_torch_rng(None):
        got = model(feats, spk_raw)
    assert got.shape == (2, 1, FRAMES * HOP)
    assert torch.equal(got, expected)


def test_overwrite_replaces_an_earlier_export_but_never_other_files(tmp_path, inversion):
    stats = make_stats()
    ckpt, cfg, _, _ = make_checkpoint(tmp_path, "hifigan", stats)
    out = ex.export_vocoder(ckpt, cfg, tmp_path / "export", stats=stats, inversion_npz=inversion)
    (out / ex.VERIFICATION_FILE).write_text("{}")
    with pytest.raises(FileExistsError):
        ex.export_vocoder(ckpt, cfg, out, stats=stats)
    ex.export_vocoder(ckpt, cfg, out, stats=stats, overwrite=True)
    assert not (out / ex.INVERSION_DIR).exists() and not (out / ex.VERIFICATION_FILE).exists()
    (out / "notes.txt").write_text("keep me")
    with pytest.raises(FileExistsError, match="not part of an export"):
        ex.export_vocoder(ckpt, cfg, out, stats=stats, overwrite=True)
    assert (out / "notes.txt").read_text() == "keep me" and (out / ex.WEIGHTS_FILE).exists()


def test_inversion_hash_mismatch_is_rejected(tmp_path, inversion):
    stats = make_stats()
    ckpt, cfg, _, _ = make_checkpoint(tmp_path, "hifigan", stats)
    with pytest.raises(ValueError, match="sha256"):
        ex.export_vocoder(ckpt, cfg, tmp_path / "export", stats=stats, inversion_npz=inversion, inversion_sha256="0" * 64)


@pytest.fixture
def exported(tmp_path):
    stats = make_stats()
    ckpt, cfg, _, _ = make_checkpoint(tmp_path, "vocos", stats)
    return ex.export_vocoder(ckpt, cfg, tmp_path / "export", stats=stats)


def test_stale_statistics_are_detected(exported):
    stats = json.loads((exported / ex.STATS_FILE).read_text())
    stats["per_mean"] += 0.1
    (exported / ex.STATS_FILE).write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="buffer"):
        ex.load_exported(exported)


@pytest.mark.parametrize(
    "target, generator_extra",
    [("os.system", {}), ("builtins.eval", {}), ("sparc.vocoders.models.vocos.VocosVocoder", {"x": {"_target_": "os.system"}})],
)
def test_only_known_generator_classes_are_built(exported, target, generator_extra):
    config = json.loads((exported / ex.CONFIG_FILE).read_text())
    config["generator"]["_target_"] = target
    config["generator"].update(generator_extra)
    (exported / ex.CONFIG_FILE).write_text(json.dumps(config))
    with pytest.raises(ValueError):
        ex.load_exported(exported)


def test_wrong_format_version_is_rejected(exported):
    config = json.loads((exported / ex.CONFIG_FILE).read_text())
    config["format_version"] = ex.FORMAT_VERSION + 1
    (exported / ex.CONFIG_FILE).write_text(json.dumps(config))
    with pytest.raises(ValueError, match="format_version"):
        ex.load_exported(exported)


def test_forward_checks_input_shapes(exported):
    model = ex.load_exported(exported)
    feats, spk_raw = make_inputs()
    with pytest.raises(ValueError):
        model(feats[:, :14], spk_raw)
    with pytest.raises(ValueError):
        model(feats, spk_raw[:, :512])


def write_table(path):
    cols = ["system", "label", "condition", "stratum", "n_utt", "n_speakers"]
    metrics = ["utmos", "wer", "f0_rmse_cents"]
    for m in metrics:
        cols += [m, f"{m}_lo", f"{m}_hi", f"{m}_n"]
    rows = [
        ["gt", "ground truth", "T1", "all", "4687", "39", "4.208", "4.17", "4.24", "4687", "0.0189", "0.016", "0.022", "4687", "5.77", "5.7", "5.8", "4686"],
        ["vocos_800k", "Vocos (800k)", "T1", "all", "4687", "39", "3.969", "3.94", "4.00", "4687", "0.0244", "0.021", "0.028", "4687", "32.9", "30.0", "35.9", "4686"],
        ["vocos_800k", "Vocos (800k)", "T2", "all", "4686", "38", "3.971", "3.94", "4.01", "4686", "0.0249", "0.021", "0.028", "4686", "33.0", "30.0", "36.0", "4685"],
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


def test_metrics_and_model_card(tmp_path, exported):
    write_table(tmp_path / "table.csv")
    metrics = ex.metrics_from_table(tmp_path / "table.csv", "vocos_800k")
    assert metrics["values"]["utmos"]["value"] == pytest.approx(3.969)
    assert metrics["reference"]["values"]["wer"]["value"] == pytest.approx(0.0189)
    with pytest.raises(KeyError):
        ex.metrics_from_table(tmp_path / "table.csv", "hifigan_800k")
    config = json.loads((exported / ex.CONFIG_FILE).read_text())
    config["metrics"] = metrics
    card = ex.render_model_card(config, license_id="cc-by-4.0", repo_id="someone/sparc-vocos")
    assert card.startswith("---\nlicense: cc-by-4.0\n")
    assert 'load_exported("someone/sparc-vocos")' in card
    assert "| UTMOS | 3.969 (3.940 to 4.000) | 4.208 (4.170 to 4.240) |" in card
    assert "| WER | 2.44 % (2.10 to 2.80) | 1.89 % (1.60 to 2.20) |" in card
    assert "| F0 RMSE | 32.9 cents (30.0 to 35.9) | 5.8 cents (5.7 to 5.8) |" in card


def test_card_usage_names_the_included_inversion_head(tmp_path, inversion):
    stats = make_stats()
    ckpt, cfg, _, _ = make_checkpoint(tmp_path, "ddsp", stats)
    out = ex.export_vocoder(ckpt, cfg, tmp_path / "export", stats=stats, inversion_npz=inversion)
    ex.set_card_repo(out, "someone/sparc-ddsp", "mit")
    card = (out / ex.CARD_FILE).read_text()
    assert 'hf_hub_download("someone/sparc-ddsp", "inversion/head.npz")' in card
    assert "seed it for reproducible output" in card  # DDSP noise note
    assert json.loads((out / ex.CONFIG_FILE).read_text())["inputs"]["inversion_head_file"] == "inversion/head.npz"


def test_extraction_to_inputs_uses_raw_loudness_and_the_speaker_layer():
    frames = 7
    feats = np.arange(frames * N_FEATURES, dtype=np.float32).reshape(frames, N_FEATURES)
    extracted = {"feats": feats, "loud_raw": np.full(frames, 0.02, np.float32), "spk_l6": np.ones(SPEAKER_RAW_DIM, np.float32),
                 "spk_l0": np.zeros(SPEAKER_RAW_DIM, np.float32)}
    features, spk_raw = ex.extraction_to_inputs(extracted, "l6", gain=0.5)
    assert features.shape == (1, N_FEATURES, frames) and spk_raw.shape == (1, SPEAKER_RAW_DIM)
    assert torch.equal(features[0, :13], torch.from_numpy(feats.T[:13].copy()))
    assert torch.allclose(features[0, 13], torch.full((frames,), 0.01))
    assert torch.equal(features[0, 14], torch.from_numpy(feats.T[14].copy()))
    assert float(spk_raw.sum()) == SPEAKER_RAW_DIM
    with pytest.raises(ValueError):
        ex.extraction_to_inputs({**extracted, "feats": feats[:, :14]})


def test_set_card_repo_rewrites_the_card(exported):
    ex.set_card_repo(exported, "someone/sparc-vocos", "apache-2.0")
    card = (exported / ex.CARD_FILE).read_text()
    assert "license: apache-2.0" in card and 'load_exported("someone/sparc-vocos")' in card


class FakeApi:
    def __init__(self):
        self.calls = []

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))
        return type("Info", (), {"commit_url": "https://huggingface.co/someone/x/commit/abc"})()


def test_upload_creates_a_private_repo_and_uploads_once(exported):
    api = FakeApi()
    url = ex.upload_export(exported, "someone/x", api=api, commit_message="first")
    assert url.endswith("/commit/abc")
    (n1, create), (n2, upload) = api.calls
    assert n1 == "create_repo" and create == {"repo_id": "someone/x", "repo_type": "model", "private": True, "exist_ok": True}
    assert n2 == "upload_folder" and upload["folder_path"] == str(exported) and upload["commit_message"] == "first"
    assert upload["allow_patterns"] == ex.UPLOAD_PATTERNS and "*" not in upload["allow_patterns"]


def test_upload_refuses_an_incomplete_export(tmp_path):
    (tmp_path / "e").mkdir()
    with pytest.raises(FileNotFoundError):
        ex.upload_export(tmp_path / "e", "someone/x", api=FakeApi())


def test_cli_argument_rules():
    with pytest.raises(SystemExit):
        parse_args(["--vocoder", "vocos", "--experiment", "main", "--out", "x", "--upload"])
    with pytest.raises(SystemExit):
        parse_args(["--vocoder", "vocos", "--experiment", "main", "--out", "x", "--metrics-system", "vocos"])
    args = parse_args(["--vocoder", "vocos", "--experiment", "main", "--out", "x", "--repo-id", "a/b", "--upload"])
    assert args.upload and not args.public and args.verify == 8
