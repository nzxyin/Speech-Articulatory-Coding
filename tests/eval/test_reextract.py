"""Re-extraction without CREPE dither, the reference systems and the vocoder loader.

The first group is quick and needs no model. The ``slow`` group uses the pinned checkpoints (private hub cache, packed
test.clean features and LibriTTS-R audio; environment as in ``~/sparc-vocoders-work/env.sh``) and runs on CPU, about a
minute per model load; it is skipped when the environment is not set.
"""

import os
from pathlib import Path

import librosa
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from sparc.vocoders.constants import EMA_NAMES, SAMPLE_RATE, feature_length
from sparc.vocoders.features.extractor import SparcFeatureExtractor, crepe_dither
from sparc.vocoders.eval import reextract
from sparc.vocoders.eval.references import fit_length, peak_normalize
from sparc.vocoders.models.frontend import load_stats

NEEDED_ENV = ("HF_HUB_CACHE", "SPARC_REFIT_NPZ", "SPARC_VOC_CACHE", "SPARC_VOC_RUNS", "LIBRITTSR_RAW")


# ---------------------------------------------------------------------------------------------------------------------
# quick tests: dither switch, config, helpers


def test_crepe_dither_switch_is_deterministic_and_restores_the_function():
    convert = pytest.importorskip("torchcrepe.convert")
    original = convert.dither
    bins = torch.arange(10, dtype=torch.float32)
    with crepe_dither(False):
        assert convert.dither is not original
        first, second = convert.bins_to_cents(bins), convert.bins_to_cents(bins)
    assert convert.dither is original
    assert torch.equal(first, second)  # no noise inside the block
    torch.testing.assert_close(first, 20.0 * bins + 1997.3794084376191)
    assert not torch.equal(convert.bins_to_cents(bins), convert.bins_to_cents(bins))  # dither is back on outside


def test_crepe_dither_restores_after_an_exception_and_enabled_changes_nothing():
    convert = pytest.importorskip("torchcrepe.convert")
    original = convert.dither
    with pytest.raises(RuntimeError, match="boom"):
        with crepe_dither(False):
            raise RuntimeError("boom")
    assert convert.dither is original
    with crepe_dither(True):
        assert convert.dither is original  # the default switch does not touch torchcrepe at all
    assert convert.dither is original


def test_fit_length_and_peak_normalize():
    x = np.arange(1, 6, dtype=np.float32)
    assert np.array_equal(fit_length(x, 3), x[:3])
    padded = fit_length(x, 8)
    assert padded.dtype == np.float32 and np.array_equal(padded[:5], x) and not padded[5:].any()
    y = peak_normalize(np.array([0.1, -0.5, 0.25], dtype=np.float32), -3.0)
    assert np.abs(y).max() == pytest.approx(10 ** (-3.0 / 20), rel=1e-6) and y.dtype == np.float32
    silent = np.zeros(4, dtype=np.float32)
    assert np.array_equal(peak_normalize(silent, -3.0), silent)


def test_extractor_config_forces_no_dither_and_reads_cache_config(monkeypatch, tmp_path):
    for name in NEEDED_ENV:
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    cfg = reextract.extractor_config(OmegaConf.create({}))
    assert cfg.extractor.dither is False
    assert cfg.extractor.hf_hub_cache == str(tmp_path / "hf_hub_cache")
    assert cfg.inversion.linear_npz == str(tmp_path / "sparc_refit_npz")
    assert cfg.inversion.pitch_q == 2 and cfg.inversion.pitch_hop_length == 160
    # a caller that supplies the blocks keeps its own values, still with the dither forced off
    own = OmegaConf.create({"extractor": {"dither": True, "x": 1}, "inversion": {"y": 2}})
    out = reextract.extractor_config(own)
    assert out.extractor.dither is False and out.extractor.x == 1 and out.inversion.y == 2
    assert own.extractor.dither is True  # the caller's config is not modified


def test_head_sha256_depends_on_weight_and_bias():
    w, b = torch.ones(3, 4), torch.zeros(3)
    assert reextract.head_sha256(w, b) == reextract.head_sha256(w.clone(), b.clone())
    assert reextract.head_sha256(w, b) != reextract.head_sha256(w * 2, b)
    assert reextract.head_sha256(w, b) != reextract.head_sha256(w, b + 1)


def test_unknown_head_is_rejected():
    with pytest.raises(ValueError, match="head must be one of"):
        reextract.ReExtractor(OmegaConf.create({}), "other", "cpu")


# ---------------------------------------------------------------------------------------------------------------------
# slow tests: real checkpoints, real utterances


def require_env() -> None:
    if any(name not in os.environ for name in NEEDED_ENV):
        pytest.skip("environment of the feature cache is not set")


@pytest.fixture(scope="module")
def utterances():
    """Two short test.clean utterances: ids, gt audio at the eval gain, cached features (loudness with the gain)."""
    require_env()
    import pandas as pd

    from sparc.vocoders.data.dataset import FullUtteranceDataset

    packed = Path(os.environ["SPARC_VOC_CACHE"]) / "packed" / "test.clean"
    if not (packed / "index.parquet").exists():
        pytest.skip(f"{packed} is missing")
    index = pd.read_parquet(packed / "index.parquet", columns=["id", "speaker", "T"])
    index = index.assign(id=index["id"].astype(str)).sort_values("id")
    short = index[(index["T"] >= 120) & (index["T"] <= 220)]
    ids = short.groupby("speaker").head(1)["id"].tolist()[:2]
    assert len(ids) == 2
    dataset = FullUtteranceDataset(packed, ids=ids, gain_db=-3.0, condition="T1")
    items = [dataset[i] for i in range(len(dataset))]
    return [
        {
            "id": it["id"],
            "wav": it["audio"][0].numpy().astype(np.float32),
            "features": it["features"].numpy().T,
            "path": str(dataset.store.wav_paths[dataset.items[i]]),
        }
        for i, it in enumerate(items)
    ]


@pytest.fixture(scope="module")
def refit(utterances):
    try:
        return reextract.ReExtractor(OmegaConf.create({}), "refit", "cpu")
    except FileNotFoundError as error:
        pytest.skip(str(error))


@pytest.fixture(scope="module")
def shipped(utterances):
    try:
        return reextract.ReExtractor(OmegaConf.create({}), "shipped", "cpu")
    except FileNotFoundError as error:
        pytest.skip(str(error))


@pytest.mark.slow
def test_active_head_is_explicit(refit, shipped):
    assert refit.head == "refit" and shipped.head == "shipped"
    assert refit.head_matches_npz() and not shipped.head_matches_npz()
    assert refit.head_hash != shipped.head_hash
    assert refit.head_weight.shape == shipped.head_weight.shape and not torch.equal(refit.head_weight, shipped.head_weight)
    assert refit.describe()["dither"] is False and shipped.describe()["head"] == "shipped"
    # the shipped head is the linear_model entry of the en+ checkpoint
    state = torch.load(reextract.pinned_paths(shipped.cfg.extractor)["ckpt"], map_location="cpu", weights_only=True)
    linear = state["state_dict"]["linear_model"]
    assert torch.equal(shipped.head_weight, linear["weight"].float()) and torch.equal(shipped.head_bias, linear["bias"].float())


@pytest.mark.slow
def test_no_dither_extraction_is_bit_identical(refit, utterances):
    wav = utterances[0]["wav"]
    np.random.seed(1)
    first = refit.extract(wav, SAMPLE_RATE)
    np.random.seed(2)
    second = refit.extract(wav, SAMPLE_RATE)
    for key in ("feats", "loud_raw"):
        np.testing.assert_array_equal(first[key], second[key])
    assert first["feats"].dtype == np.float32 and first["feats"].shape == (first["T"], 15)
    assert first["loud_raw"].shape == (first["T"],)
    assert first["T"] == feature_length(len(wav))  # 480 T samples give T - 1 frames


@pytest.mark.slow
def test_default_path_keeps_the_dither(refit, utterances):
    """``dither=True`` (the cache's setting) is reproducible for one seed and differs from no dither only in the pitch."""
    wav = utterances[0]["wav"]
    cfg = reextract.extractor_config(OmegaConf.create({}))
    cfg.extractor.dither = True
    cached_style = SparcFeatureExtractor(cfg, "cpu")
    assert cached_style.dither is True
    a, b = cached_style.extract(wav, 5), cached_style.extract(wav, 5)
    np.testing.assert_array_equal(a["feats"], b["feats"])
    c = cached_style.extract(wav, 6)
    assert not np.array_equal(a["feats"][:, 12], c["feats"][:, 12])  # another seed, another dither
    clean = refit.extract(wav, SAMPLE_RATE)
    both = (a["feats"][:, 14] > 0) & (clean["feats"][:, 14] > 0)
    cents = 1200 * np.log2(a["feats"][both, 12] / clean["feats"][both, 12])
    assert np.abs(cents).max() < 100 and np.sqrt(np.mean(cents**2)) > 1  # the dither is about +-20 cents
    np.testing.assert_array_equal(a["feats"][:, :12], clean["feats"][:, :12])
    np.testing.assert_array_equal(a["loud_raw"], clean["loud_raw"])


@pytest.mark.slow
@pytest.mark.parametrize("which", [0, 1])
def test_reextraction_of_the_full_file_reproduces_the_cache(refit, utterances, which):
    """The re-extractor is the cache pipeline: on the whole original file (any gain) the EMA equals the cached EMA."""
    import soundfile as sf

    item = utterances[which]
    full, rate = sf.read(item["path"], dtype="float32")
    assert rate == SAMPLE_RATE
    out = refit.extract(full * 0.37, SAMPLE_RATE)  # SPARC z-scores its input, so a gain changes nothing
    cached = item["features"]
    assert out["T"] == len(cached)
    assert np.abs(out["feats"][:, :12] - cached[:, :12]).max() < 1e-4
    both = (out["feats"][:, 14] > 0) & (cached[:, 14] > 0)
    cents = 1200 * np.log2(out["feats"][both, 12] / cached[both, 12])
    assert np.abs(cents).max() < 100 and np.sqrt(np.mean(cents**2)) < 30  # only the cached CREPE dither differs


@pytest.mark.slow
@pytest.mark.parametrize("which", [0, 1])
def test_gt_reextraction_matches_the_cached_features(refit, utterances, which):
    """The extraction floor: no-dither re-extraction of the gt WAV (``480 T`` samples) against the cached features.

    Cutting the file to ``480 T`` samples changes the per-utterance z-scoring slightly and, mainly, the last frames
    (WavLM sees no samples after the cut): measured on two utterances, the EMA error is below 0.13 training std in the
    interior and reaches 0.6-1.5 std in the last two frames, so the whole-utterance EMA r is 0.998-0.9986 and the RMSE
    0.04-0.05 std (r 0.9999 and RMSE 0.016-0.019 without the last 3 frames). Loudness is exact.
    """
    item = utterances[which]
    out = refit.extract(item["wav"], SAMPLE_RATE)
    cached = item["features"]
    n = out["T"]
    assert len(cached) - n == 1  # 480 T samples give T - 1 frames
    ema_std = np.asarray(load_stats(os.environ["SPARC_VOC_CACHE"] + "/stats/train_stats.json")["ema_std"])
    err = np.abs(out["feats"][:n, :12] - cached[:n, :12]) / ema_std
    rs = [np.corrcoef(out["feats"][: n - 3, c], cached[: n - 3, c])[0, 1] for c in range(12)]
    rs_all = [np.corrcoef(out["feats"][:n, c], cached[:n, c])[0, 1] for c in range(12)]
    print(f"gt vs cache {item['id']}: EMA r mean {np.mean(rs_all):.5f} (without last 3 frames {np.mean(rs):.5f}), "
          f"RMSE {np.sqrt((err**2).mean(0)).mean():.4f} std, interior max {err[5:-5].max():.3f}, last 2 frames {err[-2:].max():.3f}")
    assert np.mean(rs_all) > 0.997 and np.mean(rs) > 0.9995 and min(rs) > 0.999
    assert err[5:-5].max() < 0.25  # interior frames agree to a quarter of a training std
    both = (out["feats"][:n, 14] > 0) & (cached[:n, 14] > 0)
    assert both.sum() > 20
    cents = 1200 * np.log2(out["feats"][:n][both, 12] / cached[:n][both, 12])
    print(f"  f0 rmse {np.sqrt(np.mean(cents**2)):.1f} cents, max {np.abs(cents).max():.1f}, "
          f"voicing differs on {np.mean((out['feats'][:n, 14] > 0) != (cached[:n, 14] > 0)):.3f} of frames")
    assert np.sqrt(np.mean(cents**2)) < 30  # the dither is triangular with a peak of 20 cents
    assert np.abs(cents).max() < 100
    assert np.mean((out["feats"][:n, 14] > 0) != (cached[:n, 14] > 0)) < 0.05
    np.testing.assert_allclose(out["loud_raw"], cached[:n, 13], rtol=1e-4, atol=1e-7)  # cached loudness carries the eval gain, the WAV too


@pytest.mark.slow
def test_other_sample_rates_are_resampled(refit, utterances):
    wav = utterances[0]["wav"]
    direct = refit.extract(wav, SAMPLE_RATE)
    wav16 = librosa.resample(wav, orig_sr=SAMPLE_RATE, target_sr=16000, res_type="soxr_hq")
    out = refit.extract(wav16, 16000)
    assert abs(out["T"] - direct["T"]) <= 1
    n = min(out["T"], direct["T"])
    assert np.corrcoef(out["feats"][:n, 0], direct["feats"][:n, 0])[0, 1] > 0.99


@pytest.mark.slow
def test_shipped_head_differs_from_refit_head_on_the_same_audio(refit, shipped, utterances):
    wav = utterances[0]["wav"]
    a, b = refit.extract(wav, SAMPLE_RATE), shipped.extract(wav, SAMPLE_RATE)
    assert a["T"] == b["T"]
    assert not np.allclose(a["feats"][:, :12], b["feats"][:, :12], atol=1e-3)  # different heads, different EMA units
    np.testing.assert_array_equal(a["feats"][:, 12:], b["feats"][:, 12:])  # pitch, loudness, periodicity are head-free
    np.testing.assert_array_equal(a["loud_raw"], b["loud_raw"])


# reference systems ---------------------------------------------------------------------------------------------------


def references_cfg():
    group = OmegaConf.load(Path(reextract.CONF_DIR) / "eval_features" / "default.yaml")
    cfg = OmegaConf.create({"eval_features": group})
    cfg.eval_features.references.hub_cache = os.environ["HF_HUB_CACHE"]
    return cfg


@pytest.mark.slow
def test_vocos_mel_copy_synthesis_has_the_input_length(utterances):
    require_env()
    pytest.importorskip("vocos")
    from sparc.vocoders.eval.references import VocosMelReference

    try:
        ref = VocosMelReference(references_cfg(), "cpu")
    except FileNotFoundError as error:
        pytest.skip(str(error))
    wav = utterances[0]["wav"]
    out = ref.synth(wav)
    assert out.shape == wav.shape and out.dtype == np.float32 and np.isfinite(out).all()
    # copy synthesis is close to the input in level and not silent; the model is level-preserving
    assert 0.3 < np.sqrt(np.mean(out**2)) / np.sqrt(np.mean(wav**2)) < 3.0
    assert ref.describe()["revision"] == "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"


@pytest.mark.slow
def test_enplus_reference_lengths_levels_and_speaker_embedding(utterances):
    require_env()
    from sparc.vocoders.eval.references import EnPlusReference

    try:
        ref = EnPlusReference(references_cfg(), "cpu")
    except FileNotFoundError as error:
        pytest.skip(str(error))
    wav = utterances[0]["wav"]
    frames = len(wav) // 480
    code = ref.encode(wav)
    emb = ref.spk_emb(code)
    assert emb.shape == (64,) and emb.dtype == np.float32
    n16 = 320 * frames
    out = ref.decode(code, emb, n16)
    assert out.shape == (n16,) and out.dtype == np.float32
    assert np.abs(out).max() == pytest.approx(10 ** (-3.0 / 20), rel=1e-4)
    raw = ref.decode_raw(code, emb)
    assert len(raw) == 320 * len(code["ema"])  # the decoder returns 320 samples per encoded frame
    assert n16 - len(raw) in (0, 320)  # the shipped encoder drops one frame of 320 T input samples: zero-padded at the end
    assert not out[len(raw):].any() if len(raw) < n16 else True
    # another speaker embedding changes the waveform, the same one reproduces it
    other = ref.decode(code, np.zeros_like(emb), n16)
    assert not np.allclose(other, out)
    np.testing.assert_allclose(ref.decode(code, emb, n16), out, atol=1e-5)


# vocoder loader ------------------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_load_vocoder_from_a_fresh_checkpoint(tmp_path):
    require_env()
    from sparc.vocoders.eval.vocoder_loader import checkpoint_info, load_vocoder
    from sparc.vocoders.training.module import VocoderGANModule
    from sparc.vocoders.eval.vocoder_loader import compose_vocoder_config

    cfg = compose_vocoder_config("smoke", "hifigan")
    if not Path(cfg.stats_path).exists():
        pytest.skip("training statistics are missing")
    torch.manual_seed(0)
    fresh = VocoderGANModule(cfg)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    counters = {"g_step": 1234, "d_step": 1200, "samples_consumed": 5, "last_validated_g_step": 0}
    torch.save({"state_dict": fresh.state_dict(), "vocoder_counters": counters}, ckpt_dir / "step000001234.ckpt")
    (ckpt_dir / "step000002000.ckpt").write_bytes(b"truncated")  # newest by step but unloadable: skipped

    module, loaded_cfg = load_vocoder("smoke", "hifigan", ckpt_dir, "cpu")
    assert not module.training and checkpoint_info(module) == {"ckpt": str(ckpt_dir / "step000001234.ckpt"), "g_step": 1234}
    assert loaded_cfg.loaded.g_step == 1234 and loaded_cfg.loaded.name.endswith("/hifigan")
    for (name, a), (_, b) in zip(fresh.state_dict().items(), module.state_dict().items()):
        assert torch.equal(a, b), name
    with pytest.raises(FileNotFoundError):
        load_vocoder("smoke", "hifigan", tmp_path / "nowhere.ckpt", "cpu")


@pytest.mark.slow
def test_load_vocoder_newest_run_checkpoint_if_present():
    require_env()
    from sparc.vocoders.eval.vocoder_loader import compose_vocoder_config, load_vocoder
    from sparc.vocoders.training.callbacks import list_step_checkpoints

    cfg = compose_vocoder_config("main", "hifigan")
    steps = list_step_checkpoints(Path(cfg.run_dir) / "ckpt") if (Path(cfg.run_dir) / "ckpt").exists() else []
    if not steps:
        pytest.skip("no checkpoint of main/hifigan yet")
    module, loaded = load_vocoder("main", "hifigan", None, "cpu")
    assert loaded.loaded.g_step >= steps[0][0] and Path(loaded.loaded.ckpt).exists()
    assert module.checkpoint_step == loaded.loaded.g_step and not module.training
