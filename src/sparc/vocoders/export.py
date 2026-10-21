"""Export of a trained vocoder for inference, loading of an export, model card and Hugging Face Hub upload.

An export is a directory (or a Hub model repository) with:

- ``model.safetensors``: the generator (``generator.*``) and speaker FFN (``speaker.*``) weights of a
  :class:`~sparc.vocoders.training.module.VocoderGANModule` checkpoint, including their normalization buffers. The
  discriminators, the optimizer states and the training counters are dropped.
- ``config.json``: format and version, the generator class and arguments, the speaker FFN settings, the input
  contract (feature layout, loudness and speaker-vector definitions, the feature extractor and inversion head that the
  training features came from) and provenance (experiment, seed, generator step, source checkpoint hash).
- ``stats.json``: the training statistics the generator and speaker FFN are built with (the same values are stored as
  buffers in the weights; :func:`load_exported` checks that they agree).
- ``README.md``: model card (:func:`render_model_card`).
- optionally ``inversion/<name>.npz``: the linear inversion head that produced the 12 EMA channels of the training
  features, so that new inputs can be extracted the same way.
- ``verification.json`` (written by ``sparc-export``): exported against original synthesis on test utterances.

:func:`load_exported` rebuilds the generator from ``config.json``. Only the three vocoder classes of this package are
accepted as generator class, so a downloaded config cannot import arbitrary code.
"""

import hashlib
import importlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from sparc.vocoders.constants import (
    EMA_NAMES,
    FRAME_RATE,
    HOP,
    LOUDNESS_EPS,
    N_FEATURES,
    SAMPLE_RATE,
    SPEAKER_DIM,
    SPEAKER_RAW_DIM,
)
from sparc.vocoders.models.frontend import load_stats
from sparc.vocoders.models.speaker import SpeakerFFN

FORMAT = "sparc-articulatory-vocoder"
FORMAT_VERSION = 1
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
STATS_FILE = "stats.json"
CARD_FILE = "README.md"
VERIFICATION_FILE = "verification.json"
INVERSION_DIR = "inversion"
MODULE_PREFIXES = ("generator.", "speaker.")
GENERATOR_CLASSES = {
    "sparc.vocoders.models.hifigan.HiFiGANVocoder",
    "sparc.vocoders.models.ddsp.DDSPVocoder",
    "sparc.vocoders.models.vocos.VocosVocoder",
}
FRONTEND_STATS = ("ema_mean", "ema_std", "logf0_mean", "logf0_std", "loud_log_mean", "loud_log_std", "per_mean", "per_std")
FEATURE_CHANNELS = (*EMA_NAMES, "f0", "loudness", "periodicity")
REPO_URL = "https://github.com/nzxyin/Speech-Articulatory-Coding"
RESULTS_URL = f"{REPO_URL}/blob/vocoder-comparison/docs/vocoders/RESULTS.md"


# ---------------------------------------------------------------------------------------------------------- model


class ExportedVocoder(nn.Module):
    """Generator and speaker FFN of a trained vocoder: ``(features [B, 15, T], spk_raw [B, 1024]) -> wav [B, 1, 480 T]``.

    The same computation as :meth:`VocoderGANModule.synthesize`. ``config`` is the parsed ``config.json``. The DDSP
    generator draws noise from the global torch generator, so seed it (or use
    :func:`sparc.vocoders.training.callbacks.fixed_torch_rng`) for reproducible output.
    """

    sample_rate: int = SAMPLE_RATE
    hop_length: int = HOP

    def __init__(self, generator: nn.Module, speaker: SpeakerFFN, config: Mapping[str, Any] | None = None):
        super().__init__()
        self.generator = generator
        self.speaker = speaker
        self.config = dict(config or {})

    def forward(self, features: torch.Tensor, spk_raw: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3 or features.shape[1] != N_FEATURES:
            raise ValueError(f"expected features of shape [B, {N_FEATURES}, T], got {tuple(features.shape)}")
        if spk_raw.shape != (features.shape[0], SPEAKER_RAW_DIM):
            raise ValueError(f"expected spk_raw of shape {(features.shape[0], SPEAKER_RAW_DIM)}, got {tuple(spk_raw.shape)}")
        wav = self.generator(features, self.speaker(spk_raw))
        expected = (features.shape[0], 1, features.shape[-1] * HOP)
        if tuple(wav.shape) != expected:
            raise ValueError(f"generator returned {tuple(wav.shape)}, expected {expected}")
        return wav


def generator_class(target: str) -> type[nn.Module]:
    """The class named by ``target``; raises ``ValueError`` unless it is one of :data:`GENERATOR_CLASSES`."""
    if target not in GENERATOR_CLASSES:
        raise ValueError(f"generator class {target!r} is not one of {sorted(GENERATOR_CLASSES)}")
    module_name, _, class_name = target.rpartition(".")
    return getattr(importlib.import_module(module_name), class_name)


def _check_plain(value: Any, where: str) -> None:
    """Rejects nested Hydra targets in generator arguments (they would instantiate other classes)."""
    if isinstance(value, Mapping):
        if "_target_" in value:
            raise ValueError(f"nested _target_ in {where} is not allowed")
        for key, item in value.items():
            _check_plain(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _check_plain(item, f"{where}[{i}]")


def build_modules(config: Mapping[str, Any], stats: Mapping[str, Any]) -> ExportedVocoder:
    """Builds the (untrained) generator and speaker FFN described by an export ``config``."""
    gen = dict(config["generator"])
    target = gen.pop("_target_")
    _check_plain(gen, "generator")
    generator = generator_class(target)(stats=dict(stats), **gen)
    spk = config["speaker"]
    layer = spk["layer"]
    speaker = SpeakerFFN(
        in_dim=int(spk.get("in_dim", SPEAKER_RAW_DIM)),
        hidden_dim=int(spk.get("hidden_dim", SPEAKER_RAW_DIM)),
        out_dim=int(spk.get("out_dim", SPEAKER_DIM)),
        dropout=float(spk["dropout"]),
        mean=stats[f"spk_{layer}_mean"],
        std=stats[f"spk_{layer}_std"],
    )
    return ExportedVocoder(generator, speaker, config)


def _buffers(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: buf.detach().clone() for name, buf in module.named_buffers()}


def load_exported(
    source: str | Path,
    device: torch.device | str = "cpu",
    revision: str | None = None,
    token: str | bool | None = None,
) -> ExportedVocoder:
    """Loads an export from a local directory or a Hub model repository ``source``; returns it in eval mode.

    The weights are loaded strictly, and every buffer stored in the weights (feature and speaker normalization) must
    equal the value built from ``stats.json``.
    """
    from safetensors.torch import load_file

    path = Path(source)
    if not path.is_dir():
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(
                repo_id=str(source),
                revision=revision,
                token=token,
                allow_patterns=[CONFIG_FILE, STATS_FILE, WEIGHTS_FILE],
            )
        )
    config = json.loads((path / CONFIG_FILE).read_text())
    if config.get("format") != FORMAT:
        raise ValueError(f"{path / CONFIG_FILE} is not a {FORMAT} export (format {config.get('format')!r})")
    if int(config.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError(f"unsupported format_version {config.get('format_version')!r}, expected {FORMAT_VERSION}")
    stats = json.loads((path / STATS_FILE).read_text())
    module = build_modules(config, stats)
    built = _buffers(module)
    state = load_file(str(path / WEIGHTS_FILE), device="cpu")
    module.load_state_dict(state, strict=True)
    for name, value in _buffers(module).items():
        if not torch.equal(value, built[name]):
            raise ValueError(f"buffer {name} in {WEIGHTS_FILE} differs from the value built from {STATS_FILE}")
    return module.eval().to(torch.device(device))


# ---------------------------------------------------------------------------------------------------------- inputs


def load_feature_extractor(
    inversion_npz: str | Path,
    hub_cache: str | Path | None = None,
    device: torch.device | str = "cpu",
    download: bool = True,
):
    """The feature extractor the training features came from, with the inversion head ``inversion_npz``.

    Builds :class:`~sparc.vocoders.features.extractor.SparcFeatureExtractor` from the package's cache config (pinned
    SPARC en+ and WavLM-large snapshots, checked by sha256). With ``download`` the two snapshots are fetched into
    ``hub_cache`` first (default: the Hugging Face hub cache); the extractor itself then runs offline and sets
    ``HF_HUB_OFFLINE=1`` for the rest of the process, so download anything else from the Hub before calling this. The
    npz must be the refit head of the export (its sha256 is checked).
    """
    from hydra import compose, initialize_config_dir

    from sparc.vocoders.eval.vocoder_loader import CONF_DIR
    from sparc.vocoders.features.extractor import SparcFeatureExtractor

    if hub_cache is None:
        from huggingface_hub import constants

        hub_cache = constants.HF_HUB_CACHE
    overrides = ["stage=extract", f"inversion.linear_npz={Path(inversion_npz).resolve()}", f"extractor.hf_hub_cache={hub_cache}"]
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="cache_config", overrides=overrides)
    if download:
        from huggingface_hub import snapshot_download

        ext = cfg.extractor
        snapshot_download(ext.sparc_repo, revision=ext.sparc_snapshot, cache_dir=str(hub_cache),
                          allow_patterns=[ext.sparc_ckpt, ext.sparc_yaml])
        snapshot_download(ext.wavlm_repo, revision=ext.wavlm_snapshot, cache_dir=str(hub_cache))
    return SparcFeatureExtractor(cfg, device=device)


def extraction_to_inputs(extracted: Mapping[str, Any], speaker_layer: str = "l6", gain: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """``(features [1, 15, T], spk_raw [1, 1024])`` from the output of ``SparcFeatureExtractor.extract``.

    The loudness channel is replaced by the un-normalized loudness ``loud_raw`` times ``gain`` (1: the level of the
    input waveform), as in training; the speaker vector is ``spk_<speaker_layer>``.
    """
    import numpy as np

    from sparc.vocoders.constants import LOUDNESS_CHANNEL

    feats = np.ascontiguousarray(np.asarray(extracted["feats"], dtype=np.float32).T)
    if feats.shape[0] != N_FEATURES:
        raise ValueError(f"expected feats of shape [T, {N_FEATURES}], got {tuple(np.asarray(extracted['feats']).shape)}")
    feats[LOUDNESS_CHANNEL] = (np.asarray(extracted["loud_raw"], dtype=np.float64) * float(gain)).astype(np.float32)
    spk = np.asarray(extracted[f"spk_{speaker_layer}"], dtype=np.float32)
    return torch.from_numpy(feats)[None], torch.from_numpy(spk.copy())[None]


# ---------------------------------------------------------------------------------------------------------- export


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


EXPORT_ENTRIES = (WEIGHTS_FILE, CONFIG_FILE, STATS_FILE, CARD_FILE, VERIFICATION_FILE, INVERSION_DIR)


def clear_export_dir(out: Path) -> None:
    """Removes an earlier export from ``out``; refuses if ``out`` holds anything else (so a wrong path is never wiped)."""
    foreign = sorted(p.name for p in out.iterdir() if p.name not in EXPORT_ENTRIES)
    if foreign:
        raise FileExistsError(f"{out} holds files that are not part of an export ({foreign[:5]}); choose another directory")
    for name in EXPORT_ENTRIES:
        path = out / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def module_state(checkpoint_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The ``generator.*`` and ``speaker.*`` entries of a training state dict, as contiguous standalone tensors."""
    out = {k: v.detach().cpu().clone().contiguous() for k, v in checkpoint_state.items() if k.startswith(MODULE_PREFIXES)}
    if not any(k.startswith("generator.") for k in out) or not any(k.startswith("speaker.") for k in out):
        raise ValueError("the state dict has no generator.* or no speaker.* entries")
    return out


def stats_subset(stats: Mapping[str, Any], speaker_layer: str) -> dict[str, Any]:
    """The statistics the generator and speaker FFN are built from (feature frontend and the speaker layer)."""
    keys = (*FRONTEND_STATS, f"spk_{speaker_layer}_mean", f"spk_{speaker_layer}_std")
    missing = [k for k in keys if k not in stats]
    if missing:
        raise KeyError(f"training statistics lack {missing}")
    return {k: stats[k] for k in keys}


def input_contract(speaker_layer: str, extractor: Mapping[str, Any] | None, inversion_sha256: str | None) -> dict:
    """The description of the two inputs and of the feature extractor the model was trained on."""
    layer = int(speaker_layer.lstrip("l"))
    return {
        "features": {
            "shape": ["batch", N_FEATURES, "frames"],
            "frame_rate_hz": FRAME_RATE,
            "channels": list(FEATURE_CHANNELS),
            "ema": "12 EMA channels (tongue dorsum, body and tip, lower incisor, upper and lower lip; X and Y) from the "
            "SPARC en+ inversion of the 16 kHz waveform, with the linear head given by `inversion_head_sha256`",
            "f0": "CREPE pitch in Hz from SPARC's source extractor, defined on every frame (50 to 550 Hz); voicing is "
            "carried by the periodicity channel",
            "loudness": "un-normalized loudness: mean absolute amplitude of each 320-sample frame of the 16 kHz "
            "waveform at the level wanted in the output (the model reproduces this level; training used peaks of "
            "-6 to -1 dBFS, the evaluation -3 dBFS). This is not SPARC's z-scored loudness.",
            "periodicity": "CREPE periodicity (voicing confidence, 0 to 1), set to 0 on frames that SPARC's loudness "
            "gate marks as silent",
            "loudness_log_epsilon": LOUDNESS_EPS,
        },
        "spk_raw": {
            "shape": ["batch", SPEAKER_RAW_DIM],
            "definition": f"WavLM-large hidden state {layer} of the same SPARC forward pass, averaged over frames "
            "with the periodicity as weights (plain mean if all weights are 0)",
        },
        "output": {"shape": ["batch", 1, f"{HOP} * frames"], "sample_rate": SAMPLE_RATE, "hop_length": HOP},
        "feature_extractor": dict(extractor or {}),
        "inversion_head_sha256": inversion_sha256,
        "inversion_head_file": None,
        "extractor_code": f"{REPO_URL} (sparc.vocoders.features.extractor.SparcFeatureExtractor)",
    }


def export_vocoder(
    checkpoint: str | Path,
    cfg: DictConfig,
    out_dir: str | Path,
    *,
    stats: Mapping[str, Any] | None = None,
    extractor: Mapping[str, Any] | None = None,
    inversion_npz: str | Path | None = None,
    inversion_sha256: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Writes the export of ``checkpoint`` (trained with the composed config ``cfg``) to ``out_dir``; returns it.

    ``stats`` defaults to ``cfg.stats_path``. ``extractor`` and ``inversion_sha256`` describe the feature extraction
    (the ``extractor`` block of the cache config and the sha256 of the inversion head); ``inversion_npz`` is copied to
    ``inversion/`` when given. ``metrics`` (see :func:`metrics_from_table`) goes into the model card. The export is
    reloaded with :func:`load_exported` and compared with the checkpoint before this function returns.
    """
    from safetensors.torch import save_file

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out} is not empty (pass overwrite=True to replace an earlier export)")
        clear_export_dir(out)
    out.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(checkpoint)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = module_state(ckpt["state_dict"])
    counters = ckpt.get("vocoder_counters", {})
    layer = str(cfg.speaker.layer)
    stats = stats_subset(load_stats(cfg.stats_path if stats is None else stats), layer)

    generator = OmegaConf.to_container(cfg.vocoder.generator, resolve=True)
    generator_class(generator["_target_"])
    inv_sha = inversion_sha256
    if inversion_npz is not None:
        actual = file_sha256(inversion_npz)
        if inv_sha is not None and actual != inv_sha:
            raise ValueError(f"{inversion_npz} has sha256 {actual}, expected {inv_sha}")
        inv_sha = actual
    config = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "vocoder": str(cfg.vocoder.name),
        "generator": generator,
        "speaker": {
            "layer": layer,
            "dropout": float(cfg.speaker.dropout),
            "in_dim": SPEAKER_RAW_DIM,
            "hidden_dim": SPEAKER_RAW_DIM,
            "out_dim": SPEAKER_DIM,
        },
        "sample_rate": SAMPLE_RATE,
        "frame_rate": FRAME_RATE,
        "hop_length": HOP,
        "inputs": input_contract(layer, extractor, inv_sha),
        "training": {
            "experiment": str(cfg.get("experiment_name", "")),
            "seed": int(cfg.seed) if "seed" in cfg else None,
            "g_step": int(counters["g_step"]) if "g_step" in counters else None,
            "max_g_steps": int(cfg.train.max_g_steps),
            "batch_size": int(cfg.data.batch_size),
            "crop_frames": int(cfg.data.crop_frames),
            "precision": str(cfg.trainer.trainer.precision) if "trainer" in cfg and "trainer" in cfg.trainer else None,
            "speaker_p_cross": float(cfg.speaker.p_cross),
            "mel_warmup_steps": int(cfg.train.mel_warmup_steps),
            "discriminators": sorted(str(k) for k in cfg.loss.discriminators),
            "loss": str(cfg.loss.get("name", "")),
        },
        "source": {
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": file_sha256(checkpoint),
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **dict(extra or {}),
        },
        "metrics": dict(metrics) if metrics else None,
    }
    if inversion_npz is not None:
        (out / INVERSION_DIR).mkdir(exist_ok=True)
        shutil.copy2(inversion_npz, out / INVERSION_DIR / Path(inversion_npz).name)
        config["inputs"]["inversion_head_file"] = f"{INVERSION_DIR}/{Path(inversion_npz).name}"
    save_file(weights, str(out / WEIGHTS_FILE), metadata={"format": FORMAT, "format_version": str(FORMAT_VERSION)})
    (out / CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n")
    (out / STATS_FILE).write_text(json.dumps(stats) + "\n")
    (out / CARD_FILE).write_text(render_model_card(config))

    reloaded = load_exported(out)
    reloaded_state = reloaded.state_dict()
    if set(reloaded_state) != set(weights):
        raise RuntimeError("the reloaded export has different keys from the checkpoint")
    for key, value in weights.items():
        if not torch.equal(reloaded_state[key].cpu(), value):
            raise RuntimeError(f"{key} differs between the checkpoint and the reloaded export")
    return out


# ---------------------------------------------------------------------------------------------------------- metrics


@dataclass(frozen=True)
class Metric:
    column: str
    label: str
    digits: int
    scale: float = 1.0
    unit: str = ""


CARD_METRICS = (
    Metric("utmos", "UTMOS", 3),
    Metric("pesq_wb", "PESQ (wideband)", 3),
    Metric("wer", "WER", 2, 100.0, " %"),
    Metric("cer", "CER", 2, 100.0, " %"),
    Metric("sim_same", "Speaker similarity to the same utterance (ECAPA)", 3),
    Metric("sim_spk", "Speaker similarity to the speaker's other utterances (ECAPA)", 3),
    Metric("mcd", "MCD", 2, 1.0, " dB"),
    Metric("mel_l1", "Mel L1", 4),
    Metric("f0_rmse_cents", "F0 RMSE", 1, 1.0, " cents"),
    Metric("vde", "Voicing decision error", 4),
    Metric("loud_db_rmse", "Loudness RMSE", 2, 1.0, " dB"),
    Metric("ema_r_mean", "EMA correlation after re-inversion", 4),
)


def metrics_from_table(table_csv: str | Path, system: str, reference: str | None = "gt", condition: str = "T1") -> dict:
    """Test-set metrics of ``system`` (and of ``reference``) from an evaluation ``table.csv``, stratum ``all``."""
    import csv

    with open(table_csv, newline="") as f:
        rows = {r["system"]: r for r in csv.DictReader(f) if r["condition"] == condition and r["stratum"] == "all"}
    if system not in rows:
        raise KeyError(f"{system} has no {condition} row in {table_csv}")

    def values(row: Mapping[str, str]) -> dict:
        out = {}
        for m in CARD_METRICS:
            if row.get(m.column) not in (None, "", "nan"):
                out[m.column] = {k: float(row[c]) for k, c in (("value", m.column), ("lo", f"{m.column}_lo"), ("hi", f"{m.column}_hi"))
                                 if row.get(c) not in (None, "", "nan")}
        return out

    out = {"condition": condition, "system": system, "n_utt": int(float(rows[system]["n_utt"])),
           "n_speakers": int(float(rows[system]["n_speakers"])), "values": values(rows[system])}
    if reference and reference in rows:
        out["reference"] = {"system": reference, "values": values(rows[reference])}
    return out


# ---------------------------------------------------------------------------------------------------------- model card


VOCODER_TITLES = {"hifigan": "HiFi-GAN", "ddsp": "DDSP (harmonic plus noise)", "vocos": "Vocos"}


def _fmt(v: Mapping[str, float] | None, m: Metric) -> str:
    if not v or "value" not in v:
        return "n/a"
    text = f"{v['value'] * m.scale:.{m.digits}f}{m.unit}"
    if "lo" in v and "hi" in v:
        text += f" ({v['lo'] * m.scale:.{m.digits}f} to {v['hi'] * m.scale:.{m.digits}f})"
    return text


def render_model_card(config: Mapping[str, Any], license_id: str = "mit", repo_id: str | None = None) -> str:
    """Markdown model card with Hub metadata for an export ``config``."""
    name = config["vocoder"]
    title = VOCODER_TITLES.get(name, name)
    tr, src, inputs = config["training"], config["source"], config["inputs"]
    ext = inputs.get("feature_extractor") or {}
    load_name = repo_id or "<repo-id or local export directory>"
    head_file = inputs.get("inversion_head_file")
    lines = [
        "---",
        f"license: {license_id}",
        "library_name: pytorch",
        "language:",
        "- en",
        "tags:",
        "- audio",
        "- speech",
        "- vocoder",
        "- articulatory",
        "- sparc",
        "datasets:",
        "- parler-tts/libritts_r_filtered",
        "---",
        "",
        f"# {title} vocoder for SPARC articulatory features (24 kHz)",
        "",
        f"A {title} generator that synthesizes 24 kHz speech from SPARC articulatory features: 12 EMA channels, pitch, "
        "loudness and periodicity at 50 frames per second, plus a speaker vector. It was trained on filtered "
        "LibriTTS-R as one of three vocoders (HiFi-GAN, DDSP and Vocos) compared under the same features, speaker "
        f"conditioning, data and recipe. The comparison is documented in [RESULTS.md]({RESULTS_URL}).",
        "",
        "## Model",
        "",
        f"- Generator: `{config['generator']['_target_']}`; the speaker FFN maps the 1024-d speaker vector to a 64-d "
        "embedding that conditions the generator through FiLM.",
        f"- Training: experiment `{tr.get('experiment')}`, seed {tr.get('seed')}, generator step {tr.get('g_step')} of "
        f"{tr.get('max_g_steps')}, batch {tr.get('batch_size')} crops of {tr.get('crop_frames')} frames, precision "
        f"{tr.get('precision')}; mel L1 loss with a mel-only warm-up of {tr.get('mel_warmup_steps')} steps, then hinge "
        f"adversarial and feature-matching losses from the discriminators {', '.join(tr.get('discriminators') or [])}; "
        "AdamW with cosine learning-rate decay to zero.",
        f"- Source checkpoint: `{src.get('checkpoint')}` (sha256 `{src.get('checkpoint_sha256')}`), fork commit "
        f"`{src.get('fork_commit', 'unknown')}`. Discriminators and optimizer states are not included.",
        "",
        "## Inputs",
        "",
        f"- `features`: float tensor `[batch, 15, frames]` at 50 Hz, channels {', '.join(inputs['features']['channels'])}.",
        f"  - EMA: {inputs['features']['ema']}.",
        f"  - f0: {inputs['features']['f0']}.",
        f"  - loudness: {inputs['features']['loudness']}",
        f"  - periodicity: {inputs['features']['periodicity']}.",
        f"- `spk_raw`: float tensor `[batch, 1024]`: {inputs['spk_raw']['definition']}.",
        f"- Output: waveform `[batch, 1, 480 * frames]` at {config['sample_rate']} Hz.",
        "",
        "The features must come from the same extractor as the training features: SPARC en+ "
        f"(`{ext.get('sparc_repo', 'cheoljun95/Speech-Articulatory-Coding')}` snapshot `{ext.get('sparc_snapshot', 'n/a')}`, "
        f"`{ext.get('sparc_ckpt', 'n/a')}`), WavLM-large (`{ext.get('wavlm_repo', 'microsoft/wavlm-large')}` snapshot "
        f"`{ext.get('wavlm_snapshot', 'n/a')}`) and the refit linear inversion head with sha256 "
        f"`{inputs.get('inversion_head_sha256')}`"
        + (f" (included as `{head_file}`)" if head_file else "")
        + f". The extractor is `SparcFeatureExtractor` in [the fork]({REPO_URL}) (`sparc.vocoders.features.extractor`); "
        "`load_feature_extractor` and `extraction_to_inputs` below build it and turn its output into the two inputs.",
        "",
        "## Usage",
        "",
        "Install the package from the fork (Python 3.13 or newer): `pip install git+" + REPO_URL + "@vocoder-comparison`.",
        "",
        "```python",
        "import librosa",
        "import torch",
    ]
    if head_file:
        lines += ["from huggingface_hub import hf_hub_download"]
    lines += [
        "from sparc.vocoders.export import extraction_to_inputs, load_exported, load_feature_extractor",
        "",
        f'vocoder = load_exported("{load_name}")  # Hub repo id or local export directory',
    ]
    if head_file:
        lines += [f'head = hf_hub_download("{load_name}", "{head_file}")  # or the file in a local export']
    else:
        lines += ['head = "<path to the refit inversion head npz>"']
    lines += [
        "extractor = load_feature_extractor(head)  # downloads the pinned SPARC en+ and WavLM-large snapshots once",
        'wav, _ = librosa.load("speech.wav", sr=24000, mono=True)',
        f'features, spk_raw = extraction_to_inputs(extractor.extract(wav, seed=0), speaker_layer="{config["speaker"]["layer"]}")',
        "with torch.no_grad():",
        "    out = vocoder(features, spk_raw)[0, 0].numpy()  # 24 kHz waveform at the level of the input",
        "```",
        "",
        "`features` is `[B, 15, T]` and `spk_raw` is `[B, 1024]`; edit them (for example F0 or loudness) before synthesis "
        "to change the output. `seed` fixes CREPE's pitch dither."
        + (" The DDSP generator draws noise from the global torch generator; seed it for reproducible output."
           if name == "ddsp" else ""),
        "",
    ]
    metrics = config.get("metrics")
    if metrics:
        ref = (metrics.get("reference") or {}).get("values", {})
        lines += [
            "## Evaluation (LibriTTS-R test-clean)",
            "",
            f"{metrics['n_utt']:,} utterances, {metrics['n_speakers']} speakers, condition {metrics['condition']} (speaker "
            "vector of the utterance itself), output level -3 dBFS. Mean with speaker-cluster bootstrap CI95; WER and CER "
            "are corpus-level (Whisper large-v3). Prosody metrics are measured on features re-extracted from the output. "
            f"Details and comparisons with the other vocoders: [RESULTS.md]({RESULTS_URL}). For the F0, voicing, "
            "loudness and EMA rows the ground-truth column is an extraction floor: the original audio re-extracted and "
            "compared with the stored input features.",
            "",
            "| metric | this model | ground truth |",
            "|---|---|---|",
        ]
        for m in CARD_METRICS:
            v = metrics["values"].get(m.column)
            if v is None:
                continue
            lines.append(f"| {m.label} | {_fmt(v, m)} | {_fmt(ref.get(m.column), m)} |")
        lines.append("")
    lines += [
        "## Limitations",
        "",
        "- English read speech only (LibriTTS-R); quality on other domains is untested.",
        "- One training run; the comparison in RESULTS.md gives the run-to-run variation of the 400k-step runs.",
        "- Edits of single EMA channels are followed weakly (RESULTS.md section 5.2); F0 and loudness edits are followed "
        "closely.",
        "- No human listening test was done; quality numbers come from UTMOS and other automatic metrics.",
        "",
        "## Citation",
        "",
        "The articulatory features come from SPARC:",
        "",
        "```bibtex",
        "@article{cho2024coding,",
        "  title={Coding Speech through Vocal Tract Kinematics},",
        "  author={Cho, Cheol Jun and Wu, Peter and Prabhune, Tejas S. and Agarwal, Dhruv and Anumanchipalli, Gopala K.},",
        "  journal={IEEE Journal of Selected Topics in Signal Processing},",
        "  volume={18}, number={8}, pages={1427--1440}, year={2024}",
        "}",
        "```",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------- upload


UPLOAD_PATTERNS = [WEIGHTS_FILE, CONFIG_FILE, STATS_FILE, CARD_FILE, VERIFICATION_FILE, f"{INVERSION_DIR}/*.npz"]


def upload_export(
    export_dir: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    commit_message: str | None = None,
    token: str | bool | None = None,
    api: Any = None,
) -> str:
    """Creates the model repository ``repo_id`` if needed and uploads the export files of ``export_dir`` in one commit.

    Only the files of :data:`UPLOAD_PATTERNS` are uploaded, whatever else the directory holds. Returns the commit URL.

    ``private`` applies only when the repository is created; an existing repository keeps its visibility. ``api`` is
    a :class:`huggingface_hub.HfApi` (tests pass a stand-in).
    """
    export_dir = Path(export_dir)
    for name in (CONFIG_FILE, STATS_FILE, WEIGHTS_FILE, CARD_FILE):
        if not (export_dir / name).is_file():
            raise FileNotFoundError(f"{export_dir / name} is missing; run the export first")
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    info = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(export_dir),
        allow_patterns=UPLOAD_PATTERNS,
        commit_message=commit_message or f"Upload {export_dir.name}",
    )
    return str(getattr(info, "commit_url", info))


def set_card_repo(export_dir: str | Path, repo_id: str, license_id: str) -> None:
    """Rewrites the model card of ``export_dir`` with the final repository id and licence."""
    export_dir = Path(export_dir)
    config = json.loads((export_dir / CONFIG_FILE).read_text())
    (export_dir / CARD_FILE).write_text(render_model_card(config, license_id=license_id, repo_id=repo_id))


def list_files(export_dir: str | Path) -> Sequence[tuple[str, int]]:
    """``(relative path, size in bytes)`` of every file of an export, sorted."""
    root = Path(export_dir)
    return sorted((str(p.relative_to(root)), p.stat().st_size) for p in root.rglob("*") if p.is_file())
