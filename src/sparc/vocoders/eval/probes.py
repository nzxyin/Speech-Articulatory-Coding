"""Controllability probes (EVALUATION.md section 6): edit the raw input features, synthesize, re-extract, compare.

For a fixed subset of test utterances and a list of edits of the 15 raw feature channels (F0 shifts on voiced frames,
loudness scaling, periodicity set to 0, each EMA channel moved by a multiple of its training std), a vocoder
synthesizes the unedited and the edited features (T1 speaker vectors, eval gain, the fixed synthesis RNG of
``predict_step``). Both waveforms go through the evaluation re-extraction (refit head, no CREPE dither) and the
edited one is compared with the unedited one:

- ``f0_shift_*``: median/mean of ``1200 log2(f_edit / f_base)`` over frames voiced in both (target ``100 k`` cents);
- ``loud_shift_db_*``: median/mean of ``20 log10((l_edit + eps) / (l_base + eps))`` of ``loud_raw`` over frames above
  the ``floor_percentile``-th percentile of the unedited loudness (target ``20 log10(scale)`` dB);
- ``voiced_frac_base``/``voiced_frac_edit``: fraction of frames with periodicity above 0;
- EMA edits: ``ema_gain = mean(d EMA_c) / (s std_c)`` (1 means the edit is reproduced) and ``ema_leakage =
  mean_{c' != c} mean|d EMA_c'| / std_c'``; ``ema_shift_all`` is the same leakage measure averaged over all 12 channels
  and is reported for every edit.

Resumable: every edit writes ``<out_dir>/edits/<edit>.parquet`` atomically and existing files are skipped; once all
edits exist ``<out_dir>/results.parquet`` (one row per utterance and edit) is written. ``stop()`` is checked between
edits; a stopped run leaves no ``results.parquet`` and can be called again.
"""

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from omegaconf import DictConfig, OmegaConf

from sparc.vocoders.constants import (
    EMA_NAMES,
    F0_CHANNEL,
    LOUDNESS_CHANNEL,
    LOUDNESS_EPS,
    N_EMA,
    PERIODICITY_CHANNEL,
    SAMPLE_RATE,
)
from sparc.vocoders.data.dataset import FullUtteranceDataset
from sparc.vocoders.models.frontend import load_stats
from sparc.vocoders.training.callbacks import EVAL_SEED, fixed_torch_rng

EDIT_TYPES = ("f0", "loudness", "periodicity", "ema")


def _tag(value: float) -> str:
    return f"{abs(value):g}"


def edit_name(edit: dict) -> str:
    """File-system-safe name of an edit, e.g. ``f0_p2``, ``loud_x0.5``, ``per_zero``, ``ema_TDX_m1``."""
    kind = edit["type"]
    if kind == "f0":
        k = float(edit["semitones"])
        return f"f0_{'p' if k >= 0 else 'm'}{_tag(k)}"
    if kind == "loudness":
        return f"loud_x{float(edit['scale']):g}"
    if kind == "periodicity":
        return "per_zero"
    if kind == "ema":
        s = float(edit["multiple"])
        return f"ema_{EMA_NAMES[int(edit['channel'])]}_{'p' if s >= 0 else 'm'}{_tag(s)}"
    raise ValueError(f"unknown edit type {kind!r}")


def probe_edits(cfg: DictConfig) -> list[dict]:
    """The edits of ``cfg.eval_features.probes`` in a fixed order, each a dict with a ``name`` key.

    Types: ``{"type": "f0", "semitones": k}``, ``{"type": "loudness", "scale": s}``, ``{"type": "periodicity",
    "value": 0.0}`` and ``{"type": "ema", "channel": c, "multiple": s}``.
    """
    spec = cfg.eval_features.probes
    edits: list[dict] = [{"type": "f0", "semitones": float(k)} for k in spec.f0_semitones]
    edits += [{"type": "loudness", "scale": float(s)} for s in spec.loudness_scales]
    if spec.periodicity_zero:
        edits.append({"type": "periodicity", "value": 0.0})
    for channel in range(N_EMA):
        edits += [{"type": "ema", "channel": channel, "multiple": float(s)} for s in spec.ema_std_multiples]
    for edit in edits:
        edit["name"] = edit_name(edit)
    names = [e["name"] for e in edits]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate edit names in {names}")
    return edits


def apply_edit(features: torch.Tensor, edit: dict, stats: dict) -> torch.Tensor:
    """Returns an edited copy of raw features ``[B, 15, T]``; ``features`` is never modified.

    - ``f0``: channel 12 times ``2 ** (k / 12)`` on frames whose periodicity (channel 14) is above 0;
    - ``loudness``: channel 13 times ``scale``;
    - ``periodicity``: channel 14 set to ``value`` (0) on every frame;
    - ``ema``: channel ``c`` plus ``multiple`` times the training std of EMA channel ``c`` (``stats["ema_std"][c]``).
    Every other channel and frame is returned unchanged.
    """
    if features.dim() != 3 or features.shape[1] != 15:
        raise ValueError(f"expected features of shape [B, 15, T], got {tuple(features.shape)}")
    out = features.clone()
    kind = edit["type"]
    if kind == "f0":
        voiced = features[:, PERIODICITY_CHANNEL] > 0
        factor = 2.0 ** (float(edit["semitones"]) / 12.0)
        out[:, F0_CHANNEL] = torch.where(voiced, features[:, F0_CHANNEL] * factor, features[:, F0_CHANNEL])
    elif kind == "loudness":
        out[:, LOUDNESS_CHANNEL] = features[:, LOUDNESS_CHANNEL] * float(edit["scale"])
    elif kind == "periodicity":
        out[:, PERIODICITY_CHANNEL] = float(edit.get("value", 0.0))
    elif kind == "ema":
        channel = int(edit["channel"])
        if not 0 <= channel < N_EMA:
            raise ValueError(f"EMA channel must be in [0, {N_EMA}), got {channel}")
        std = float(np.asarray(stats["ema_std"], dtype=np.float64)[channel])
        out[:, channel] = features[:, channel] + float(edit["multiple"]) * std
    else:
        raise ValueError(f"unknown edit type {kind!r}")
    return out


def probe_subset(cfg: DictConfig, split: str = "test.clean") -> list[str]:
    """Ids of the probe utterances: ``n_utterances`` of ``duration_range_s`` with at most ``max_per_speaker`` per speaker.

    From the packed index of ``split``, utterances within the duration range are sorted by id, permuted with
    ``default_rng(seed)`` and taken in that order while their speaker has fewer than ``max_per_speaker`` chosen; the
    result is sorted by id. Deterministic.
    """
    spec = cfg.eval_features.probes
    index = pd.read_parquet(Path(cfg.paths.cache_root) / "packed" / split / "index.parquet", columns=["id", "speaker", "n24"])
    index = index.assign(id=index["id"].astype(str), speaker=index["speaker"].astype(str)).sort_values("id")
    low, high = (float(d) for d in spec.duration_range_s)
    seconds = index["n24"].to_numpy(dtype=np.float64) / SAMPLE_RATE
    index = index[(seconds >= low) & (seconds <= high)]
    ids, speakers = index["id"].tolist(), index["speaker"].tolist()
    order = np.random.default_rng(int(spec.seed)).permutation(len(ids))
    chosen: list[str] = []
    count: dict[str, int] = {}
    for i in order:
        if count.get(speakers[i], 0) >= int(spec.max_per_speaker):
            continue
        chosen.append(ids[i])
        count[speakers[i]] = count.get(speakers[i], 0) + 1
        if len(chosen) == int(spec.n_utterances):
            break
    return sorted(chosen)


def measure_edit(
    base_feats: np.ndarray,
    base_loud: np.ndarray,
    edit_feats: np.ndarray,
    edit_loud: np.ndarray,
    edit: dict,
    ema_std: np.ndarray,
    floor_percentile: float = 5.0,
    loud_eps: float = LOUDNESS_EPS,
) -> dict[str, float]:
    """Row of metrics comparing the re-extraction of an edited synthesis with that of the unedited one (pure numpy)."""
    n = min(len(base_feats), len(edit_feats), len(base_loud), len(edit_loud))
    bf, ef = base_feats[:n].astype(np.float64), edit_feats[:n].astype(np.float64)
    bl, el = base_loud[:n].astype(np.float64), edit_loud[:n].astype(np.float64)
    ema_std = np.asarray(ema_std, dtype=np.float64)
    nan = float("nan")
    base_voiced, edit_voiced = bf[:, PERIODICITY_CHANNEL] > 0, ef[:, PERIODICITY_CHANNEL] > 0
    both = base_voiced & edit_voiced
    row: dict[str, float] = {"n_frames": float(n), "n_voiced_both": float(both.sum())}
    row["voiced_frac_base"] = float(base_voiced.mean()) if n else nan
    row["voiced_frac_edit"] = float(edit_voiced.mean()) if n else nan
    if both.any():
        cents = 1200.0 * np.log2(ef[both, F0_CHANNEL] / bf[both, F0_CHANNEL])
        row["f0_shift_median_cents"] = float(np.median(cents))
        row["f0_shift_mean_cents"] = float(np.mean(cents))
    else:
        row["f0_shift_median_cents"] = row["f0_shift_mean_cents"] = nan
    keep = bl > np.percentile(bl, floor_percentile) if n else np.zeros(0, dtype=bool)
    if keep.any():
        db = 20.0 * np.log10((el[keep] + loud_eps) / (bl[keep] + loud_eps))
        row["loud_shift_db_median"] = float(np.median(db))
        row["loud_shift_db_mean"] = float(np.mean(db))
    else:
        row["loud_shift_db_median"] = row["loud_shift_db_mean"] = nan
    delta = ef[:, :N_EMA] - bf[:, :N_EMA]  # (n, 12)
    mean_abs = np.abs(delta).mean(0) / ema_std if n else np.full(N_EMA, nan)
    row["ema_shift_all"] = float(mean_abs.mean())
    row["ema_gain"] = row["ema_leakage"] = nan
    if edit["type"] == "ema" and n:
        c, s = int(edit["channel"]), float(edit["multiple"])
        row["ema_gain"] = float(delta[:, c].mean() / (s * ema_std[c]))
        others = np.delete(np.arange(N_EMA), c)
        row["ema_leakage"] = float(mean_abs[others].mean())
    return row


def edit_target(edit: dict, ema_std: np.ndarray) -> float:
    """Ideal measured effect of an edit: cents (F0), dB (loudness), voiced fraction 0 (periodicity), EMA units."""
    kind = edit["type"]
    if kind == "f0":
        return 100.0 * float(edit["semitones"])
    if kind == "loudness":
        return float(20.0 * np.log10(float(edit["scale"])))
    if kind == "periodicity":
        return 0.0
    return float(edit["multiple"]) * float(np.asarray(ema_std)[int(edit["channel"])])


def probe_batches(vcfg: DictConfig, ids: Sequence[str], device: torch.device | str, split: str) -> list[dict]:
    """T1 batches (batch size 1, eval gain) of ``ids`` from ``FullUtteranceDataset``, on ``device``.

    Each dict has ``id``, ``features [1, 15, T]``, ``spk_raw [1, 1024]``, ``condition`` and ``ref_id`` in the form the
    training module's ``synthesize`` expects. ``vcfg`` is the composed vocoder config (cache paths, eval gain and
    speaker layer come from it).
    """
    dataset = FullUtteranceDataset(
        Path(vcfg.paths.cache_root) / "packed" / split,
        ids=list(ids),
        gain_db=vcfg.data.eval_gain_db,
        condition="T1",
        speaker_layer=vcfg.speaker.layer,
        ref_min_dur=vcfg.data.ref_min_dur,
    )
    device = torch.device(device)
    batches = []
    for i in range(len(dataset)):
        item = dataset[i]
        batches.append(
            {
                "id": item["id"],
                "condition": item["condition"],
                "ref_id": item["ref_id"],
                "features": item["features"][None].to(device),
                "spk_raw": item["spk_raw"][None].to(device),
            }
        )
    return batches


def synthesize(module, features: torch.Tensor, spk_raw: torch.Tensor, device: torch.device) -> np.ndarray:
    """Mono float32 waveform ``[480 T]`` as ``predict_step`` produces it (``synthesize`` under the fixed eval RNG)."""
    with torch.no_grad(), fixed_torch_rng(device, EVAL_SEED):
        wav = module.synthesize({"features": features, "spk_raw": spk_raw})
    return wav[0, 0].float().cpu().numpy()


def _atomic_write_bytes(path: Path, writer: Callable[[Path], None]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    writer(tmp)
    os.replace(tmp, path)


def _write_wav(path: Path, wav: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(path, lambda tmp: sf.write(str(tmp), wav, SAMPLE_RATE, format="WAV", subtype="FLOAT"))


def _write_parquet(path: Path, table: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(path, lambda tmp: table.to_parquet(tmp, index=False))


def run_probes(
    cfg: DictConfig, system: str, device: torch.device | str, out_dir: Path, stop: Callable[[], bool] = lambda: False
) -> None:
    """Runs (or resumes) the probes of vocoder ``system`` and writes them under ``out_dir`` (see the module docstring).

    ``cfg`` is the full evaluation config: ``cfg.eval.systems[system]`` supplies ``experiment``, ``vocoder`` and the
    optional ``ckpt`` and ``overrides`` of :func:`~sparc.vocoders.eval.vocoder_loader.load_vocoder`;
    ``cfg.eval_features.probes`` the subset and the edits; ``cfg.paths.cache_root`` the packed features.
    """
    from sparc.vocoders.eval.reextract import ReExtractor
    from sparc.vocoders.eval.vocoder_loader import checkpoint_info, load_vocoder

    out_dir = Path(out_dir)
    spec = cfg.eval_features.probes
    edits = probe_edits(cfg)
    edit_dir = out_dir / str(spec.edit_dir)
    edit_dir.mkdir(parents=True, exist_ok=True)
    todo = [e for e in edits if not (edit_dir / f"{e['name']}.parquet").exists()]
    results_path = out_dir / str(spec.results_name)
    if not todo:
        if not results_path.exists():
            _write_parquet(results_path, _concat_edits(edit_dir, edits))
        return
    device = torch.device(device)
    sys_spec = cfg.eval.systems[system]
    module, vcfg = load_vocoder(
        sys_spec.experiment,
        sys_spec.vocoder,
        sys_spec.get("ckpt", None),
        device,
        list(sys_spec.get("overrides", []) or []),
    )
    stats = load_stats(vcfg.stats_path)
    ema_std = np.asarray(stats["ema_std"], dtype=np.float64)
    ids = probe_subset(cfg, spec.split)
    batches = probe_batches(vcfg, ids, device, spec.split)
    extractor = ReExtractor(cfg, spec.head, device)
    _atomic_write_bytes(
        out_dir / "probes_meta.json",
        lambda tmp: tmp.write_text(
            json.dumps(
                {
                    "system": system,
                    "checkpoint": checkpoint_info(module),
                    "split": spec.split,
                    "ids": ids,
                    "edits": [e["name"] for e in edits],
                    "probes": OmegaConf.to_container(spec, resolve=True),
                    "reextract": extractor.describe(),
                },
                indent=1,
                default=str,
            )
        ),
    )

    def reextract(wav: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = extractor.extract(wav, SAMPLE_RATE)
        return out["feats"], out["loud_raw"]

    base, examples = [], int(spec.example_utterances)
    for i, batch in enumerate(batches):
        wav = synthesize(module, batch["features"], batch["spk_raw"], device)
        base.append(reextract(wav))
        if i < examples:
            _write_wav(out_dir / "wav" / batch["id"] / "unedited.wav", wav)
    for edit in todo:
        if stop():
            return
        rows = []
        for i, batch in enumerate(batches):
            edited = apply_edit(batch["features"], edit, stats)
            wav = synthesize(module, edited, batch["spk_raw"], device)
            feats, loud = reextract(wav)
            row = measure_edit(
                base[i][0], base[i][1], feats, loud, edit, ema_std, float(spec.floor_percentile), float(spec.loud_eps)
            )
            rows.append(
                {
                    "id": batch["id"],
                    "edit": edit["name"],
                    "type": edit["type"],
                    "channel": int(edit.get("channel", -1)),
                    "param": float(edit.get("semitones", edit.get("scale", edit.get("multiple", edit.get("value", 0.0))))),
                    "target": edit_target(edit, ema_std),
                    **row,
                }
            )
            if i < examples:
                _write_wav(out_dir / "wav" / batch["id"] / f"{edit['name']}.wav", wav)
        _write_parquet(edit_dir / f"{edit['name']}.parquet", pd.DataFrame(rows))
    if all((edit_dir / f"{e['name']}.parquet").exists() for e in edits):
        _write_parquet(results_path, _concat_edits(edit_dir, edits))


def _concat_edits(edit_dir: Path, edits: Sequence[dict]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(edit_dir / f"{e['name']}.parquet") for e in edits], ignore_index=True)
