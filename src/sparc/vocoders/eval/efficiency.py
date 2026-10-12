"""Efficiency of the synthesis paths (EVALUATION.md section 7): parameters, real-time factor, receptive field.

Parameters: generator, speaker FFN and total (the frontend holds buffers only). The generator count is given as stored
(weight-normalized convolutions hold a gain ``g`` next to the direction ``v``) and folded (``g`` dropped, the count of an
exported inference model).

Real-time factor ``RTF = compute time / audio duration`` with batch size 1 in fp32, five warm-up calls, device
synchronization around every call and the data already on the device. ``measure_rtf`` times
``VocoderGANModule.synthesize``, i.e. the speaker FFN and the generator including its frontend, not feature extraction;
the extraction cost is reported once, separately, as the system ``extractor`` of :func:`run_efficiency`.

Receptive field and lookahead, measured: a real 400-frame input (a centred window of a long test utterance) is
synthesized twice with the same RNG state, the second time with ``std_multiple`` (0.5) training stds added to frame
``frame`` (200). The perturbation is applied to the normalized network input: EMA channels get ``+0.5 std_c``, F0 and
loudness are multiplied/shifted in the log domain (``ln f0 + 0.5 logf0_std``, ``ln(l + 1e-4) + 0.5 loud_log_std``) and
periodicity gets ``+0.5 per_std``. The first and last output samples whose absolute change exceeds ``1e-4 max|y|``
give, with the frame occupying samples ``[hop * frame, hop * (frame + 1))``,

    lookahead = (hop * frame - first) / sr           (how far before the frame the output already reacts)
    past      = (last - hop * (frame + 1)) / sr      (how long after the frame the output still reacts)

(negative values mean the output starts reacting after the frame starts). The measurement is made for all channels
perturbed together and also for the channel groups EMA, F0, loudness and periodicity separately; for DDSP the F0
perturbation changes the integrated phase of every later sample, so its ``past`` reaches the end of the input.
``analytic_receptive_field`` states the receptive field implied by the layer sizes (left and right extent in samples,
for the architecture's convolutions, up-samplers, windows and filters; a sum of per-layer extents, hence an upper bound
on the dependency range, and the measured values are smaller because contributions fade below the threshold).
"""

import json
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch import nn

from sparc.vocoders.constants import (
    F0_CHANNEL,
    HOP,
    LOUDNESS_CHANNEL,
    LOUDNESS_EPS,
    N_EMA,
    PERIODICITY_CHANNEL,
    SAMPLE_RATE,
)
from sparc.vocoders.models.frontend import load_stats
from sparc.vocoders.training.callbacks import EVAL_SEED, fixed_torch_rng

CHANNEL_GROUPS = {
    "all": tuple(range(15)),
    "ema": tuple(range(N_EMA)),
    "f0": (F0_CHANNEL,),
    "loudness": (LOUDNESS_CHANNEL,),
    "periodicity": (PERIODICITY_CHANNEL,),
}
REFERENCE_SYSTEMS = ("vocos_mel", "enplus16")
EXTRACTOR_SYSTEM = "extractor"


# ---------------------------------------------------------------------------------------------------------------------
# parameters


def _is_weight_norm_gain(name: str) -> bool:
    return name.endswith("parametrizations.weight.original0") or name == "weight_g" or name.endswith(".weight_g")


def count_parameters(module: nn.Module, folded: bool = False) -> int:
    """Number of parameters of ``module``; with ``folded`` the weight-norm gains are left out."""
    return sum(p.numel() for name, p in module.named_parameters() if not (folded and _is_weight_norm_gain(name)))


def param_counts(module) -> dict[str, int]:
    """Parameter counts of a ``VocoderGANModule``-like object with ``generator`` and ``speaker`` submodules.

    Keys: ``generator``, ``generator_folded`` (weight-norm gains dropped), ``speaker_ffn``, ``total`` (generator plus
    speaker FFN), ``total_folded`` and ``frontend_buffers`` (elements of the frontend's statistics buffers; they are
    not parameters and not counted in ``total``). Discriminators are not part of the synthesis path.
    """
    generator, speaker = module.generator, module.speaker
    counts = {
        "generator": count_parameters(generator),
        "generator_folded": count_parameters(generator, folded=True),
        "speaker_ffn": count_parameters(speaker),
    }
    counts["total"] = counts["generator"] + counts["speaker_ffn"]
    counts["total_folded"] = counts["generator_folded"] + counts["speaker_ffn"]
    frontend = getattr(generator, "frontend", None)
    counts["frontend_buffers"] = sum(b.numel() for b in frontend.buffers()) if frontend is not None else 0
    return counts


# ---------------------------------------------------------------------------------------------------------------------
# real-time factor


def time_calls(
    calls: Sequence[Callable[[], object]],
    durations_s: Sequence[float],
    device: torch.device | str,
    threads: int | None = None,
    warmup: int = 5,
) -> dict:
    """Times each call once after ``warmup`` untimed calls; ``durations_s[i]`` is the audio duration of ``calls[i]``.

    ``threads`` sets ``torch.set_num_threads`` for the measurement (restored afterwards). On CUDA every call is
    bracketed by ``torch.cuda.synchronize``. Returns total and per-utterance real-time factors.
    """
    if len(calls) != len(durations_s) or not calls:
        raise ValueError("need one duration per call and at least one call")
    device = torch.device(device)
    cuda = device.type == "cuda"
    previous = torch.get_num_threads()
    if threads:
        torch.set_num_threads(int(threads))
    try:
        for i in range(int(warmup)):
            calls[i % len(calls)]()
        if cuda:
            torch.cuda.synchronize(device)
        times = []
        for call in calls:
            if cuda:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            call()
            if cuda:
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - start)
        used_threads = torch.get_num_threads()
    finally:
        if threads:  # re-setting the thread count when it was not changed still breaks later DataLoader workers
            torch.set_num_threads(previous)
    times_a, audio_a = np.asarray(times), np.asarray(durations_s, dtype=np.float64)
    per_utt = times_a / audio_a
    return {
        "device": "cuda" if cuda else "cpu",
        "gpu": torch.cuda.get_device_name(device) if cuda else None,
        "threads": None if cuda else used_threads,
        "n": len(calls),
        "warmup_calls": int(warmup),
        "audio_s": float(audio_a.sum()),
        "time_s": float(times_a.sum()),
        "rtf": float(times_a.sum() / audio_a.sum()),
        "rtf_mean": float(per_utt.mean()),
        "rtf_median": float(np.median(per_utt)),
        "rtf_p95": float(np.percentile(per_utt, 95)),
        "rtf_max": float(per_utt.max()),
    }


def measure_rtf(module, batches: list[dict], device: torch.device | str, threads: int | None = None, warmup: int = 5) -> dict:
    """Real-time factor of ``module.synthesize`` over ``batches`` (dicts with ``features [1, 15, T]``, ``spk_raw``).

    The module is switched to eval mode for the measurement and restored; batches must already be on ``device``.
    """
    was_training = module.training
    module.eval()
    try:

        def make(batch: dict) -> Callable[[], object]:
            def call():
                with torch.no_grad():
                    return module.synthesize(batch)

            return call

        durations = [batch["features"].shape[-1] * HOP / SAMPLE_RATE for batch in batches]
        return time_calls([make(b) for b in batches], durations, device, threads, warmup)
    finally:
        module.train(was_training)


# ---------------------------------------------------------------------------------------------------------------------
# receptive field


def extent_from_outputs(
    y0: np.ndarray, y1: np.ndarray, frame: int, hop: int, sample_rate: int, threshold: float = 1e-4
) -> dict:
    """First and last sample whose change exceeds ``threshold * max|y0|`` and the lookahead/past derived from them."""
    y0, y1 = np.asarray(y0, dtype=np.float64).ravel(), np.asarray(y1, dtype=np.float64).ravel()
    n = min(len(y0), len(y1))
    change = np.abs(y1[:n] - y0[:n])
    limit = threshold * float(np.abs(y0[:n]).max())
    hit = np.flatnonzero(change > limit)
    result = {
        "n_samples": int(n),
        "threshold": float(limit),
        "max_change": float(change.max()) if n else float("nan"),
        "first_sample": None,
        "last_sample": None,
        "lookahead_s": float("nan"),
        "past_s": float("nan"),
        "span_s": float("nan"),
        "reaches_start": False,
        "reaches_end": False,
    }
    if len(hit):
        first, last = int(hit[0]), int(hit[-1])
        result.update(
            first_sample=first,
            last_sample=last,
            lookahead_s=(hop * frame - first) / sample_rate,
            past_s=(last - hop * (frame + 1)) / sample_rate,
            span_s=(last - first + 1) / sample_rate,
            reaches_start=first == 0,
            reaches_end=last == n - 1,
        )
    return result


def perturb_features(features: torch.Tensor, frame: int, multiple: float, stats: dict, channels: Sequence[int]) -> torch.Tensor:
    """Copy of raw features ``[B, 15, T]`` with ``multiple`` training stds added to ``frame`` of ``channels``.

    The shift is applied in the domain the frontend normalizes in: linear for EMA and periodicity, log for F0
    (``ln f0 + multiple logf0_std``) and for loudness (``ln(l + 1e-4) + multiple loud_log_std``).
    """
    out = features.clone()
    ema_std = torch.as_tensor(stats["ema_std"], dtype=features.dtype, device=features.device)
    for c in channels:
        value = features[:, c, frame]
        if c < N_EMA:
            out[:, c, frame] = value + multiple * ema_std[c]
        elif c == F0_CHANNEL:
            out[:, c, frame] = value * float(np.exp(multiple * float(stats["logf0_std"])))
        elif c == LOUDNESS_CHANNEL:
            shifted = torch.log(value.clamp_min(0.0) + LOUDNESS_EPS) + multiple * float(stats["loud_log_std"])
            out[:, c, frame] = torch.exp(shifted) - LOUDNESS_EPS
        else:
            out[:, c, frame] = value + multiple * float(stats["per_std"])
    return out


def receptive_field(
    module,
    features: torch.Tensor,
    spk_raw: torch.Tensor,
    stats: dict,
    frame: int = 200,
    std_multiple: float = 0.5,
    threshold: float = 1e-4,
    groups: Sequence[str] = tuple(CHANNEL_GROUPS),
) -> dict:
    """Measured receptive field of ``module.synthesize`` around ``frame`` of ``features [1, 15, T]`` (module docstring).

    Returns ``{group: extent dict}`` for each channel group in ``groups`` (``all`` is the headline number) plus
    ``frame``, ``frames``, ``hop``, ``sample_rate`` and ``std_multiple``. Both syntheses run under the same fixed RNG
    state; ``features`` is not modified.
    """
    device = features.device
    frames = features.shape[-1]
    if not 0 <= frame < frames:
        raise ValueError(f"frame {frame} outside the {frames} input frames")
    was_training = module.training
    module.eval()

    def synth(x: torch.Tensor) -> np.ndarray:
        with torch.no_grad(), fixed_torch_rng(device, EVAL_SEED):
            return module.synthesize({"features": x, "spk_raw": spk_raw})[0, 0].float().cpu().numpy()

    try:
        base = synth(features)
        result: dict = {
            "frame": int(frame),
            "frames": int(frames),
            "hop": HOP,
            "sample_rate": SAMPLE_RATE,
            "std_multiple": float(std_multiple),
        }
        for name in groups:
            moved = synth(perturb_features(features, frame, std_multiple, stats, CHANNEL_GROUPS[name]))
            result[name] = extent_from_outputs(base, moved, frame, HOP, SAMPLE_RATE, threshold)
    finally:
        module.train(was_training)
    return result


def _convs(module: nn.Module) -> list[nn.Conv1d]:
    return [m for m in module.modules() if isinstance(m, nn.Conv1d)]


def _half(conv: nn.Conv1d) -> int:
    """Half extent of a same-padded convolution in input steps."""
    return (conv.kernel_size[0] - 1) // 2 * conv.dilation[0]


def hifigan_like_extent(input_conv: nn.Module, upsamples, blocks, num_blocks: int, output_conv: nn.Module, hop: int) -> tuple[float, float]:
    """Left and right extent in output samples of a HiFi-GAN generator (ours or SPARC's), duck-typed on its layers.

    ``hop`` is the number of output samples per input frame; a frame occupies ``[0, hop)``. A transposed convolution
    with kernel ``k``, stride ``s`` and padding ``p`` reaches ``p`` output steps to the left and ``k - s - p`` to the
    right of the input cell; a residual block adds the half extents of all its convolutions (the parallel blocks of a
    stage contribute their maximum).
    """
    step = float(hop)
    left = right = sum(_half(c) for c in _convs(input_conv)) * step
    for i, up in enumerate(upsamples):
        convt = next(m for m in up.modules() if isinstance(m, nn.ConvTranspose1d))
        s, k, p = convt.stride[0], convt.kernel_size[0], convt.padding[0]
        step /= s
        left += p * step
        right += max(k - s - p, 0) * step
        stage = [blocks[i * num_blocks + j] for j in range(num_blocks)]
        extra = max(sum(_half(c) for c in _convs(block)) for block in stage) * step
        left += extra
        right += extra
    tail = sum(_half(c) for c in _convs(output_conv)) * step
    return left + tail, right + tail


def analytic_receptive_field(generator: nn.Module) -> dict | None:
    """Receptive field implied by the architecture of one of the three vocoders, or ``None`` for an unknown module.

    ``left_samples`` / ``right_samples`` are the distances (24 kHz samples) from the edges of an input frame's cell
    ``[480 t, 480 (t + 1))`` to the furthest output samples that depend on it; ``lookahead_s = left / 24000`` and
    ``past_s = right / 24000`` correspond to the measured quantities. ``unbounded_right`` marks a path (the DDSP phase
    integral) that influences every later sample.
    """
    name = type(generator).__name__
    notes: list[str] = []
    unbounded = False
    if name == "HiFiGANVocoder":
        left, right = hifigan_like_extent(
            generator.input_conv, generator.upsamples, generator.blocks, generator.num_blocks, generator.output_conv, HOP
        )
    elif name == "VocosVocoder":
        hop_b = HOP // generator.upsample
        interp = HOP // 2 if generator.upsample > 1 else 0
        backbone = generator.backbone
        convs = [backbone.embed] + [block.dwconv for block in backbone.blocks]
        istft = generator.head.istft
        pad = (istft.n_fft - istft.hop_length) // 2
        left = right = interp + sum(_half(c) for c in convs) * hop_b + pad
        notes.append(f"backbone at {HOP // hop_b * 50} Hz, iSTFT n_fft {istft.n_fft}")
    elif name == "DDSPVocoder":
        step = float(HOP)
        left = right = _half(generator.in_conv) * step
        for block in generator.trunk:
            left += sum(_half(c) for c in _convs(block)) * step
        left += _half(generator.out_conv) * step
        right = left
        for stage in generator.up:
            step /= 2
            left += 1 * step  # linear x2 sampling at i / 2 reaches one fine step back
            extra = (_half(stage.conv) + sum(_half(c) for c in _convs(stage.res))) * step
            left += extra
            right += extra
        hop_c = generator.frame_hop
        taps = generator.post_filter.numel()
        fir = generator.fir_window.numel()
        synthesis = max(hop_c, (fir - 1) // 2 + hop_c) + (taps - 1) // 2
        left += synthesis
        right += synthesis
        unbounded = True
        notes.append("the fundamental phase is a cumulative sum of F0: an F0 change shifts every later harmonic sample")
    else:
        return None
    return {
        "architecture": name,
        "left_samples": float(left),
        "right_samples": float(right),
        "lookahead_s": float(left) / SAMPLE_RATE,
        "past_s": float(right) / SAMPLE_RATE,
        "unbounded_right": unbounded,
        "notes": notes,
    }


# ---------------------------------------------------------------------------------------------------------------------
# reference systems


def _normal_perturbation(values: np.ndarray, frame: int, multiple: float) -> np.ndarray:
    """``values [T, C]`` with ``multiple`` per-channel stds (over time) added to row ``frame``."""
    out = values.copy()
    out[frame] += multiple * values.std(0)
    return out


def vocos_mel_receptive_field(ref, wav24: np.ndarray, frame: int = 200, frames: int = 400, std_multiple: float = 0.5, threshold: float = 1e-4) -> dict:
    """Receptive field of Vocos mel: perturb mel frame ``frame`` of a real ``frames``-frame mel (own input representation).

    The mel is computed from the centred ``frames * hop`` samples of ``wav24``; ``std_multiple`` times each mel
    channel's std over time is added to frame ``frame`` and both mels are decoded. A mel frame is hop-based, so the
    frame occupies ``[256 frame, 256 (frame + 1))`` for the lookahead/past formulas (its analysis window is centred on
    the frame start, so the centre convention differs by half a hop, 128 samples).
    """
    hop = ref.hop
    need = frames * hop
    start = max((len(wav24) - need) // 2, 0)
    mel = ref.mel(wav24[start : start + need])[0].cpu().numpy().T  # (frames + 1, 100)
    mel = mel[:frames]
    moved = _normal_perturbation(mel, frame, std_multiple)
    to_tensor = lambda m: torch.from_numpy(np.ascontiguousarray(m.T[None])).float().to(ref.device)
    y0, y1 = ref.decode(to_tensor(mel)), ref.decode(to_tensor(moved))
    return {
        "frame": frame,
        "frames": frames,
        "hop": hop,
        "sample_rate": SAMPLE_RATE,
        "std_multiple": std_multiple,
        "all": extent_from_outputs(y0, y1, frame, hop, SAMPLE_RATE, threshold),
    }


def enplus_receptive_field(ref, wav24: np.ndarray, frame: int = 200, frames: int = 400, std_multiple: float = 0.5, threshold: float = 1e-4) -> dict:
    """Receptive field of the shipped en+ generator: perturb frame ``frame`` of its 14-channel input (own representation).

    The input is the real encoding of ``wav24`` (EMA from the shipped head, pitch, z-scored loudness), a centred window
    of ``frames`` frames; every channel of ``frame`` gets ``std_multiple`` times its std over the window. The frame
    occupies ``[320 frame, 320 (frame + 1))`` samples at 16 kHz.
    """
    code = ref.encode(wav24)
    spk = ref.spk_emb(code)
    n = min(len(code["ema"]), len(code["pitch"]), len(code["loudness"]))
    if n < frames:
        raise ValueError(f"the utterance gives {n} en+ frames, need {frames}")
    start = (n - frames) // 2
    window = np.concatenate([code["ema"][:n], code["pitch"][:n], code["loudness"][:n]], axis=1)[start : start + frames]
    moved = _normal_perturbation(window, frame, std_multiple)

    def decode(values: np.ndarray) -> np.ndarray:
        part = {"ema": values[:, :N_EMA], "pitch": values[:, N_EMA : N_EMA + 1], "loudness": values[:, N_EMA + 1 :]}
        return ref.decode_raw(part, spk)

    y0, y1 = decode(window), decode(moved)
    return {
        "frame": frame,
        "frames": frames,
        "hop": ref.hop,
        "sample_rate": ref.sample_rate,
        "std_multiple": std_multiple,
        "all": extent_from_outputs(y0, y1, frame, ref.hop, ref.sample_rate, threshold),
    }


def reference_analytic(system: str, ref) -> dict:
    """Analytic extents (samples of the system's own rate) of the Vocos mel and the shipped en+ generators."""
    if system == "vocos_mel":
        backbone, istft = ref.model.backbone, ref.model.head.istft
        convs = [backbone.embed] + [m.dwconv for m in backbone.convnext]
        left = right = sum(_half(c) for c in convs) * ref.hop + (istft.n_fft - istft.hop_length) // 2
        rate, note = SAMPLE_RATE, "mel frames at hop 256: backbone convolutions plus the iSTFT window"
    else:
        gen = ref.coder.generator
        left, right = hifigan_like_extent(gen.input_conv, gen.upsamples, gen.blocks, gen.num_blocks, gen.output_conv, ref.hop)
        rate, note = ref.sample_rate, "shipped HiFi-GAN generator at 16 kHz"
    return {
        "left_samples": float(left),
        "right_samples": float(right),
        "lookahead_s": float(left) / rate,
        "past_s": float(right) / rate,
        "notes": [note],
    }


# ---------------------------------------------------------------------------------------------------------------------
# driver


def _merge_json(path: Path, update: dict) -> dict:
    """Merges ``update`` (nested dicts merged, other values replaced) into the JSON at ``path`` and rewrites it atomically."""
    data = json.loads(path.read_text()) if path.exists() else {}

    def merge(into: dict, new: dict) -> None:
        for key, value in new.items():
            if isinstance(value, dict) and isinstance(into.get(key), dict):
                merge(into[key], value)
            else:
                into[key] = value

    merge(data, update)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str))
    os.replace(tmp, path)
    return data


def rtf_keys(device: torch.device, cpu_threads: Sequence[int]) -> list[tuple[str, int | None]]:
    """Result keys and thread counts to measure on ``device``: ``gpu`` on CUDA, ``cpu_<n>_threads`` on CPU."""
    if device.type == "cuda":
        return [("gpu", None)]
    return [(f"cpu_{int(n)}_threads", int(n)) for n in cpu_threads]


def _long_utterance(cfg: DictConfig, split: str, frames: int) -> str:
    """First id (sorted) of ``split`` with at least ``frames + 10`` feature frames: source of the receptive-field input."""
    index = pd.read_parquet(Path(cfg.paths.cache_root) / "packed" / split / "index.parquet", columns=["id", "T"])
    index = index.assign(id=index["id"].astype(str)).sort_values("id")
    long = index[index["T"] >= frames + 10]
    if long.empty:
        raise ValueError(f"no utterance of {split} has {frames + 10} frames")
    return str(long["id"].iloc[0])


def _gt_audio(vcfg: DictConfig, ids: Sequence[str], split: str) -> list[np.ndarray]:
    """Eval-gain gt audio (``480 T`` samples, float32) of ``ids`` as ``FullUtteranceDataset`` builds it."""
    from sparc.vocoders.data.dataset import FullUtteranceDataset

    dataset = FullUtteranceDataset(
        Path(vcfg.paths.cache_root) / "packed" / split, ids=list(ids), gain_db=vcfg.data.eval_gain_db, condition="T1"
    )
    return [dataset[i]["audio"][0].numpy().astype(np.float32) for i in range(len(dataset))]


def _measure_vocos_mel(cfg: DictConfig, dev: torch.device, audio: list[np.ndarray], rf_audio: np.ndarray, durations: list[float]):
    from sparc.vocoders.eval.references import VocosMelReference

    spec = cfg.eval_features.efficiency
    rf = spec.receptive_field
    ref = VocosMelReference(cfg, dev)
    mels = [ref.mel(a) for a in audio]
    warm = int(spec.warmup_calls)
    params = {"generator": count_parameters(ref.model), "total": count_parameters(ref.model)}
    rtf = {
        key: {
            "decode": time_calls([lambda m=m: ref.decode(m) for m in mels], durations, dev, threads, warm),
            "copy_synthesis": time_calls([lambda a=a: ref.synth(a) for a in audio], durations, dev, threads, warm),
        }
        for key, threads in rtf_keys(dev, spec.cpu_threads)
    }
    field = {
        "measured": vocos_mel_receptive_field(
            ref, rf_audio, int(rf.perturb_frame), int(rf.frames), float(rf.perturb_std), float(rf.threshold)
        ),
        "analytic": reference_analytic("vocos_mel", ref),
    }
    return params, rtf, field, ref.describe()


def _measure_enplus(cfg: DictConfig, dev: torch.device, audio: list[np.ndarray], rf_audio: np.ndarray, durations: list[float]):
    from sparc.vocoders.eval.references import EnPlusReference

    spec = cfg.eval_features.efficiency
    rf = spec.receptive_field
    ref = EnPlusReference(cfg, dev)
    codes = [ref.encode(a) for a in audio]
    embs = [ref.spk_emb(c) for c in codes]
    warm = int(spec.warmup_calls)

    def encode_decode(wav: np.ndarray) -> np.ndarray:
        code = ref.encode(wav)
        return ref.decode_raw(code, ref.spk_emb(code))

    generator, ffn, head = ref.coder.generator, ref.coder.speaker_encoder.spk_enc, ref.coder.inverter.linear_model
    params = {
        "generator": count_parameters(generator),
        "generator_folded": count_parameters(generator, folded=True),
        "speaker_ffn": count_parameters(ffn),
        "linear_head": count_parameters(head),
        "total": count_parameters(generator) + count_parameters(ffn),
    }
    rtf = {
        key: {
            "decode": time_calls(
                [lambda c=c, e=e: ref.decode_raw(c, e) for c, e in zip(codes, embs)], durations, dev, threads, warm
            ),
            "encode_decode": time_calls([lambda a=a: encode_decode(a) for a in audio], durations, dev, threads, warm),
        }
        for key, threads in rtf_keys(dev, spec.cpu_threads)
    }
    field = {
        "measured": enplus_receptive_field(
            ref, rf_audio, int(rf.perturb_frame), int(rf.frames), float(rf.perturb_std), float(rf.threshold)
        ),
        "analytic": reference_analytic("enplus16", ref),
    }
    return params, rtf, field, ref.describe()


def _measure_extractor(cfg: DictConfig, dev: torch.device, audio: list[np.ndarray], durations: list[float]):
    from sparc.vocoders.eval.reextract import ReExtractor

    spec = cfg.eval_features.efficiency
    extractor = ReExtractor(cfg, "refit", dev)
    rtf = {
        key: {
            "extract": time_calls(
                [lambda a=a: extractor.extract(a, SAMPLE_RATE) for a in audio], durations, dev, threads, int(spec.warmup_calls)
            )
        }
        for key, threads in rtf_keys(dev, spec.cpu_threads)
    }
    return rtf, extractor.describe()


def run_efficiency(cfg: DictConfig, system: str, device: str, out_path: Path) -> dict:
    """Measures ``system`` on ``device`` and merges the results into the JSON file ``out_path``; returns the file's content.

    ``system`` is a vocoder of ``cfg.eval.systems`` (with ``experiment``, ``vocoder`` and optional ``ckpt`` and
    ``overrides``), ``vocos_mel``, ``enplus16`` or ``extractor`` (SPARC feature extraction with the refit head: the
    shared cost of the vocoders' input, reported once). The utterances are the probe subset (the first
    ``cfg.eval_features.efficiency.n_utterances`` of it: 3-10 s, at most 2 per speaker). On ``cuda`` the ``gpu`` RTF is
    measured, on ``cpu`` one RTF per entry of ``cpu_threads`` (``cpu_1_threads``, ``cpu_8_threads``). Parameters, the
    analytic and the measured receptive field do not depend on the device and are rewritten by every call; entries
    measured on the other device are kept. Layout: ``params``, ``rtf.<device key>``, ``receptive_field.{measured,
    analytic}`` and ``meta``.
    """
    from sparc.vocoders.eval.probes import probe_batches, probe_subset
    from sparc.vocoders.eval.vocoder_loader import checkpoint_info, compose_vocoder_config, load_vocoder

    spec = cfg.eval_features.efficiency
    split = cfg.eval_features.probes.split
    dev = torch.device(device)
    ids = probe_subset(cfg, split)[: int(spec.n_utterances)]
    rf = spec.receptive_field
    frames, frame = int(rf.frames), int(rf.perturb_frame)
    rf_id = _long_utterance(cfg, split, frames)
    meta: dict = {
        "system": system,
        "torch": torch.__version__,
        "n_rtf_utterances": len(ids),
        "rtf_utterance_ids": ids,
        "receptive_field_source": rf_id,
        "runs": {dev.type: {"gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None}},
    }
    result: dict = {}

    if system in (*REFERENCE_SYSTEMS, EXTRACTOR_SYSTEM):
        vcfg = compose_vocoder_config(None, "hifigan")  # paths and the eval gain only
        audio = _gt_audio(vcfg, ids, split)
        durations = [len(a) / SAMPLE_RATE for a in audio]
        if system == EXTRACTOR_SYSTEM:
            result["rtf"], meta["extractor"] = _measure_extractor(cfg, dev, audio, durations)
        else:
            rf_audio = _gt_audio(vcfg, [rf_id], split)[0]
            measure = _measure_vocos_mel if system == "vocos_mel" else _measure_enplus
            result["params"], result["rtf"], result["receptive_field"], meta["reference"] = measure(
                cfg, dev, audio, rf_audio, durations
            )
    else:
        sys_spec = cfg.eval.systems[system]
        module, vcfg = load_vocoder(
            sys_spec.experiment, sys_spec.vocoder, sys_spec.get("ckpt", None), dev, list(sys_spec.get("overrides", []) or [])
        )
        stats = load_stats(vcfg.stats_path)
        batches = probe_batches(vcfg, ids, dev, split)
        long_batch = probe_batches(vcfg, [rf_id], dev, split)[0]
        start = (long_batch["features"].shape[-1] - frames) // 2
        window = long_batch["features"][:, :, start : start + frames].contiguous()
        result["params"] = param_counts(module)
        result["rtf"] = {
            key: measure_rtf(module, batches, dev, threads, int(spec.warmup_calls)) for key, threads in rtf_keys(dev, spec.cpu_threads)
        }
        result["receptive_field"] = {
            "measured": receptive_field(
                module, window, long_batch["spk_raw"], stats, frame, float(rf.perturb_std), float(rf.threshold)
            ),
            "analytic": analytic_receptive_field(module.generator),
            "source_start_frame": int(start),
        }
        meta["checkpoint"] = checkpoint_info(module)
        meta["experiment"] = str(vcfg.experiment_name)
        meta["vocoder"] = str(vcfg.vocoder.name)
    return _merge_json(Path(out_path), {**result, "meta": meta})
