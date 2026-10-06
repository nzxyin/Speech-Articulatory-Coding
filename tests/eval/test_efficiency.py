"""Efficiency measurements: parameter counts, timing, receptive field (CPU; the ``slow`` tests use real checkpoints)."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn.utils import weight_norm

import sparc
from sparc.vocoders.constants import F0_CHANNEL, HOP, LOUDNESS_CHANNEL, N_EMA, PERIODICITY_CHANNEL, SAMPLE_RATE
from sparc.vocoders.eval import efficiency
from sparc.vocoders.eval.efficiency import (
    CHANNEL_GROUPS,
    analytic_receptive_field,
    count_parameters,
    extent_from_outputs,
    measure_rtf,
    param_counts,
    perturb_features,
    receptive_field,
    rtf_keys,
    time_calls,
)
from sparc.vocoders.models.hifigan import HiFiGANVocoder
from sparc.vocoders.models.speaker import SpeakerFFN
from sparc.vocoders.models.vocos import VocosVocoder
from sparc.vocoders.models.ddsp import DDSPVocoder

CONF = Path(sparc.__file__).parent / "conf"
STATS = {
    "ema_mean": [0.1 * i for i in range(12)],
    "ema_std": [1.0 + 0.1 * i for i in range(12)],
    "logf0_mean": 5.0,
    "logf0_std": 0.3,
    "loud_log_mean": -5.0,
    "loud_log_std": 1.5,
    "per_mean": 0.5,
    "per_std": 0.4,
}


def make_features(frames: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(1, 15, frames, generator=g)
    feats[:, F0_CHANNEL] = 100.0 + 50.0 * torch.rand(1, frames, generator=g)
    feats[:, LOUDNESS_CHANNEL] = 0.01 + 0.05 * torch.rand(1, frames, generator=g)
    feats[:, PERIODICITY_CHANNEL] = 0.5 + 0.4 * torch.rand(1, frames, generator=g)
    return feats


class Wrapper(nn.Module):
    """The three parts of ``VocoderGANModule`` that the efficiency code touches."""

    def __init__(self, generator: nn.Module):
        super().__init__()
        self.generator = generator
        self.speaker = SpeakerFFN()

    def synthesize(self, batch):
        return self.generator(batch["features"], self.speaker(batch["spk_raw"]))


class ToyConv(nn.Module):
    """Frame-wise ``Conv1d`` (kernel 5, all-ones weights) whose output is repeated ``HOP`` times per frame."""

    def __init__(self, kernel: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(15, 1, kernel, padding=kernel // 2, bias=False)
        nn.init.constant_(self.conv.weight, 1.0)
        self.speaker = nn.Linear(1, 1)
        self.generator = nn.Identity()

    def synthesize(self, batch):
        return self.conv(batch["features"]).repeat_interleave(HOP, dim=-1)


# ---------------------------------------------------------------------------------------------------------------------
# parameters


def test_count_parameters_handles_weight_norm_gains():
    conv_old = weight_norm(nn.Conv1d(3, 4, 5))  # weight_g / weight_v
    plain = nn.Conv1d(3, 4, 5)
    assert count_parameters(plain) == 3 * 4 * 5 + 4
    assert count_parameters(conv_old) == 3 * 4 * 5 + 4 + 4  # direction, bias and the per-output-channel gain
    assert count_parameters(conv_old, folded=True) == 3 * 4 * 5 + 4
    conv_new = nn.utils.parametrizations.weight_norm(nn.Conv1d(3, 4, 5))
    assert count_parameters(conv_new) == count_parameters(conv_old)
    assert count_parameters(conv_new, folded=True) == count_parameters(plain)


def test_param_counts_of_a_tiny_module():
    generator = nn.Sequential(weight_norm(nn.Conv1d(2, 3, 3)), nn.Conv1d(3, 1, 1))
    generator.frontend = nn.Module()
    generator.frontend.register_buffer("mean", torch.zeros(7))
    module = Wrapper(generator)
    speaker = sum(p.numel() for p in SpeakerFFN().parameters())
    gen = (2 * 3 * 3 + 3 + 3) + (3 * 1 + 1)
    counts = param_counts(module)
    assert counts == {
        "generator": gen,
        "generator_folded": gen - 3,
        "speaker_ffn": speaker,
        "total": gen + speaker,
        "total_folded": gen - 3 + speaker,
        "frontend_buffers": 7,
    }
    assert speaker == 1024 * 1024 + 1024 + 1024 * 64 + 64  # SpeakerFFN: buffers (z-score) are not parameters


@pytest.mark.parametrize(
    "build",
    [
        lambda: HiFiGANVocoder(STATS, channels=32),
        lambda: VocosVocoder(STATS, option="B", dim=32, intermediate_dim=96, num_layers=2),
        lambda: DDSPVocoder(STATS, channels=24, n_harmonics=12, n_noise_bands=17, post_filter_taps=65, head_hidden=24, up_channels=(16, 12)),
    ],
    ids=["hifigan", "vocos", "ddsp"],
)
def test_param_counts_of_the_real_generators(build):
    generator = build()
    counts = param_counts(Wrapper(generator))
    assert counts["generator"] == sum(p.numel() for p in generator.parameters())
    assert counts["generator_folded"] <= counts["generator"]
    assert counts["total"] == counts["generator"] + counts["speaker_ffn"]
    assert counts["frontend_buffers"] > 0  # the frontend holds the normalization statistics as buffers only


# ---------------------------------------------------------------------------------------------------------------------
# timing


def test_time_calls_counts_warmup_and_reports_rtf():
    calls_made = []

    def call():
        calls_made.append(1)
        time.sleep(0.01)

    result = time_calls([call, call, call], [2.0, 1.0, 1.0], "cpu", threads=None, warmup=5)
    assert len(calls_made) == 5 + 3
    assert result["n"] == 3 and result["warmup_calls"] == 5 and result["device"] == "cpu" and result["gpu"] is None
    assert result["audio_s"] == 4.0 and 0.03 <= result["time_s"] < 0.2
    assert result["rtf"] == pytest.approx(result["time_s"] / 4.0)
    assert result["rtf_median"] <= result["rtf_p95"] <= result["rtf_max"]
    with pytest.raises(ValueError):
        time_calls([call], [1.0, 2.0], "cpu")


def test_time_calls_sets_and_restores_the_thread_count():
    before = torch.get_num_threads()
    seen = []
    result = time_calls([lambda: seen.append(torch.get_num_threads())], [1.0], "cpu", threads=1, warmup=1)
    assert seen == [1, 1] and result["threads"] == 1
    assert torch.get_num_threads() == before


def test_measure_rtf_uses_synthesize_in_eval_mode_and_restores_training_mode():
    module = Wrapper(HiFiGANVocoder(STATS, channels=16))
    module.train()
    batches = [{"features": make_features(6, seed=i), "spk_raw": torch.randn(1, 1024)} for i in range(2)]
    result = measure_rtf(module, batches, "cpu", threads=1, warmup=1)
    assert result["n"] == 2 and result["audio_s"] == pytest.approx(2 * 6 * HOP / SAMPLE_RATE)
    assert result["rtf"] > 0 and module.training


def test_rtf_keys():
    assert rtf_keys(torch.device("cuda"), [1, 8]) == [("gpu", None)]
    assert rtf_keys(torch.device("cpu"), [1, 8]) == [("cpu_1_threads", 1), ("cpu_8_threads", 8)]


# ---------------------------------------------------------------------------------------------------------------------
# receptive field


def test_extent_from_outputs_known_impulse():
    y0 = np.ones(4800)
    y1 = y0.copy()
    y1[1000:1500] += 0.5
    out = extent_from_outputs(y0, y1, frame=3, hop=480, sample_rate=24000, threshold=1e-4)
    assert (out["first_sample"], out["last_sample"]) == (1000, 1499)
    assert out["lookahead_s"] == pytest.approx((480 * 3 - 1000) / 24000)  # reacts 440 samples before the frame starts
    assert out["past_s"] == pytest.approx((1499 - 480 * 4) / 24000)
    assert out["span_s"] == pytest.approx(500 / 24000) and not out["reaches_start"] and not out["reaches_end"]
    below = y0.copy()
    below[10] += 0.5e-4  # under 1e-4 of max |y0| = 1: ignored
    assert extent_from_outputs(y0, below, 3, 480, 24000)["first_sample"] is None
    assert np.isnan(extent_from_outputs(y0, y0, 3, 480, 24000)["lookahead_s"])
    edge = y0.copy()
    edge[0] += 1.0
    edge[-1] += 1.0
    out = extent_from_outputs(y0, edge, 3, 480, 24000)
    assert out["reaches_start"] and out["reaches_end"]


def test_perturb_features_changes_one_frame_of_the_chosen_channels_by_the_stated_amount():
    x = make_features(30)
    before = x.clone()
    for name, channels in CHANNEL_GROUPS.items():
        y = perturb_features(x, 12, 0.5, STATS, channels)
        assert torch.equal(x, before)
        changed = (y != x)
        assert not changed[:, :, [t for t in range(30) if t != 12]].any(), name
        assert set(torch.nonzero(changed[0, :, 12]).flatten().tolist()) <= set(channels)
        assert torch.nonzero(changed[0, :, 12]).numel() == len(channels), name
    y = perturb_features(x, 12, 0.5, STATS, (3,))
    assert y[0, 3, 12] == pytest.approx(x[0, 3, 12] + 0.5 * STATS["ema_std"][3])
    y = perturb_features(x, 12, 0.5, STATS, (F0_CHANNEL,))
    assert torch.log(y[0, F0_CHANNEL, 12] / x[0, F0_CHANNEL, 12]) == pytest.approx(0.5 * STATS["logf0_std"], rel=1e-5)
    y = perturb_features(x, 12, 0.5, STATS, (LOUDNESS_CHANNEL,))
    ratio = (y[0, LOUDNESS_CHANNEL, 12] + 1e-4) / (x[0, LOUDNESS_CHANNEL, 12] + 1e-4)
    assert torch.log(ratio) == pytest.approx(0.5 * STATS["loud_log_std"], rel=1e-4)
    y = perturb_features(x, 12, 0.5, STATS, (PERIODICITY_CHANNEL,))
    assert y[0, PERIODICITY_CHANNEL, 12] == pytest.approx(x[0, PERIODICITY_CHANNEL, 12] + 0.5 * STATS["per_std"])


@pytest.mark.parametrize("kernel", [1, 3, 5, 9])
def test_receptive_field_of_a_toy_conv_has_the_known_extent(kernel):
    half = kernel // 2
    frame = 40
    module = ToyConv(kernel)
    out = receptive_field(module, make_features(80), torch.zeros(1, 1024), STATS, frame=frame, std_multiple=0.5, groups=("all", "ema"))
    for group in ("all", "ema"):
        ext = out[group]
        # a kernel of k frames reacts from frame - half to frame + half: samples [480 (frame - half), 480 (frame + half + 1))
        assert ext["first_sample"] == HOP * (frame - half)
        assert ext["last_sample"] == HOP * (frame + half + 1) - 1
        assert ext["lookahead_s"] == pytest.approx(half * HOP / SAMPLE_RATE)
        assert ext["past_s"] == pytest.approx((HOP * (frame + half + 1) - 1 - HOP * (frame + 1)) / SAMPLE_RATE)
    # the output is the same length as the input and the call did not modify the features
    assert out["frame"] == frame and out["frames"] == 80 and out["hop"] == HOP


def test_receptive_field_of_a_toy_conv_past_context_formula():
    module = ToyConv(5)
    out = receptive_field(module, make_features(80), torch.zeros(1, 1024), STATS, frame=40, groups=("all",))["all"]
    # frames 38..42 react; the frame itself is 40, so the output reacts for 2 frames after the frame ends:
    assert out["past_s"] == pytest.approx((HOP * 43 - 1 - HOP * 41) / SAMPLE_RATE)
    assert out["span_s"] == pytest.approx(5 * HOP / SAMPLE_RATE)


def test_receptive_field_restores_training_mode_and_checks_the_frame():
    module = ToyConv(3)
    module.train()
    receptive_field(module, make_features(20), torch.zeros(1, 1024), STATS, frame=10, groups=("all",))
    assert module.training
    with pytest.raises(ValueError):
        receptive_field(module, make_features(20), torch.zeros(1, 1024), STATS, frame=20)


GENERATORS = {
    "hifigan": lambda: HiFiGANVocoder(STATS, channels=16),
    "vocos_B": lambda: VocosVocoder(STATS, option="B", dim=32, intermediate_dim=96, num_layers=2),
    "vocos_A": lambda: VocosVocoder(STATS, option="A", dim=32, intermediate_dim=96, num_layers=2),
    "ddsp": lambda: DDSPVocoder(STATS, channels=24, n_harmonics=12, n_noise_bands=17, post_filter_taps=65, head_hidden=24, up_channels=(16, 12)),
}


@pytest.mark.parametrize("name", list(GENERATORS))
def test_measured_receptive_field_lies_within_the_analytic_one(name):
    """The analytic extent (layer sizes) bounds what the perturbation can reach; the measured one is no larger."""
    torch.manual_seed(0)
    generator = GENERATORS[name]().eval()
    analytic = analytic_receptive_field(generator)
    assert analytic is not None and analytic["left_samples"] > 0 and analytic["right_samples"] > 0
    frames, frame = 120, 60
    module = Wrapper(generator).eval()
    out = receptive_field(module, make_features(frames, seed=3), torch.randn(1, 1024), STATS, frame=frame, groups=("all", "ema"))
    ext = out["all"]
    assert ext["first_sample"] is not None, "the perturbation had no effect"
    left = HOP * frame - ext["first_sample"]  # samples before the frame that already react
    right = ext["last_sample"] - HOP * (frame + 1)
    slack = 8
    assert left <= analytic["left_samples"] + slack, (left, analytic)
    if analytic["unbounded_right"]:
        assert ext["reaches_end"] or right >= analytic["right_samples"] * 0.5  # DDSP: the pitch phase reaches every later sample
    else:
        assert right <= analytic["right_samples"] + slack, (right, analytic)
    assert ext["lookahead_s"] == pytest.approx(left / SAMPLE_RATE)
    # the model is non-causal in a bounded window: it reacts around the frame, not at the far ends of the input
    assert not ext["reaches_start"]
    assert analytic["lookahead_s"] == pytest.approx(analytic["left_samples"] / SAMPLE_RATE)


def test_analytic_receptive_field_of_an_unknown_module_is_none():
    assert analytic_receptive_field(nn.Linear(1, 1)) is None


def test_analytic_extent_of_a_hifigan_matches_a_hand_count():
    """``hifigan_like_extent`` for one input conv, one transposed conv and one residual conv, counted by hand."""
    input_conv = nn.Sequential(nn.Conv1d(2, 4, 7, padding=3))  # half extent 3 input frames
    up = nn.Sequential(nn.ConvTranspose1d(4, 4, 16, stride=8, padding=4))  # reaches 4 left and 16 - 8 - 4 = 4 right at the new step
    block = nn.Sequential(nn.Conv1d(4, 4, 3, padding=1, dilation=1))  # half extent 1 at the new step
    output = nn.Conv1d(4, 1, 7, padding=3)  # half extent 3 at the new step
    left, right = efficiency.hifigan_like_extent(input_conv, [up], [block], 1, output, hop=64)
    step = 64 / 8
    expected_left = 3 * 64 + 4 * step + 1 * step + 3 * step
    expected_right = 3 * 64 + 4 * step + 1 * step + 3 * step
    assert (left, right) == (expected_left, expected_right)


# ---------------------------------------------------------------------------------------------------------------------
# JSON merging


def test_merge_json_merges_nested_dicts_and_writes_atomically(tmp_path):
    path = tmp_path / "eff" / "sys.json"
    efficiency._merge_json(path, {"params": {"total": 1}, "rtf": {"gpu": {"rtf": 0.1}}, "meta": {"a": 1}})
    data = efficiency._merge_json(path, {"params": {"total": 2}, "rtf": {"cpu_1_threads": {"rtf": 0.5}}, "meta": {"b": 2}})
    assert data == {
        "params": {"total": 2},
        "rtf": {"gpu": {"rtf": 0.1}, "cpu_1_threads": {"rtf": 0.5}},  # the other device's entry is kept
        "meta": {"a": 1, "b": 2},
    }
    assert json.loads(path.read_text()) == data and not list(path.parent.glob("*.tmp"))


# ---------------------------------------------------------------------------------------------------------------------
# slow: the driver on real data and checkpoints (CPU)

NEEDED_ENV = ("HF_HUB_CACHE", "SPARC_REFIT_NPZ", "SPARC_VOC_CACHE", "SPARC_VOC_RUNS", "LIBRITTSR_RAW")


def driver_cfg():
    if any(name not in os.environ for name in NEEDED_ENV):
        pytest.skip("environment of the feature cache is not set")
    if not (Path(os.environ["SPARC_VOC_CACHE"]) / "packed" / "test.clean" / "index.parquet").exists():
        pytest.skip("packed test.clean features are missing")
    cfg = OmegaConf.create(
        {
            "eval_features": OmegaConf.load(CONF / "eval_features" / "default.yaml"),
            "paths": {"cache_root": os.environ["SPARC_VOC_CACHE"]},
            "eval": {"systems": {"hifigan": {"kind": "vocoder", "experiment": "main", "vocoder": "hifigan"}}},
        }
    )
    cfg.eval_features.references.hub_cache = os.environ["HF_HUB_CACHE"]
    cfg.eval_features.efficiency.n_utterances = 2
    cfg.eval_features.efficiency.warmup_calls = 1
    cfg.eval_features.efficiency.cpu_threads = [2]
    cfg.eval_features.efficiency.receptive_field.frames = 80
    cfg.eval_features.efficiency.receptive_field.perturb_frame = 40
    return cfg


@pytest.mark.slow
def test_run_efficiency_of_a_trained_vocoder(tmp_path):
    cfg = driver_cfg()
    ckpt_dir = Path(os.environ["SPARC_VOC_RUNS"]) / "main" / "hifigan" / "ckpt"
    if not ckpt_dir.exists() or not list(ckpt_dir.glob("step*.ckpt")):
        pytest.skip("no checkpoint of main/hifigan yet")
    out = efficiency.run_efficiency(cfg, "hifigan", "cpu", tmp_path / "hifigan.json")
    assert json.loads((tmp_path / "hifigan.json").read_text()) == out
    assert out["params"]["total"] == out["params"]["generator"] + out["params"]["speaker_ffn"]
    rtf = out["rtf"]["cpu_2_threads"]
    assert rtf["n"] == 2 and rtf["rtf"] > 0 and rtf["threads"] == 2
    field = out["receptive_field"]
    assert field["measured"]["all"]["first_sample"] is not None and field["analytic"]["architecture"] == "HiFiGANVocoder"
    assert out["meta"]["checkpoint"]["g_step"] > 0 and out["meta"]["n_rtf_utterances"] == 2
    print("hifigan:", json.dumps({k: out[k] for k in ("params", "rtf")}, indent=1)[:1500])
    print("receptive field (all):", {k: v for k, v in field["measured"]["all"].items() if k.endswith("_s")}, field["analytic"])


@pytest.mark.slow
@pytest.mark.parametrize("system", ["vocos_mel", "enplus16", "extractor"])
def test_run_efficiency_of_the_reference_systems_and_the_extractor(tmp_path, system):
    cfg = driver_cfg()
    pytest.importorskip("vocos")
    try:
        out = efficiency.run_efficiency(cfg, system, "cpu", tmp_path / f"{system}.json")
    except FileNotFoundError as error:
        pytest.skip(str(error))
    key = "cpu_2_threads"
    assert all(v["rtf"] > 0 for v in out["rtf"][key].values())
    if system != "extractor":
        assert out["params"]["generator"] > 0
        measured = out["receptive_field"]["measured"]["all"]
        assert measured["first_sample"] is not None and out["receptive_field"]["analytic"]["right_samples"] > 0
        print(system, "params", out["params"], "rtf", {k: round(v["rtf"], 3) for k, v in out["rtf"][key].items()},
              "measured", {k: v for k, v in measured.items() if k.endswith("_s")}, "analytic", out["receptive_field"]["analytic"])
    else:
        print("extractor rtf", {k: round(v["rtf"], 3) for k, v in out["rtf"][key].items()})
