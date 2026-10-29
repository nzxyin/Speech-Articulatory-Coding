"""Phase 8.2 constrained code fitting: selection, rule and helpers (CPU, synthetic rows)."""

import math

import numpy as np
import pandas as pd
import pytest

from sparc.vocoders.eval import ood_svfit_p8 as p8


def _rows(sim_by_cfg: dict, utmos_by_cfg: dict, wer_by_cfg: dict | None = None) -> pd.DataFrame:
    """Two pools x two items; the standard code has CAM++ 0.5, UTMOS 3.5, 4 % WER."""
    out = []
    for pool in ("a", "b"):
        for i in range(2):
            base = {"pool": pool, "own_id": f"{pool}{i}", "speaker": f"{pool}{i}", "wer_len": 100.0}
            out.append({**base, "arm": "std", "lr": 0.0, "lam": 0.0, "nat": 0.0, "step": 0, "shift": 0.0, "holdout/sim": 0.5, "wespeaker/sim": 0.4, "ecapa/sim": 0.5, "utmos": 3.5, "wer_err": 4.0})
            for cfg, sim in sim_by_cfg.items():
                arm, lr, lam, nat, step = cfg
                out.append({**base, "arm": arm, "lr": lr, "lam": lam, "nat": nat, "step": step, "shift": 0.1, "holdout/sim": sim, "wespeaker/sim": sim - 0.1, "ecapa/sim": sim + 0.1,
                            "utmos": utmos_by_cfg[cfg], "wer_err": (wer_by_cfg or {}).get(cfg, 4.0)})
    return pd.DataFrame(out)


def test_selection_applies_guard_in_every_pool_and_picks_cam_plus_only():
    cfgs = {("l2", 0.03, 1.0, 0.0, 50): 0.6, ("l2", 0.03, 1.0, 0.0, 100): 0.8, ("nat", 0.03, 1.0, 3.0, 100): 0.7, ("unc", 0.03, 0.0, 0.0, 100): 0.9}
    utmos = {("l2", 0.03, 1.0, 0.0, 50): 3.49, ("l2", 0.03, 1.0, 0.0, 100): 3.30, ("nat", 0.03, 1.0, 3.0, 100): 3.47, ("unc", 0.03, 0.0, 0.0, 100): 3.0}
    choice, _ = p8.select_choice(_rows(cfgs, utmos))
    assert choice["l2"]["steps"] == 50  # the 100-step l2 fit breaks the 0.05 UTMOS guard
    assert choice["nat"]["steps"] == 100 and choice["nat"]["nat"] == 3.0
    assert choice["unc"]["reference_flagged"] and choice["unc"]["steps"] == 100  # nothing passes: best overall, flagged
    assert choice["chosen"] == "nat"  # better CAM++ score of the two constrained arms (0.7 > 0.6)
    assert choice["l2"]["dev_score"] == pytest.approx(0.6)


def test_selection_falls_back_to_standard_when_nothing_beats_it_and_wer_guard_blocks():
    cfgs = {("l2", 0.01, 0.3, 0.0, 25): 0.45, ("nat", 0.01, 0.3, 1.0, 25): 0.9}
    utmos = {k: 3.5 for k in cfgs}
    choice, _ = p8.select_choice(_rows(cfgs, utmos, {("nat", 0.01, 0.3, 1.0, 25): 7.0}))
    assert choice["l2"]["steps"] == 0 and choice["nat"]["steps"] == 0 and choice["chosen"] == "standard"


def test_swap_selection_is_seeded_and_restricted():
    plan = {"swaps": [{"swap_id": f"s{i:03d}", "target": f"t{i % 5}"} for i in range(100)]}
    a = p8.swap_selection(plan, {"t0", "t1"}, 10)
    assert a == p8.swap_selection(plan, {"t0", "t1"}, 10) and len(a) == 10 and {s["target"] for s in a} <= {"t0", "t1"}
    assert len(p8.swap_selection(plan, {"t0"}, 50)) == 20


def test_arm_grid_covers_the_three_arms_and_the_widened_edges():
    cfgs = p8.arm_configs()
    assert {c[0] for c in cfgs} == {"unc", "l2", "nat"} and max(c[1] for c in cfgs) == 0.1 and max(p8.SNAPS) == 400
    assert all(c[2] == 0 and c[3] == 0 for c in cfgs if c[0] == "unc") and all(c[3] == 0 and c[2] > 0 for c in cfgs if c[0] == "l2")
    assert all(c[2] > 0 and c[3] > 0 for c in cfgs if c[0] == "nat")


def test_mel_statistics_match_the_reference_distance():
    torch = pytest.importorskip("torch")
    torchaudio = pytest.importorskip("torchaudio")
    rng = np.random.default_rng(0)
    w = torch.from_numpy(rng.standard_normal(24000).astype(np.float32))[None]
    mel = torchaudio.transforms.MelSpectrogram(p8.SAMPLE_RATE, n_fft=p8.N_FFT, hop_length=p8.HOP, n_mels=p8.N_MEL, f_min=0.0, f_max=p8.SAMPLE_RATE / 2)
    lm = torch.log(mel(w).clamp_min(1e-5))
    torch_stats = torch.cat([lm.mean(-1), lm.std(-1)], -1)[0].numpy()
    assert np.allclose(torch_stats, p8.mel_stats_np(lm[0].numpy()), atol=1e-5)
    assert p8.mel_distance(torch_stats, torch_stats) == 0.0 and p8.mel_distance(torch_stats, torch_stats + 0.5) == pytest.approx(0.25)


def _own_df(gain: float, utmos_drop: float, n: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for s in range(n):
        g = {"holdout": 0.8, "wespeaker": 0.8}
        std = {"holdout": 0.4 + 0.02 * rng.standard_normal(), "wespeaker": 0.4 + 0.02 * rng.standard_normal()}
        r = {"speaker": f"s{s}", "n_held": 5}
        for c in ("std", "nat", "unc", "l2"):
            for m in p8.MODELS:
                base = std.get(m, 0.4)
                r[f"{c}/{m}/sim"] = base + (gain if c != "std" else 0.0)
            r[f"{c}/utmos"] = 3.5 - (utmos_drop if c != "std" else 0.0)
            r[f"{c}/wer_err"], r[f"{c}/wer_len"] = 4.0, 100.0
        for m in p8.MODELS:
            r[f"gt/{m}/sim"] = g.get(m, 0.8)
            r[f"std/{m}/sim"] = std.get(m, 0.4)
        r["gt/utmos"], r["gt/wer_err"], r["gt/wer_len"] = 3.9, 3.0, 100.0
        rows.append(r)
    # std similarity must be the same column used for the gain
    df = pd.DataFrame(rows)
    for c in ("nat", "unc", "l2"):
        for m in p8.MODELS:
            df[f"{c}/{m}/sim"] = df[f"std/{m}/sim"] + gain
    return df


def test_closure_and_rule_pass_and_fail():
    ok = {p: p8.summarize_pool(_own_df(0.16, 0.02), p8.ARMS, 300) for p in ("libritts", "vctk", "globe")}
    assert ok["vctk"]["models"]["wespeaker"]["nat"]["closure"]["mean"] == pytest.approx(0.4, abs=0.02)
    swap = {p: {"models": {m: {"src": {"diff": {"mean": 0.005}}} for m in p8.MODELS}} for p in ("libritts", "vctk", "globe_spk")}
    rule = p8.advance_rule(ok, swap, "nat")
    # libritts own-speaker similarity rises in this fixture, so no drop; closure 0.4 with a tight interval
    assert rule["advances"] and rule["checks"]["closure_lower_bound_>=0.25_each_held_out_verifier"]
    bad_utmos = {p: p8.summarize_pool(_own_df(0.16, 0.2), p8.ARMS, 300) for p in ("libritts", "vctk", "globe")}
    assert not p8.advance_rule(bad_utmos, swap, "nat")["advances"]
    weak = {p: p8.summarize_pool(_own_df(0.02, 0.0), p8.ARMS, 300) for p in ("libritts", "vctk", "globe")}
    assert not p8.advance_rule(weak, swap, "nat")["checks"]["closure_lower_bound_>=0.25_each_held_out_verifier"]
    src_up = {p: {"models": {m: {"src": {"diff": {"mean": 0.05}}} for m in p8.MODELS}} for p in swap}
    assert not p8.advance_rule(ok, src_up, "nat")["advances"]
    assert math.isfinite(rule["libritts_own_drop"]["wespeaker"])
