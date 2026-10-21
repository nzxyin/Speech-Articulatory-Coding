"""Speaker-embedding Phase 4 cloning evaluation (``sparc.vocoders.eval.clone``, ``clone_pools``): pure functions with fakes."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from sparc.vocoders.constants import F0_CHANNEL, LOUDNESS_CHANNEL, N_FEATURES, N_EMA, PERIODICITY_CHANNEL, SAMPLE_RATE
from sparc.vocoders.eval import clone, clone_pools
from sparc.vocoders.models.speaker import SpeakerFFN


def fake_table(n_speakers: int = 16, per_speaker: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_speakers):
        spk = f"spk{s:02d}"
        for u in range(per_speaker):
            dur = float(rng.uniform(1.2, 14.0))
            rows.append({"id": f"{spk}_{u:03d}", "speaker": spk, "gender": "F" if s % 2 == 0 else "M", "n24": int(dur * SAMPLE_RATE)})
    return pd.DataFrame(rows)


def voiced_mask(x):
    return x[PERIODICITY_CHANNEL] > 0


def fake_features(t: int = 200, f0: float = 120.0, spread: float = 0.2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((N_FEATURES, t)).astype(np.float32)
    voiced = rng.random(t) < 0.6
    x[PERIODICITY_CHANNEL] = np.where(voiced, 0.8, 0.0)
    x[F0_CHANNEL] = np.where(voiced, f0 * np.exp(spread * rng.standard_normal(t)), 0.0)
    x[LOUDNESS_CHANNEL] = np.exp(rng.standard_normal(t) * 0.3 - 2.0)
    return x


# ------------------------------------------------------------------------------------------------------ plan


def test_allocate_quota_redistributes_shortfalls():
    assert clone.allocate_quota(400, [1000, 1000, 1000, 1000]) == [100, 100, 100, 100]
    q = clone.allocate_quota(400, [50, 1000, 1000, 10])
    assert q[0] == 50 and q[3] == 10 and sum(q) == 400 and abs(q[1] - q[2]) <= 1
    assert clone.allocate_quota(10, [2, 3]) == [2, 3]


def test_plan_is_deterministic_and_stratified():
    table = fake_table()
    kw = dict(seed=3, n_swap=80, n_own=20, n_curve=10, n_refs=10, min_refs=6)
    p1 = clone.build_plan(table, "fake", **kw)
    p2 = clone.build_plan(table.sample(frac=1.0, random_state=1), "fake", **kw)
    assert p1 == p2
    assert clone.build_plan(table, "fake", **{**kw, "seed": 4}) != p1
    swaps = p1["swaps"]
    assert len(swaps) == 80
    # half same / half cross gender
    assert sum(s["same_gender"] for s in swaps) == 40
    # equal duration quotas when every bin has enough utterances
    assert p1["quota_per_bin"] == [20, 20, 20, 20]
    bins = np.bincount([s["dur_bin"] for s in swaps], minlength=4)
    assert list(bins) == [20, 20, 20, 20]
    for s in swaps:
        assert clone.duration_bin(s["dur_s"]) == s["dur_bin"]
        assert len({s["source"], s["target"], s["wrong"]}) == 3
        assert p1["gender"][s["wrong"]] == p1["gender"][s["target"]]
        assert (p1["gender"][s["source"]] == p1["gender"][s["target"]]) == s["same_gender"]
        # references: >= 3 s, at most 10, content is never one of its own speaker's references
        assert s["content_id"] not in p1["refs"][s["source"]]
        assert sorted(s["target_order"]) == sorted(p1["refs"][s["target"]])
        assert 6 <= len(p1["refs"][s["target"]]) <= 10
    n24 = table.set_index("id")["n24"]
    assert all(n24[u] >= 3 * SAMPLE_RATE for r in p1["refs"].values() for u in r)
    # targets are used evenly
    use = pd.Series([s["target"] for s in swaps]).value_counts()
    assert use.max() - use.min() <= 2
    assert p1["curve_ids"] == [s["swap_id"] for s in swaps[:10]]
    for o in p1["own"]:
        assert o["content_id"] not in p1["refs"][o["speaker"]]


def test_plan_bin_shortfall_goes_to_other_bins():
    table = fake_table()
    table.loc[table["n24"] > 10 * SAMPLE_RATE, "n24"] = 9 * SAMPLE_RATE  # no utterance above 10 s
    plan = clone.build_plan(table, "fake", n_swap=60, n_own=0, n_refs=10, min_refs=6)
    assert plan["quota_per_bin"][3] == 0 and sum(plan["quota_per_bin"]) == 60 and len(plan["swaps"]) == 60


def test_plan_needs_enough_eligible_speakers():
    table = fake_table(n_speakers=2)
    with pytest.raises(ValueError):
        clone.build_plan(table, "fake", min_refs=6)


def test_stratified_speakers_balance_gender_and_accent():
    spk = pd.DataFrame({"speaker": [f"s{i}" for i in range(20)], "gender": ["F", "M"] * 10, "accent": ["a", "b", "c", "d", "e"] * 4})
    chosen = clone_pools.stratified_speakers(spk, 8, seed=0)
    assert len(chosen) == 8 and chosen == clone_pools.stratified_speakers(spk, 8, seed=0)
    g = spk.set_index("speaker").loc[chosen]
    assert (g["gender"] == "F").sum() == 4
    assert g[g["gender"] == "F"]["accent"].nunique() == 4
    assert len(clone_pools.stratified_speakers(spk.head(4), 10)) == 4


def test_vctk_info_and_majority_gender():
    info = clone_pools.parse_vctk_speaker_info("ID  AGE  GENDER  ACCENTS  REGION\n225  23  F    English    Southern  England\n226  22  M    Scottish\n")
    assert info == {"p225": ("F", "English"), "p226": ("M", "Scottish")}
    assert clone_pools.majority_gender(["female", "male", "female"]) == "F"
    assert clone_pools.majority_gender(["female", "male"]) is None


# ------------------------------------------------------------------------------------------------------ matching


def test_f0_matching_hits_target_median_and_range():
    content = fake_features(f0=100.0, spread=0.14, seed=1)
    target = fake_features(f0=210.0, spread=0.20, seed=2)
    med_t, rng_t = clone.f0_stats([target])
    out = clone.match_f0(content, med_t, rng_t)
    med_o, rng_o = clone.f0_stats([out])
    assert med_o == pytest.approx(med_t, abs=1e-4)
    assert rng_o == pytest.approx(rng_t, rel=1e-3)  # the ratio is inside the clip
    # unvoiced frames and other channels untouched, input not modified
    unv = content[PERIODICITY_CHANNEL] <= 0
    assert np.array_equal(out[F0_CHANNEL][unv], content[F0_CHANNEL][unv])
    assert np.array_equal(np.delete(out, F0_CHANNEL, 0), np.delete(content, F0_CHANNEL, 0))
    assert clone.f0_stats([content])[0] == pytest.approx(math.log(100.0), abs=0.05)


def test_f0_range_ratio_is_clipped_and_output_in_bounds():
    content = fake_features(f0=100.0, spread=0.02, seed=1)
    target = fake_features(f0=200.0, spread=0.5, seed=2)
    med_t, rng_t = clone.f0_stats([target])
    out = clone.match_f0(content, med_t, rng_t)
    ratio = clone.f0_stats([out])[1] / clone.f0_stats([content])[1]
    assert ratio == pytest.approx(clone.F0_RANGE_CLIP[1], rel=1e-3)
    high = clone.match_f0(content, math.log(2000.0), rng_t)
    assert high[F0_CHANNEL].max() <= clone.F0_MAX_HZ + 1e-3 and high[F0_CHANNEL][voiced_mask(high)].min() >= clone.F0_MIN_HZ
    assert clone.semitone_range(math.log(2.0)) == pytest.approx(12.0)


def test_f0_matching_without_voiced_frames_is_identity():
    x = fake_features()
    x[PERIODICITY_CHANNEL] = 0.0
    assert np.array_equal(clone.match_f0(x, math.log(200.0), 0.3), x)


def test_loudness_matching_sets_voiced_median():
    content = fake_features(seed=1)
    target = fake_features(seed=2)
    target[LOUDNESS_CHANNEL] *= 4.0
    out = clone.match_loudness(content, clone.loudness_stat([target]))
    assert clone.loudness_stat([out]) == pytest.approx(clone.loudness_stat([target]), abs=1e-5)
    assert np.allclose(out[LOUDNESS_CHANNEL] / content[LOUDNESS_CHANNEL], out[LOUDNESS_CHANNEL][0] / content[LOUDNESS_CHANNEL][0], rtol=1e-4)


def test_ema_matching_sets_mean_and_std():
    content = fake_features(seed=1)
    target = fake_features(seed=2)
    target[:N_EMA] = target[:N_EMA] * 3.0 + 1.5
    mean_t, std_t = clone.ema_stats([target])
    out = clone.match_ema(content, mean_t, std_t)
    mean_o, std_o = clone.ema_stats([out])
    assert np.allclose(mean_o, mean_t, atol=1e-4) and np.allclose(std_o, std_t, atol=1e-4)
    assert np.array_equal(out[N_EMA:], content[N_EMA:])


def test_apply_matching_flags_and_copy():
    content, target = fake_features(f0=100.0, seed=1), fake_features(f0=200.0, seed=2)
    stats = clone.Matching.from_features([target])
    orig = content.copy()
    none = clone.apply_matching(content, stats, False, False, False)
    assert np.array_equal(none, content) and none is not content
    f0 = clone.apply_matching(content, stats, True, False, False)
    assert not np.array_equal(f0[F0_CHANNEL], content[F0_CHANNEL]) and np.array_equal(f0[LOUDNESS_CHANNEL], content[LOUDNESS_CHANNEL])
    full = clone.apply_matching(content, stats, True, True, True)
    assert not np.array_equal(full[LOUDNESS_CHANNEL], content[LOUDNESS_CHANNEL]) and not np.array_equal(full[:N_EMA], content[:N_EMA])
    assert np.array_equal(content, orig)


# ------------------------------------------------------------------------------------------------------ variants / controls


def test_controls_wiring():
    plan = clone.build_plan(fake_table(), "fake", seed=1, n_swap=24, n_own=8, n_curve=6, min_refs=6)
    variants = clone.all_variants(5)
    items = clone.build_items(plan, variants, list(variants))
    by = {}
    for it in items:
        by.setdefault(it.variant, []).append(it)
    s0 = plan["swaps"][0]
    pick = {v: next(i for i in by[v] if i.pair_id == s0["swap_id"]) for v in ("base", "k5", "k10", "ctl_src", "ctl_wrong", "k5_f0_loud_ema")}
    assert pick["base"].code_ids == s0["target_order"][:1] and pick["k5"].code_ids == s0["target_order"][:5]
    assert pick["k10"].code_ids == s0["target_order"][:10]
    assert pick["ctl_src"].code_speaker == s0["source"] and pick["ctl_src"].code_ids == s0["source_order"][:5]
    assert pick["ctl_src"].match_ids == s0["target_order"][:5] and pick["ctl_src"].f0 and pick["ctl_src"].loud and not pick["ctl_src"].ema
    assert pick["ctl_wrong"].code_speaker == s0["wrong"] and pick["ctl_wrong"].match_ids == s0["target_order"][:5]
    assert pick["ctl_wrong"].code_speaker not in (s0["source"], s0["target"])
    assert pick["k5_f0_loud_ema"].ema and pick["k5_f0_loud_ema"].code_speaker == s0["target"]
    assert all(i.target == i.code_speaker for i in by["k5"])
    assert len({i.key for i in items}) == len(items)
    # curves only on flagged pairs, own on own items, length clips use all references
    assert {i.pair_id for i in by["len10_best"]} == set(plan["curve_ids"]) and len(by["noisy_base"]) == 6
    assert all(i.cond == "len10" and i.code_ids == next(s for s in plan["swaps"] if s["swap_id"] == i.pair_id)["target_order"] for i in by["len10_base"])
    assert by["len3_best"][0].f0 and by["len3_best"][0].loud and by["noisy_best"][0].f0
    assert len(by["own_t2"]) == len(plan["own"]) == 8 and all(len(i.code_ids) == 1 and i.kind == "own" for i in by["own_t2"])
    assert all(i.source == i.target == i.code_speaker for i in by["own_t3"])
    assert clone._wanted("ctl_src", "controls") and clone._wanted("k5_f0", "main") and not clone._wanted("noisy_base", "main,own")
    assert clone._wanted("own_t2", "own") and clone._wanted("len1_base", "curve") and clone._wanted("anything", "all")


def test_code_from_vectors_is_mean_of_reference_codes():
    torch.manual_seed(0)

    class M:
        speaker = SpeakerFFN(in_dim=8, hidden_dim=8, out_dim=4, dropout=0.0).eval()

    vecs = [np.random.default_rng(i).standard_normal(8).astype(np.float32) for i in range(3)]
    z = clone.code_from_vectors(M, vecs, "cpu")
    with torch.no_grad():
        expect = M.speaker(torch.from_numpy(np.stack(vecs))).mean(0).numpy()
    assert np.allclose(z, expect, atol=1e-6)


# ------------------------------------------------------------------------------------------------------ metrics


def test_eer_threshold_and_top1():
    same = np.linspace(0.5, 1.0, 101)
    diff = np.linspace(0.0, 0.6, 101)
    thr, eer = clone.eer_threshold(same, diff)
    assert 0.5 <= thr <= 0.6 and eer < 0.2
    assert clone.eer_threshold([0.9, 0.95], [0.1, 0.2]) [1] == 0.0
    t, e = clone.eer_threshold([], [0.1])
    assert math.isnan(t) and math.isnan(e)
    means = np.eye(3)
    emb = np.array([[0.9, 0.1, 0.0], [0.0, 0.2, 1.0], [np.nan] * 3])
    assert list(clone.top1_ids(emb, means)) == [0, 2, -1]


def test_calibration_stats_separates_speakers():
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((5, 16)) * 3
    ref = {f"s{i}": centers[i] + 0.3 * rng.standard_normal((8, 16)) for i in range(5)}
    cal = clone.calibration_stats(ref)
    assert cal["top1"] == 1.0 and cal["same_mean"] > cal["diff_mean"] + 0.3
    assert cal["n_same"] == 40 and cal["n_diff"] == 160 and cal["eer"] == 0.0
    assert cal["diff_mean"] < cal["eer_threshold"] <= cal["same_mean"] + 0.2


def test_holdout_refs_excludes_used_without_fallback():
    assert clone.holdout_refs(["a", "b", "c"], ["a"]) == ["b", "c"]
    assert clone.holdout_refs(["a", "b"], ["a", "b"]) == []


# ------------------------------------------------------------------------------------------------------ aggregation


def test_per_speaker_aggregation_not_per_item():
    # speaker A has 9 items at 0.2, speaker B one item at 0.8: per-speaker mean 0.5 (per-item would be 0.26)
    df = pd.DataFrame({"target": ["A"] * 9 + ["B"], "v": [0.2] * 9 + [0.8]})
    assert clone.speaker_means(df, "v").to_dict() == pytest.approx({"A": 0.2, "B": 0.8})
    ci = clone.speaker_ci(df, "v", n_boot=200, seed=0)
    assert ci["mean"] == pytest.approx(0.5) and ci["n"] == 2 and ci["lo"] <= 0.5 <= ci["hi"]
    df.loc[0, "v"] = np.nan
    assert clone.speaker_means(df, "v")["A"] == pytest.approx(0.2)


def test_speaker_bootstrap_covers_speaker_variation():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"target": np.repeat(np.arange(30), 5), "v": np.repeat(rng.normal(0.6, 0.1, 30), 5) + rng.normal(0, 0.01, 150)})
    ci = clone.speaker_ci(df, "v", n_boot=2000, seed=1)
    assert ci["hi"] - ci["lo"] == pytest.approx(2 * 1.96 * 0.1 / math.sqrt(30), rel=0.4)
    assert clone.speaker_ci(df, "v", n_boot=500, seed=1) == clone.speaker_ci(df, "v", n_boot=500, seed=1)


def test_speaker_diff_is_paired_on_items():
    a = pd.DataFrame({"pair_id": ["x", "y", "z"], "target": ["A", "A", "B"], "v": [1.0, 2.0, 5.0]})
    b = pd.DataFrame({"pair_id": ["z", "y", "x"], "target": ["B", "A", "A"], "v": [4.0, 1.0, 1.0]})
    d = clone.speaker_diff_ci(a, b, "v", n_boot=100)
    assert d["mean"] == pytest.approx(0.5 * (0.5 + 1.0)) and d["n"] == 2


def test_speaker_wer_and_strata():
    df = pd.DataFrame({"target": ["A", "A", "B"], "wer_err": [1, 0, 4], "wer_len": [10, 10, 10]})
    w = clone.speaker_wer(df, n_boot=100)
    assert w["per_speaker"]["mean"] == pytest.approx(0.5 * (0.05 + 0.4)) and w["corpus"]["mean"] == pytest.approx(5 / 30)
    sw = pd.DataFrame({"same_gender": [True, False, True], "dur_bin": [0, 1, 1], "target": list("ABC")})
    s = clone.strata(sw)
    assert set(s) == {"all", "same_gender", "cross_gender", "bin:lt3", "bin:3-6"} and len(s["same_gender"]) == 2


def synthetic_items(gain_a: float, wer_a: int = 0, utmos_a: float = 0.0, src_a: float = 0.0, own_drop: float = 0.0, seed: int = 0):
    """Per-output tables of a baseline and an arm on a shared fake plan (3 pools, 12 speakers, 6 items each)."""
    rng = np.random.default_rng(seed)
    base_rows, arm_rows = [], []
    for pool in ("libritts", "vctk", "globe"):
        for s in range(12):
            for k in range(6):
                pid = f"s{s:02d}{k}"
                eff = rng.normal(0.5, 0.05)
                common = dict(pool=pool, kind="swap", variant="v", pair_id=pid, target=f"{pool}{s}", source="x", same_gender=bool(k % 2), dur_bin=k % 4, dur_s=2.0 + k)
                for rows, g, w, u, sc in ((base_rows, 0.0, 0, 0.0, 0.0), (arm_rows, gain_a, wer_a, utmos_a, src_a)):
                    r = {**common, "wer_err": 1 + w, "wer_len": 100, "utmos": 3.0 + u}
                    for m in ("ecapa", "wavlm_sv", "holdout"):
                        r[f"{m}/sim_target"] = eff + g + rng.normal(0, 0.005)
                        r[f"{m}/sim_source"] = 0.3 + sc
                    rows.append(r)
            for k in range(3):
                pid = f"o{s:02d}{k}"
                for rows, drop in ((base_rows, 0.0), (arm_rows, own_drop)):
                    r = dict(pool=pool, kind="own", variant="own_t2", pair_id=pid, target=f"{pool}{s}", source=f"{pool}{s}", same_gender=True, dur_bin=0, dur_s=3.0 + 2 * k,
                             wer_err=0, wer_len=0, utmos=3.0)
                    for m in ("ecapa", "wavlm_sv", "holdout"):
                        r[f"{m}/sim_spk"] = 0.7 - drop
                    rows.append(r)
    return pd.DataFrame(arm_rows), pd.DataFrame(base_rows)


MODELS = ("ecapa", "wavlm_sv", "holdout")


def test_decision_rule_advances_on_clear_gain():
    arm, base = synthetic_items(0.05)
    d = clone.decision_rule(arm, base, MODELS, margin=0.02, n_boot=300)
    assert d["advance"] and d["primary_ok"] and d["same_sign"] and d["guards_ok"]
    assert d["gain"]["libritts"]["ecapa"]["mean"] == pytest.approx(0.05, abs=0.005) and d["gain"]["libritts"]["ecapa"]["lo"] > 0.04
    assert "advance=True" in clone.render_decision(d)


def test_decision_rule_gain_below_margin_or_noise():
    arm, base = synthetic_items(0.01)
    d = clone.decision_rule(arm, base, MODELS, margin=0.02, n_boot=300)
    assert not d["primary_ok"] and not d["advance"]
    arm, base = synthetic_items(0.0)
    assert not clone.decision_rule(arm, base, MODELS, margin=0.02, n_boot=300)["advance"]


def test_decision_rule_sign_must_agree_across_pools_and_models():
    arm, base = synthetic_items(0.05)
    arm.loc[arm["pool"] == "globe", "holdout/sim_target"] -= 0.2
    d = clone.decision_rule(arm, base, MODELS, margin=0.02, n_boot=300)
    assert d["primary_ok"] and not d["same_sign"] and not d["advance"]
    # an arm trained against WavLM-SV is judged without it
    arm, base = synthetic_items(0.05)
    arm["wavlm_sv/sim_target"] -= 0.2
    assert not clone.decision_rule(arm, base, MODELS, margin=0.02, n_boot=300)["advance"]
    assert clone.decision_rule(arm, base, MODELS, margin=0.02, exclude_models=["wavlm_sv"], n_boot=300)["advance"]


@pytest.mark.parametrize(
    "kw, field",
    [({"wer_a": 1}, "wer"), ({"utmos_a": -0.2}, "utmos"), ({"src_a": 0.05}, "source"), ({"own_drop": 0.05}, "t23")],
)
def test_decision_rule_guards_block_advance(kw, field):
    arm, base = synthetic_items(0.05, **kw)
    d = clone.decision_rule(arm, base, MODELS, margin=0.02, n_boot=300)
    assert d["primary_ok"] and not d["guards_ok"] and not d["advance"]
    assert all(not g["pass"][field] for g in d["guards"].values())
    # a change inside the guard limits passes
    arm, base = synthetic_items(0.05, wer_a=0, utmos_a=-0.05, src_a=0.01, own_drop=0.005)
    assert clone.decision_rule(arm, base, MODELS, margin=0.02, n_boot=300)["advance"]


def test_decision_rule_needs_variant_choice_when_ambiguous():
    arm, base = synthetic_items(0.05)
    extra = arm[arm["kind"] == "swap"].assign(variant="w")
    arm2 = pd.concat([arm, extra], ignore_index=True)
    with pytest.raises(ValueError):
        clone.decision_rule(arm2, base, MODELS, margin=0.02, n_boot=100)
    d = clone.decision_rule(arm2, base, MODELS, margin=0.02, variant_a="w", variant_b="v", n_boot=100)
    assert d["advance"]


def test_item_table_and_summary_end_to_end():
    rng = np.random.default_rng(0)
    plan = clone.build_plan(fake_table(n_speakers=8, per_speaker=24), "fake", n_swap=16, n_own=4, n_curve=2, n_refs=8, min_refs=6)
    spk = plan["speakers"]
    dim = 12
    centers = {s: rng.standard_normal(dim) * 3 for s in spk}
    ref_utt = {u: centers[s] + 0.2 * rng.standard_normal(dim) for s, r in plan["refs"].items() for u in r}
    ref_means = {"fake": {"ecapa": {s: clone.mean_unit(np.stack([ref_utt[u] for u in r])) for s, r in plan["refs"].items()}}}
    ref_by_utt = {"fake": {"ecapa": ref_utt}}
    cids = {s["content_id"]: s["source"] for s in plan["swaps"]} | {o["content_id"]: o["speaker"] for o in plan["own"]}
    gt = {"fake": {"ecapa": {u: centers[s] + 0.2 * rng.standard_normal(dim) for u, s in cids.items()}}}
    variants = clone.all_variants(5)
    items = clone.build_items(plan, variants, ["base", "k5", "ctl_src", "ctl_wrong", "own_t2"])
    manifest = pd.DataFrame([{"key": i.key, "kind": i.kind, "pool": i.pool, "pair_id": i.pair_id, "variant": i.variant, "content_id": i.content_id,
                              "source": i.source, "target": i.target, "same_gender": i.same_gender, "dur_bin": i.dur_bin, "dur_s": i.dur_s,
                              "code_ids": i.code_ids} for i in items])
    # outputs: sound like the code's speaker (target for base / k5, source for ctl_src, wrong for ctl_wrong)
    code_spk = {i.key: i.code_speaker for i in items}
    emb = np.stack([centers[code_spk[k]] + 0.2 * rng.standard_normal(dim) for k in manifest["key"]])
    thresholds = {"fake": {"ecapa": 0.5}}
    table = clone.item_table(manifest, {"ecapa": emb}, np.full(len(manifest), 3.5), [(1, 20)] * len(manifest), ref_means, ref_by_utt, gt, thresholds, {"fake": plan})
    sw = table[table["kind"] == "swap"]
    ok = sw[sw["variant"].isin(["base", "k5"])]
    assert (ok["ecapa/top1"] == 1.0).all() and (ok["ecapa/accept"] == 1.0).all() and (ok["ecapa/sim_target"] > 0.8).all()
    src = sw[sw["variant"] == "ctl_src"]
    assert (src["ecapa/top1_source"] == 1.0).all() and (src["ecapa/sim_source"] > src["ecapa/sim_target"]).all()
    assert (sw[sw["variant"] == "ctl_wrong"]["ecapa/top1"] == 0.0).all()
    own = table[table["kind"] == "own"]
    assert own["ecapa/sim_spk"].notna().all() and (own["ecapa/sim_same"] > 0.8).all() and sw["ecapa/sim_spk"].isna().all()
    assert sw["ecapa/gt_same"].notna().all()
    cal = {"fake": {"ecapa": clone.calibration_stats({s: np.stack([ref_utt[u] for u in r]) for s, r in plan["refs"].items()})}}
    summary = clone.summarize_items(table, cal, ["ecapa"], n_boot=100)
    v = summary["fake"]["variants"]
    assert v["k5"]["all"]["metrics"]["ecapa/sim_target"]["mean"] > 0.8
    assert v["k5"]["all"]["n_speakers"] == sw[sw["variant"] == "k5"]["target"].nunique()
    margin = summary["fake"]["margins"]["k5"]["ctl_src"]["ecapa"]
    assert margin["mean"] > 0.3 and margin["lo"] > 0
    assert "ctl_src" not in summary["fake"]["margins"]["ctl_src"] and "all" in summary["fake"]["own"]["own_t2"] and "gt4s" in summary["fake"]["own"]["own_t2"]
    assert any(k.startswith("bin:") for k in v["k5"])
    text = clone.render_summary(clone.sanitize({"meta": {}, "pools": summary}))
    assert "| k5 |" in text and "EER" in text


# ------------------------------------------------------------------------------------------------------ review fixes


class FakeStore:
    def __init__(self, vectors: dict[str, np.ndarray]):
        self.vectors, self.ids = vectors, list(vectors)

    def index_of(self, uid: str) -> int:
        return self.ids.index(uid)

    def speaker_vector(self, i: int) -> np.ndarray:
        return self.vectors[self.ids[i]]


class FakeConds:
    """Reference feature source: per-id fake features (also used for the length / noisy conditions)."""

    def __init__(self, feats: dict[str, np.ndarray]):
        self.feats, self.calls = feats, 0

    def clean_features(self, ids):
        return [self.feats[u] for u in ids]

    def encode_conditioned(self, cond, ids):
        self.calls += 1
        return [np.full(8, 0.5, dtype=np.float32)], [self.feats[u] for u in ids]


class FakeModule:
    """Minimal vocoder module: code = mean of the references' first 4 dims (+ pitch code), output length from the frames."""

    def __init__(self, rel_f0: bool):
        from sparc.vocoders.models.speaker import PitchCode

        self.rel_f0 = rel_f0
        self.speaker = self
        self.pitch_code = None
        if rel_f0:
            self.pitch_code = PitchCode(math.log(150.0), 0.3, out_dim=4)
            torch.nn.init.constant_(self.pitch_code.proj.weight, 0.5)
        self.batches: list[dict] = []

    def embed(self, raw, mask, sample=False):
        return raw[:, :, :4].mean(1), {}

    def synthesize(self, batch, z=None, features=None):
        if self.rel_f0 and "utt_f0" not in batch:
            raise KeyError("utt_f0")
        self.batches.append({**batch, "z": z})
        return torch.zeros(1, 1, batch["features"].shape[-1] * 480)


def make_renderer(rel_f0: bool):
    plan = clone.build_plan(fake_table(n_speakers=6, per_speaker=14), "fake", seed=2, n_swap=8, n_own=4, n_curve=2, n_refs=6, min_refs=6)
    ids = {u for r in plan["refs"].values() for u in r} | {s["content_id"] for s in plan["swaps"]} | {o["content_id"] for o in plan["own"]}
    rng = np.random.default_rng(0)
    spk_f0 = {s: 100.0 + 25.0 * i for i, s in enumerate(plan["speakers"])}
    feats = {u: fake_features(f0=spk_f0[u.split("_")[0]], seed=int(u[-3:]) + 5 * int(u[3:5])) for u in ids}
    store = FakeStore({u: rng.standard_normal(8).astype(np.float32) for u in ids})
    module = FakeModule(rel_f0)
    conds = FakeConds(feats)
    renderer = clone.ItemRenderer(module, {"fake": store}, {"fake": conds}, 0.0, torch.device("cpu"), features_fn=lambda st, u, g: feats[u])
    return plan, module, renderer, feats, conds, store


def test_missing_plan_ids_detects_unpacked_utterances():
    plan, _, _, _, _, store = make_renderer(False)
    assert clone.missing_plan_ids(plan, store) == []
    victim = plan["swaps"][0]["content_id"]
    del store.vectors[victim]
    store.ids.remove(victim)
    assert clone.missing_plan_ids(plan, store) == [victim]


def test_renderer_plain_checkpoint_renders_with_matching():
    plan, module, renderer, feats, _, _ = make_renderer(False)
    variants = clone.all_variants(5)
    it = next(i for i in clone.build_items(plan, variants, ["k5_f0_loud"]) if i.pair_id == plan["swaps"][0]["swap_id"])
    wav = renderer.render(it)
    b = module.batches[-1]
    assert wav.shape == (feats[it.content_id].shape[1] * 480,) and "utt_f0" not in b
    target_med = clone.f0_stats([feats[u] for u in it.match_ids])[0]
    assert clone.f0_stats([b["features"][0].numpy()])[0] == pytest.approx(target_med, abs=1e-3)  # absolute F0 matched to the target


def test_renderer_relative_f0_checkpoint_gets_utt_and_ref_f0_and_pitch_code():
    plan, module, renderer, feats, _, store = make_renderer(True)
    variants = clone.all_variants(5)
    s0 = plan["swaps"][0]
    by = {i.variant: i for i in clone.build_items(plan, variants, ["k5", "k5_f0", "ctl_src"]) if i.pair_id == s0["swap_id"]}
    renderer.render(by["k5"])
    b = module.batches[-1]
    content = feats[by["k5"].content_id]
    assert float(b["utt_f0"][0]) == pytest.approx(float(np.median(content[F0_CHANNEL][voiced_mask(content)])), rel=1e-6)
    ref_f0 = clone.reference_f0_hz([feats[u] for u in by["k5"].code_ids])
    raw = np.stack([store.vectors[u] for u in by["k5"].code_ids])
    expect = raw[:, :4].mean(0) + module.pitch_code(torch.tensor([ref_f0])).detach().numpy()[0]
    assert np.allclose(b["z"][0].numpy(), expect, atol=1e-5)
    assert np.allclose(b["features"][0].numpy(), content)  # no F0 / loudness change without matching
    # F0 matching keeps the content median (the generator divides it out), only the range is matched
    renderer.render(by["k5_f0"])
    f = module.batches[-1]["features"][0].numpy()
    assert clone.f0_stats([f])[0] == pytest.approx(clone.f0_stats([content])[0], abs=1e-4)
    t_rng = clone.f0_stats([feats[u] for u in by["k5_f0"].match_ids])[1]
    ratio = float(np.clip(t_rng / clone.f0_stats([content])[1], *clone.F0_RANGE_CLIP))
    assert clone.f0_stats([f])[1] == pytest.approx(ratio * clone.f0_stats([content])[1], rel=1e-3)
    # the control takes the source speaker's references for the code and its pitch level
    renderer.render(by["ctl_src"])
    z = module.batches[-1]["z"][0].numpy()
    ref_src = clone.reference_f0_hz([feats[u] for u in by["ctl_src"].code_ids])
    raw = np.stack([store.vectors[u] for u in by["ctl_src"].code_ids])
    assert np.allclose(z, raw[:, :4].mean(0) + module.pitch_code(torch.tensor([ref_src])).detach().numpy()[0], atol=1e-5)


def test_renderer_curve_conditions_use_conditioned_reference_features():
    plan, module, renderer, feats, conds, _ = make_renderer(True)
    variants = clone.all_variants(5)
    it = next(i for i in clone.build_items(plan, variants, ["len3_best"]))
    renderer.render(it)
    assert conds.calls == 1 and it.cond == "len3"
    ref = clone.reference_f0_hz(conds.encode_conditioned("len3", it.code_ids)[1])
    assert torch.isfinite(torch.tensor(ref))


def test_reference_f0_matches_the_dataset_definition():
    from sparc.vocoders.data.dataset import geometric_mean_f0, median_voiced_f0

    fs = [fake_features(f0=100.0 + 30 * i, seed=i) for i in range(4)]
    assert clone.median_f0_hz(fs[0]) == pytest.approx(median_voiced_f0(fs[0].T))
    assert clone.reference_f0_hz(fs) == pytest.approx(geometric_mean_f0([median_voiced_f0(f.T) for f in fs]))
    novoice = fake_features()
    novoice[PERIODICITY_CHANNEL] = 0.0
    assert math.isnan(clone.median_f0_hz(novoice)) and math.isnan(clone.reference_f0_hz([novoice]))


def test_code_from_vectors_adds_pitch_code_only_for_relative_f0():
    m = FakeModule(True)
    vecs = [np.arange(8, dtype=np.float32), np.ones(8, dtype=np.float32)]
    plain = clone.code_from_vectors(FakeModule(False), vecs, "cpu")
    z = clone.code_from_vectors(m, vecs, "cpu", ref_f0=220.0)
    assert not np.allclose(z, plain) and np.allclose(z - plain, m.pitch_code(torch.tensor([220.0])).detach().numpy()[0], atol=1e-6)
    assert np.allclose(clone.code_from_vectors(m, vecs, "cpu", ref_f0=math.nan), plain)  # unknown pitch level maps to the mean


def test_build_items_curve_and_own_variants():
    plan = clone.build_plan(fake_table(), "fake", seed=1, n_swap=24, n_own=8, n_curve=6, min_refs=6)
    variants = clone.all_variants(5, "k5_f0")
    names = [n for n in variants if n.startswith(("len", "noisy", "own_"))]
    items = clone.build_items(plan, variants, names)
    by: dict[str, list] = {}
    for it in items:
        by.setdefault(it.variant, []).append(it)
    assert len(by["len1_best"]) == len(by["len30_base"]) == len(by["noisy_best"]) == 6 and len(by["own_t3"]) == 8
    for it in by["len3_best"]:
        s = next(s for s in plan["swaps"] if s["swap_id"] == it.pair_id)
        assert it.code_ids == s["target_order"] and it.match_ids == s["target_order"] and it.f0 and not it.loud
    for it in by["noisy_best"]:
        s = next(s for s in plan["swaps"] if s["swap_id"] == it.pair_id)
        assert it.cond == "noisy" and it.code_ids == s["target_order"][:5] and it.f0
    for it in by["own_t3"]:
        o = next(o for o in plan["own"] if o["own_id"] == it.pair_id)
        assert it.code_ids == o["order"][:5] and it.match_ids == it.code_ids and it.content_id == o["content_id"]
    assert all(it.variant != "len3_base" or not it.f0 for it in items)


def test_refconditions_disk_cache_is_shared(tmp_path):
    class Enc:
        calls = 0

        def encode(self, audio, seed):
            Enc.calls += 1
            return {"feats": np.ones((10, N_FEATURES), np.float32), "loud_raw": np.ones(10, np.float32), "peak24": 1.0, "vec": audio[:3]}

    class Store:
        n24 = np.array([48000, 48000])

        def index_of(self, u):
            return {"a": 0, "b": 1}[u]

        def read_audio(self, u, s, n):
            return np.full(n, 0.1 * (u + 1), np.float32)

    rc = clone.RefConditions(None, Store(), "6", 0.0, "cpu", tmp_path)
    rc._enc = Enc()
    import sparc.vocoders.eval.clone_pools as cp
    import sparc.vocoders.data.gain as gain

    orig_vec, orig_apply, orig_factor = cp.speaker_vector, gain.apply_gain, gain.gain_factor
    cp.speaker_vector = lambda e, layer: np.asarray(e["vec"], np.float32)
    gain.apply_gain, gain.gain_factor = lambda loud, f: loud, lambda peak, db: 1.0
    try:
        v1, f1 = rc.encode_conditioned("noisy", ["a", "b"])
        assert Enc.calls == 2 and len(v1) == 2 and f1[0].shape[0] == N_FEATURES
        rc2 = clone.RefConditions(None, Store(), "6", 0.0, "cpu", tmp_path)  # another shard / arm: no encoder needed
        v2, f2 = rc2.encode_conditioned("noisy", ["a", "b"])
        assert Enc.calls == 2 and np.array_equal(v1[1], v2[1]) and np.array_equal(f1[0], f2[0])
        assert list((tmp_path / "noisy").glob("*.npz")) and not list(tmp_path.rglob("*.tmp.npz"))
    finally:
        cp.speaker_vector, gain.apply_gain, gain.gain_factor = orig_vec, orig_apply, orig_factor


# ------------------------------------------------------------------------------------------------------ metrics / decision


def test_item_table_heldout_target_similarity_and_own_without_holdout():
    rng = np.random.default_rng(0)
    plan = clone.build_plan(fake_table(n_speakers=6, per_speaker=24), "fake", n_swap=8, n_own=4, n_curve=1, n_refs=6, min_refs=6)
    dim = 8
    centers = {s: rng.standard_normal(dim) * 3 for s in plan["speakers"]}
    ref_utt = {u: centers[s] + 0.2 * rng.standard_normal(dim) for s, r in plan["refs"].items() for u in r}
    ref_means = {"fake": {"ecapa": {s: clone.mean_unit(np.stack([ref_utt[u] for u in r])) for s, r in plan["refs"].items()}}}
    variants = clone.all_variants(5)
    items = clone.build_items(plan, variants, ["base", "k5", "k10", "own_t3"])
    o_full = [dict(o, order=o["order"][:5]) for o in plan["own"]]  # own speaker with exactly the code's 5 references
    plan2 = {**plan, "own": o_full, "refs": {**plan["refs"], **{o["speaker"]: o["order"][:5] for o in o_full}}}
    manifest = pd.DataFrame([{"key": i.key, "kind": i.kind, "pool": i.pool, "pair_id": i.pair_id, "variant": i.variant, "content_id": i.content_id,
                              "source": i.source, "target": i.target, "same_gender": i.same_gender, "dur_bin": i.dur_bin, "dur_s": i.dur_s,
                              "code_ids": i.code_ids, "match_ids": i.match_ids} for i in items])
    emb = np.stack([centers[i.code_speaker] + 0.2 * rng.standard_normal(dim) for i in items])
    ref_by = {"fake": {"ecapa": ref_utt}}
    table = clone.item_table(manifest, {"ecapa": emb}, np.zeros(len(items)), [(0, 1)] * len(items), ref_means, ref_by, {"fake": {"ecapa": {}}},
                             {"fake": {"ecapa": 0.5}}, {"fake": plan2})
    sw = table[table["kind"] == "swap"]
    n_refs = {s: len(r) for s, r in plan["refs"].items()}
    for var, k in (("base", 1), ("k5", 5), ("k10", 10)):
        v = sw[sw["variant"] == var]
        left = np.array([min(k, n_refs[t]) < n_refs[t] for t in v["target"]])
        assert v["ecapa/sim_target_heldout"].notna().to_numpy().tolist() == left.tolist()
        assert (v["ecapa/sim_target_heldout"].dropna() > 0.6).all()
    assert sw[sw["variant"] == "k10"]["ecapa/sim_target_heldout"].isna().all() or max(n_refs.values()) > 10
    own = table[table["kind"] == "own"]
    assert own["ecapa/sim_spk"].isna().all()  # code references are all there is: no fallback to them


def test_decision_requires_holdout_unless_allowed():
    arm, base = synthetic_items(0.05)
    models = ("ecapa", "wavlm_sv")
    with pytest.raises(ValueError, match="holdout"):
        clone.decision_rule(arm, base, models, margin=0.02, n_boot=100)
    d = clone.decision_rule(arm, base, models, margin=0.02, n_boot=100, allow_missing_holdout=True)
    assert d["primary_ok"] and d["holdout_missing"] == ["holdout"] and not d["advance"]
    assert "WARNING" in clone.render_decision(d)
    with pytest.raises(ValueError):  # excluding it counts as missing too
        clone.decision_rule(arm, base, MODELS, margin=0.02, exclude_models=["holdout"], n_boot=100)


def test_excluded_model_is_ignored_by_all_guards():
    arm, base = synthetic_items(0.05)
    arm["wavlm_sv/sim_source"] += 0.2
    arm.loc[arm["kind"] == "own", "wavlm_sv/sim_spk"] -= 0.2
    assert not clone.decision_rule(arm, base, MODELS, margin=0.02, n_boot=100)["advance"]
    d = clone.decision_rule(arm, base, MODELS, margin=0.02, exclude_models=["wavlm_sv"], n_boot=100)
    assert d["advance"] and all(g["pass"]["source"] and g["pass"]["t23"] for g in d["guards"].values())


def test_t23_guard_is_per_model_not_averaged():
    arm, base = synthetic_items(0.05)
    arm.loc[arm["kind"] == "own", "holdout/sim_spk"] -= 0.06  # one model drops, the others do not: a mean would hide it
    d = clone.decision_rule(arm, base, MODELS, margin=0.02, n_boot=100)
    assert not d["guards"]["libritts"]["pass"]["t23"] and not d["advance"]


def test_globe_ecapa_is_exempt_from_same_sign_only_there():
    arm, base = synthetic_items(0.05)
    arm.loc[arm["pool"] == "globe", "ecapa/sim_target"] -= 0.2
    d = clone.decision_rule(arm, base, MODELS, margin=0.02, n_boot=100)
    assert d["same_sign"] and d["advance"]
    arm.loc[arm["pool"] == "globe", "wavlm_sv/sim_target"] -= 0.2
    assert not clone.decision_rule(arm, base, MODELS, margin=0.02, n_boot=100)["same_sign"]
    arm, base = synthetic_items(0.05)
    arm.loc[arm["pool"] == "vctk", "ecapa/sim_target"] -= 0.2
    assert not clone.decision_rule(arm, base, MODELS, margin=0.02, n_boot=100)["same_sign"]
    assert "anti-conservative" in clone.render_decision(d)


def test_decision_rejects_tables_from_different_plans():
    arm, base = synthetic_items(0.05)
    arm["content_id"], base["content_id"] = "c1", "c1"
    clone.decision_rule(arm, base, MODELS, margin=0.02, n_boot=100)
    base.loc[(base["pool"] == "vctk") & (base["pair_id"] == "s000"), "content_id"] = "other"
    with pytest.raises(ValueError, match="content_id"):
        clone.decision_rule(arm, base, MODELS, margin=0.02, n_boot=100)
    arm, base = synthetic_items(0.05)
    with pytest.raises(ValueError, match="pair ids"):
        clone.decision_rule(arm, base[base["pair_id"] != "s010"], MODELS, margin=0.02, n_boot=100)
    plan = clone.build_plan(fake_table(), "fake", seed=1, n_swap=8, n_own=2, n_curve=1, min_refs=6)
    other = clone.build_plan(fake_table(), "fake", seed=2, n_swap=8, n_own=2, n_curve=1, min_refs=6)
    assert clone.plan_hash(plan) == clone.plan_hash(dict(reversed(list(plan.items())))) and clone.plan_hash(plan) != clone.plan_hash(other)
    with pytest.raises(ValueError, match="different plans"):
        clone.check_same_plans({"plan_hashes": {"p": "a" * 64}}, {"plan_hashes": {"p": "b" * 64}}, ["p"])
    with pytest.raises(ValueError, match="missing"):
        clone.check_same_plans({}, {"plan_hashes": {"p": "a"}}, ["p"])


def test_noise_floor_margin():
    a, b = synthetic_items(0.0)[0], synthetic_items(0.0)[1]
    r = clone.noise_floor(a, b, "libritts", "ecapa", None, n_boot=100)
    assert r["margin"] == pytest.approx(0.02) and abs(r["diff"]["mean"]) < 0.01
    big = a.copy()
    big["ecapa/sim_target"] += 0.03
    assert clone.noise_floor(big, b, "libritts", "ecapa", None, n_boot=100)["margin"] == pytest.approx(0.06, abs=0.004)


def test_select_baseline_ranks_on_heldout_references():
    def entry(primary, heldout):
        m = {"ecapa/sim_target": {"mean": primary}}
        if heldout is not None:
            m["ecapa/sim_target_heldout"] = {"mean": heldout}
        return {"all": {"metrics": m}}

    summary = {"pools": {"libritts_dev": {"variants": {
        "base": entry(0.60, 0.58), "k5": entry(0.66, 0.60), "k10": entry(0.70, None), "k5_f0": entry(0.65, 0.63), "ctl_src": entry(0.9, 0.9)}}}}
    r = clone.select_baseline(summary, "libritts_dev")
    assert r["best_variant"] == "k5_f0" and r["best_by_primary"] == "k10" and not r["agrees_with_primary"] and "ctl_src" not in r["table"]
    del summary["pools"]["libritts_dev"]["variants"]["k5"]["all"]["metrics"]["ecapa/sim_target_heldout"]
    assert clone.select_baseline(summary, "libritts_dev")["best_variant"] == "k5_f0"


def test_resolve_best_variant_reads_selection_json(tmp_path):
    f = tmp_path / "best.json"
    f.write_text('{"best_variant": "k5_f0"}')
    assert clone.resolve_best_variant(str(f)) == "k5_f0" and clone.resolve_best_variant("k5_f0_loud") == "k5_f0_loud" and clone.resolve_best_variant(None) is None


# ------------------------------------------------------------------------------------------------------ held-out SV model


def test_holdout_requirement_and_loader(tmp_path):
    assert clone.load_holdout(None, "cpu") is None
    with pytest.raises(ValueError):
        clone.load_holdout(None, "cpu", require=True)
    assert clone.load_holdout("nonsense:x", "cpu") is None
    with pytest.raises(RuntimeError):
        clone.load_holdout("nonsense:x", "cpu", require=True)
    clone.HoldoutEmbedder.LOADERS["fake"] = lambda loc, dev: (lambda w: np.ones(5) * len(w) if len(w) else (_ for _ in ()).throw(ValueError("empty")))
    try:
        h = clone.load_holdout("fake:x", "cpu", require=True)
        e = h.embed([np.zeros(0, np.float32), np.ones(10, np.float32), np.ones(4, np.float32)], errors=[])
        assert e.shape == (3, 5) and np.isnan(e[0]).all() and np.allclose(np.linalg.norm(e[1:], axis=1), 1.0)  # failed first item: width taken from later ones
        with pytest.raises(ValueError):
            h.embed([np.zeros(0, np.float32)])
    finally:
        del clone.HoldoutEmbedder.LOADERS["fake"]


def test_read_chunks_rejects_partial_models(tmp_path):
    manifest = pd.DataFrame({"key": ["a", "b"]})
    for k, with_holdout in enumerate((True, False)):
        d = {"keys": np.array([manifest["key"][k]]), "utmos": np.zeros(1), "text": np.array(["x"]), "emb_ecapa": np.ones((1, 3)), "emb_wavlm_sv": np.ones((1, 3))}
        if with_holdout:
            d["emb_holdout"] = np.ones((1, 3))
        (tmp_path / "scores").mkdir(exist_ok=True)
        np.savez(tmp_path / "scores" / f"chunk_{k:04d}.npz", **d)
    with pytest.raises(ValueError, match="holdout"):
        clone.read_chunks(tmp_path, manifest, 1)


def test_compare_excludes_models_trained_against(tmp_path, monkeypatch):
    import argparse

    arm, base = synthetic_items(0.05)
    arm["wavlm_sv/sim_target"] -= 0.2  # as if the arm were trained against WavLM-SV and the metric no longer meaningful
    h = {p: clone.plan_hash({"pool": p}) for p in ("libritts", "vctk", "globe")}
    for name, df, meta in (("arm", arm, {"plan_hashes": h, "trained_against": ["wavlm_sv"]}), ("base", base, {"plan_hashes": h})):
        (tmp_path / name).mkdir()
        df.to_parquet(tmp_path / name / "items.parquet")
        (tmp_path / name / "meta.json").write_text(__import__("json").dumps(meta))
    args = argparse.Namespace(a=str(tmp_path / "arm"), b=str(tmp_path / "base"), variant_a=None, variant_b=None, margin=0.02, primary_pool="libritts",
                              exclude_sv=None, allow_missing_holdout=False, n_boot=100, seed=0, md=None, json=None)
    d = clone.run_compare(args)
    assert d["excluded"] == ["wavlm_sv"] and d["advance"]
    (tmp_path / "base" / "meta.json").write_text(__import__("json").dumps({"plan_hashes": {**h, "vctk": "0" * 64}}))
    with pytest.raises(ValueError, match="different plans"):
        clone.run_compare(args)
