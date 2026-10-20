"""Phase 3 resynthesis tables (``sparc.vocoders.eval.p3_tables``): pairing, bootstrap, aggregation with fakes."""

import numpy as np
import pandas as pd
import pytest

from sparc.vocoders.eval import p3_tables as t

N_SPK, PER = 8, 6
BOOT = {"n_boot": 400, "seed": 0, "ci": 0.95}


def fake_table() -> pd.DataFrame:
    rows = []
    for s in range(N_SPK):
        for u in range(PER):
            ref = f"{100 + s}_1_{(u + 1) % PER:06d}_000000"
            rows.append({"id": f"{100 + s}_1_{u:06d}_000000", "speaker": str(100 + s), "ref_id": ref, "T": 100 + u, "dur_s": 2.0, "chapter": "1"})
    return pd.DataFrame(rows)


def embeddings(table: pd.DataFrame, rng, noise: float, dim: int = 8) -> dict:
    centers = {s: rng.normal(size=dim) for s in table["speaker"].unique()}
    return {r.id: centers[r.speaker] + noise * rng.normal(size=dim) for r in table.itertuples()}


def cell(table, rng, noise, utmos, wer_err) -> pd.DataFrame:
    gt = {m: embeddings(table, np.random.default_rng(5), 0.1) for m in t.SV_MODELS}
    sys_emb = {m: {k: v + noise * rng.normal(size=v.shape) for k, v in gt[m].items()} for m in t.SV_MODELS}
    utt = pd.DataFrame({"id": table["id"], "utmos": utmos})
    asr = pd.DataFrame({"id": table["id"], "word_errors": wer_err, "word_ref_len": 10.0, "char_errors": wer_err * 4, "char_ref_len": 40.0})
    # shuffled inputs: the assembly must key on id, not on position
    return t.condition_frame(table, "T1", table["id"].tolist(), utt.sample(frac=1, random_state=1), asr.sample(frac=1, random_state=2), sys_emb, gt)


def test_resolve_run():
    r = t.resolve_run("b3")
    assert r.run_dir == t.P3_RUNS / "p3_b3" / "hifigan"
    b0 = t.resolve_run("b0")
    assert b0.experiment == "ablation_100k" and b0.runs_root == t.B0_RUNS
    custom = t.resolve_run("x", "/tmp/runs/exp/voc")
    assert (str(custom.runs_root), custom.experiment, custom.vocoder) == ("/tmp/runs", "exp", "voc")
    with pytest.raises(ValueError):
        t.resolve_run("zzz")
    assert r.system()["experiment"] == "p3_b3" and r.system()["conditions"] == ["T1", "T2", "T3"]


def test_parse_pairs():
    assert t.parse_pairs(t.DEFAULT_PAIRS) == [("b1", "b0"), ("b2", "b1"), ("b3", "b2"), ("b4", "b3"), ("b5", "b3")]


def test_read_only_guard():
    with pytest.raises(SystemExit):
        t.check_writable_target(t.resolve_run("b0"))
    t.check_writable_target(t.resolve_run("b1"))


def test_condition_frame_similarity_and_mean():
    table = fake_table()
    frame = cell(table, np.random.default_rng(0), 0.05, 3.0, 1.0)
    assert len(frame) == len(table) and t.completeness(frame) == {k: 0 for k in t.completeness(frame)}
    assert (frame[t.SIM_COLUMNS] <= 1.0 + 1e-9).all().all()
    # same-utterance similarity beats the other-utterances speaker mean when the noise is small
    assert frame["sim_same_ecapa"].mean() > 0.9
    assert np.allclose(frame["sim_same_mean"], (frame["sim_same_ecapa"] + frame["sim_same_wavlm_sv"]) / 2)
    assert np.allclose(frame["sim_spk_mean"], (frame["sim_spk_ecapa"] + frame["sim_spk_wavlm_sv"]) / 2)
    assert frame["utmos"].eq(3.0).all()  # shuffled parts joined by id


def test_sim_spk_excludes_target_and_reference():
    table = fake_table()
    spk = table["speaker"].iloc[0]
    members = table[table["speaker"] == spk]
    gt = {r.id: np.eye(8)[i] for i, r in enumerate(members.itertuples())}
    # system embedding of utterance 0 equals its own gt: sim_spk must still not see it (nor its T2 reference, utterance 1)
    sys_emb = dict(gt)
    frame = members.assign(t2_ref=members["ref_id"].to_numpy())
    from sparc.vocoders.eval.aggregate import similarity_columns

    same, sim_spk = similarity_columns(frame, sys_emb, gt, False)
    assert same[0] == pytest.approx(1.0)
    # mean of the remaining one-hot vectors e2..e5 normalized: cosine with e0 is exactly 0
    assert sim_spk[0] == pytest.approx(0.0, abs=1e-12)


def test_paired_difference_known_answer_and_pairing_by_id():
    table = fake_table()
    rng = np.random.default_rng(1)
    a = cell(table, rng, 0.05, 3.5, 1.0)
    b = a.copy()
    b["utmos"] = a["utmos"] - 0.25
    b["sim_same_mean"] = a["sim_same_mean"] - 0.1
    cells = {("a", "T1"): a, ("b", "T1"): b.sample(frac=1, random_state=3).reset_index(drop=True)}
    paired = t.paired_summary(cells, [("a", "b")], ["T1"], BOOT).set_index("metric")
    assert paired.loc["utmos", "diff"] == pytest.approx(0.25)
    assert paired.loc["utmos", "lo"] == pytest.approx(0.25) and paired.loc["utmos", "hi"] == pytest.approx(0.25)
    assert paired.loc["sim_same_mean", "diff"] == pytest.approx(0.1)
    assert bool(paired.loc["sim_same_mean", "ci_excludes_zero"])
    assert paired.loc["wer", "diff"] == pytest.approx(0.0) and not bool(paired.loc["wer", "ci_excludes_zero"])
    assert paired.loc["utmos", "n_speakers"] == N_SPK and paired.loc["utmos", "n_utt"] == len(table)
    assert paired.loc["utmos", "frac_speakers_same_direction"] == pytest.approx(1.0)


def test_paired_uses_only_shared_ids_and_resamples_speakers():
    table = fake_table()
    rng = np.random.default_rng(2)
    a = cell(table, rng, 0.05, 3.0, 1.0)
    # per-speaker shift of different sign: the CI must be wide because speakers are the resampling unit
    b = a.copy()
    b["utmos"] = a["utmos"] + np.where(a["speaker"].astype(int) % 2 == 0, 0.5, -0.5)
    b = b[b["id"] != a["id"].iloc[0]]  # one utterance missing in b
    paired = t.paired_summary({("a", "T1"): a, ("b", "T1"): b}, [("a", "b")], ["T1"], BOOT).set_index("metric")
    assert paired.loc["utmos", "n_utt"] == len(a) - 1
    assert paired.loc["utmos", "lo"] < 0 < paired.loc["utmos", "hi"]
    assert not bool(paired.loc["utmos", "ci_excludes_zero"])


def test_summary_table_means_and_wer_is_corpus_level():
    table = fake_table()
    frame = cell(table, np.random.default_rng(3), 0.05, 3.0, 2.0)
    out = t.summary_table({("x", "T1"): frame}, BOOT).iloc[0]
    assert out["utmos"] == pytest.approx(3.0) and out["wer"] == pytest.approx(0.2) and out["cer"] == pytest.approx(0.2)
    assert out["sim_same_mean"] == pytest.approx(frame["sim_same_mean"].mean())
    assert out["n_utt"] == len(table) and out["n_speakers"] == N_SPK
    md = t.render_table_md(pd.DataFrame([out]))
    assert "## T1" in md and "| x |" in md


def test_compare_with_reference():
    mine = pd.DataFrame([{"system": "b0", "condition": "T2", "utmos": 3.0, "wer": 0.03, "cer": 0.01, "sim_same_ecapa": 0.681, "sim_spk_ecapa": 0.719}])
    ref = pd.DataFrame([{"system": "a100k_hifigan", "condition": "T2", "stratum": "all", "utmos": 3.0, "wer": 0.03, "cer": 0.01, "sim_same": 0.6812, "sim_spk": 0.7300}])
    cmp = t.compare_with_reference(mine, ref, {"b0": "a100k_hifigan"}).set_index("metric")
    assert bool(cmp.loc["sim_same_ecapa", "ok"]) and not bool(cmp.loc["sim_spk_ecapa", "ok"])


def test_render_paired_md():
    table = fake_table()
    rng = np.random.default_rng(4)
    a = cell(table, rng, 0.05, 3.5, 1.0)
    b = a.assign(utmos=a["utmos"] - 0.1)
    paired = t.paired_summary({("a", "T2"): a, ("b", "T2"): b}, [("a", "b")], ["T2"], BOOT)
    md = t.render_paired_md(paired)
    assert "## a - b, T2" in md and "| utmos | +0.100 [+0.100, +0.100] * |" in md
