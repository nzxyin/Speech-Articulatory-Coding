"""Aggregation: speaker-cluster bootstrap, paired differences, corpus WER, strata, similarity and the table files (CPU)."""

import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sparc.vocoders.eval import aggregate as agg
from sparc.vocoders.eval import io
from sparc.vocoders.eval.systems import load_systems
from toy_split import SPLIT, build_toy_cache, set_env, toy_config

ALPHA = 0.95


def explicit_bootstrap(values_by_speaker: list[list[float]], idx: np.ndarray) -> np.ndarray:
    """Mean over utterances of each resample, written as the obvious double loop."""
    out = []
    for row in idx:
        pooled = [v for s in row for v in values_by_speaker[s]]
        out.append(np.mean(pooled))
    return np.array(out)


# ------------------------------------------------------------------------------------------------- bootstrap


def test_bootstrap_matches_the_explicit_resampling_loop():
    rng = np.random.default_rng(1)
    values = [rng.normal(size=n).tolist() for n in (3, 5, 2, 7, 4)]
    codes = np.concatenate([[s] * len(v) for s, v in enumerate(values)])
    flat = np.concatenate(values)
    num_s, den_s = agg.cluster_sums(codes, 5, flat, np.ones_like(flat))
    idx = agg.draw_clusters(5, 500, seed=0)
    result = agg.bootstrap_ratio(num_s, den_s, idx, ALPHA)
    boots = explicit_bootstrap(values, idx)
    assert result["point"] == pytest.approx(flat.mean())
    assert (result["lo"], result["hi"]) == pytest.approx(tuple(np.percentile(boots, [2.5, 97.5])))
    assert np.array_equal(idx, agg.draw_clusters(5, 500, seed=0)) and not np.array_equal(idx, agg.draw_clusters(5, 500, seed=1))


def test_bootstrap_known_answer_two_clusters():
    # speakers with all-0 and all-1 utterances: a resample holds 0, 1 or 2 copies of the 1-speaker: mean 0, 0.5, 1
    num_s, den_s = np.array([0.0, 3.0]), np.array([3.0, 3.0])
    idx = agg.draw_clusters(2, 10000, seed=0)
    result = agg.bootstrap_ratio(num_s, den_s, idx, ALPHA)
    assert result["point"] == pytest.approx(0.5)
    assert result["lo"] == 0.0 and result["hi"] == 1.0  # P(mean = 0) = P(mean = 1) = 1/4, far above 2.5 %
    boots = agg._ratio(num_s[idx].sum(1), den_s[idx].sum(1))
    assert set(np.unique(boots)) == {0.0, 0.5, 1.0}
    assert np.mean(boots == 0.5) == pytest.approx(0.5, abs=0.03)
    # constant values: no sampling variability at all
    flat = agg.bootstrap_ratio(np.array([2.0, 4.0, 6.0]), np.array([1.0, 2.0, 3.0]), agg.draw_clusters(3, 200, 0), ALPHA)
    assert flat == {"point": 2.0, "lo": 2.0, "hi": 2.0}


def test_single_speaker_has_a_point_but_no_interval():
    result = agg.bootstrap_ratio(np.array([5.0]), np.array([2.0]), agg.draw_clusters(1, 10, 0), ALPHA)
    assert result["point"] == 2.5 and np.isnan(result["lo"]) and np.isnan(result["hi"])


def test_corpus_ratio_is_not_the_mean_of_utterance_rates():
    frame = pd.DataFrame(
        {
            "speaker": ["a", "a", "b", "b"],
            "word_errors": [1.0, 0.0, 6.0, 2.0],
            "word_ref_len": [10.0, 10.0, 20.0, 20.0],
            "char_errors": [2.0, 0.0, np.nan, 4.0],  # a failed utterance does not count
            "char_ref_len": [50.0, 50.0, np.nan, 100.0],
        }
    )
    wer = next(m for m in agg.METRICS if m.key == "wer")
    cer = next(m for m in agg.METRICS if m.key == "cer")
    summary = agg.summarize_cell(frame, 200, 0, ALPHA, [wer, cer])
    assert summary["wer"]["point"] == pytest.approx(9 / 60)  # sums, not (0.1 + 0 + 0.3 + 0.1) / 4
    assert summary["wer"]["point"] != pytest.approx(np.mean([0.1, 0.0, 0.3, 0.1]))
    assert summary["cer"]["point"] == pytest.approx(6 / 200) and summary["cer"]["n"] == 3
    assert summary["n_utt"] == 4 and summary["n_speakers"] == 2
    per_speaker = agg.per_speaker_means(frame, [wer]).set_index("speaker")
    assert per_speaker["wer"].to_dict() == pytest.approx({"a": 1 / 20, "b": 8 / 40})


def test_mean_metric_ignores_nan_and_missing_columns():
    frame = pd.DataFrame({"speaker": ["a", "a", "b"], "utmos": [3.0, np.nan, 4.0]})
    utmos = next(m for m in agg.METRICS if m.key == "utmos")
    pesq = next(m for m in agg.METRICS if m.key == "pesq_wb")
    summary = agg.summarize_cell(frame, 100, 0, ALPHA, [utmos, pesq])
    assert summary["utmos"]["point"] == pytest.approx(3.5) and summary["utmos"]["n"] == 2
    assert np.isnan(summary["pesq_wb"]["point"]) and summary["pesq_wb"]["n"] == 0  # reported as n/a, not dropped
    empty = agg.summarize_cell(frame.iloc[:0], 10, 0, ALPHA, [utmos])
    assert empty["n_utt"] == 0 and np.isnan(empty["utmos"]["point"])


# ------------------------------------------------------------------------------------------------- paired differences


def paired_frames(rng, offset: float = 1.0):
    ids = [f"{s}_{k}" for s in "abcde" for k in range(4)]
    speakers = [i[0] for i in ids]
    b = pd.DataFrame({"id": ids, "speaker": speakers, "utmos": rng.normal(3, 0.3, len(ids))})
    a = b.assign(utmos=b["utmos"] + offset + rng.normal(0, 0.05, len(ids)))
    return a, b


def test_paired_difference_sign_and_direction_fraction():
    a, b = paired_frames(np.random.default_rng(0))
    utmos = [next(m for m in agg.METRICS if m.key == "utmos")]
    up = agg.paired_cell(a, b, 1000, 0, ALPHA, utmos)["utmos"]
    assert up["point"] == pytest.approx(1.0, abs=0.1) and up["lo"] > 0 and up["frac_same_direction"] == 1.0
    down = agg.paired_cell(b, a, 1000, 0, ALPHA, utmos)["utmos"]
    assert down["point"] == pytest.approx(-up["point"]) and down["hi"] < 0 and down["frac_same_direction"] == 1.0
    assert (down["lo"], down["hi"]) == pytest.approx((-up["hi"], -up["lo"]))  # same resamples, mirrored interval
    tie = agg.paired_cell(a, a, 100, 0, ALPHA, utmos)["utmos"]
    assert tie["point"] == 0.0 and np.isnan(tie["frac_same_direction"])


def test_paired_uses_only_shared_ids_and_counts_split_directions():
    a, b = paired_frames(np.random.default_rng(1), offset=0.0)
    a["utmos"] = b["utmos"] + np.where(a["speaker"].isin(["a", "b", "c"]), 1.0, -0.5)  # 3 of 5 speakers improve
    b = b[~b["id"].isin(["a_0", "e_3"])]  # system b lacks two utterances
    utmos = [next(m for m in agg.METRICS if m.key == "utmos")]
    result = agg.paired_cell(a, b, 200, 0, ALPHA, utmos)
    assert result["n_utt"] == 18 and result["utmos"]["n"] == 18
    # point = mean over the 18 shared utterances: 3 speakers (11 utterances) +1.0, 2 speakers (7) -0.5
    assert result["utmos"]["point"] == pytest.approx((11 * 1.0 - 7 * 0.5) / 18)
    assert result["utmos"]["frac_same_direction"] == pytest.approx(3 / 5)


def test_paired_corpus_wer_difference():
    ids = [f"{s}_{k}" for s in "ab" for k in range(2)]
    base = pd.DataFrame({"id": ids, "speaker": [i[0] for i in ids], "word_ref_len": 10.0})
    a = base.assign(word_errors=[1.0, 1.0, 1.0, 1.0])
    b = base.assign(word_errors=[3.0, 3.0, 3.0, 5.0])
    wer = [next(m for m in agg.METRICS if m.key == "wer")]
    result = agg.paired_cell(a, b, 100, 0, ALPHA, wer)["wer"]
    assert result["point"] == pytest.approx(4 / 40 - 14 / 40) and result["frac_same_direction"] == 1.0


def test_paired_pairs_cover_vocoder_pairs_and_references(tmp_path, monkeypatch):
    set_env(monkeypatch, tmp_path)
    systems = load_systems(toy_config(tmp_path))
    pairs = agg.paired_pairs(systems, ["vocos_mel", "enplus16"])
    names = [(a.name, ca, b.name, cb) for a, ca, b, cb in pairs]
    assert ("hifigan", "T2", "ddsp", "T2") in names and ("ddsp", "T3", "vocos", "T3") in names
    assert ("hifigan", "T1", "vocos_mel", "T1") in names and ("hifigan", "T2", "vocos_mel", "T1") in names  # copy synthesis has T1 only
    assert ("vocos", "T2", "enplus16", "T2") in names and ("vocos", "T3", "enplus16", "T1") in names
    assert len([n for n in names if n[2] in ("ddsp", "vocos")]) == 9  # 3 vocoder pairs x 3 conditions
    assert not any(n[0] == "gt" or n[2] == "gt" for n in names)


# ------------------------------------------------------------------------------------------------- strata


def test_t2_strata_split_by_chapter_of_the_reference():
    frame = pd.DataFrame(
        {
            "id": ["1_1_a", "1_1_b", "1_2_c"],
            "speaker": "1",
            "chapter": ["1", "1", "2"],
            "ref_id": ["1_1_z", "1_2_y", "1_2_x"],
        }
    )
    strata = agg.with_strata(frame, "T2", True)
    assert list(strata) == ["all", "same_chapter", "diff_chapter"]
    assert strata["same_chapter"]["id"].tolist() == ["1_1_a", "1_2_c"] and strata["diff_chapter"]["id"].tolist() == ["1_1_b"]
    assert list(agg.with_strata(frame, "T1", True)) == ["all"] and list(agg.with_strata(frame, "T2", False)) == ["all"]


# ------------------------------------------------------------------------------------------------- speaker similarity


def test_speaker_reference_excludes_target_and_t2_reference():
    gt = {f"s_{k}": np.eye(4)[k] for k in range(4)}  # four orthonormal gt embeddings of one speaker
    gt["t_0"] = np.array([0.0, 0, 0, 1.0])
    ids = ["s_0", "s_1", "s_2", "s_3", "t_0"]
    speakers = ["s", "s", "s", "s", "t"]
    refs = ["s_1", "s_0", "s_3", "s_3", ""]  # T2 reference of s_3 is itself (no double exclusion)
    mean = agg.speaker_references(ids, speakers, refs, gt)
    assert np.allclose(mean[0], (gt["s_2"] + gt["s_3"]) / 2)  # target s_0 and its reference s_1 left out
    assert np.allclose(mean[2], (gt["s_1"] + gt["s_0"]) / 2)  # target s_2 and reference s_3 left out
    assert np.allclose(mean[3], (gt["s_0"] + gt["s_1"] + gt["s_2"]) / 3)
    assert np.isnan(mean[4]).all()  # a speaker with a single utterance has nothing left
    gt["s_2"] = np.full(4, np.nan)  # a failed embedding does not count
    assert np.allclose(agg.speaker_references(ids, speakers, refs, gt)[0], gt["s_3"])


def test_similarity_columns_ceiling_and_same_utterance():
    rng = np.random.default_rng(0)
    speaker_vec = rng.normal(size=8)
    ids = [f"s_{k}" for k in range(5)]
    gt = {i: speaker_vec + 0.05 * rng.normal(size=8) for i in ids}
    frame = pd.DataFrame({"id": ids, "speaker": "s", "t2_ref": [ids[1], ids[0], ids[0], ids[0], ids[0]]})
    same, spk = agg.similarity_columns(frame, gt, gt, is_gt=True)
    assert np.isnan(same).all() and (spk > 0.95).all()  # gt against itself: no sim_same, the ceiling for sim_spk
    same, spk = agg.similarity_columns(frame, {i: v * 3.0 for i, v in gt.items()}, gt, is_gt=False)
    assert np.allclose(same, 1.0)  # cosine is scale free
    other = {i: -speaker_vec for i in ids}
    same, spk = agg.similarity_columns(frame, other, gt, is_gt=False)
    assert (same < -0.9).all() and (spk < -0.9).all()


# ------------------------------------------------------------------------------------------------- end to end


@pytest.fixture
def toy(tmp_path):
    build_toy_cache(tmp_path)
    return tmp_path


def fake_results(paths, systems, table, rng, skip=()):
    """Result parts and 4-d speaker embeddings for every reported system; ``skip`` lists (system, condition, group)."""
    quality = {"gt": 4.2, "vocos_mel": 3.8, "enplus16": 3.3, "hifigan": 3.0, "ddsp": 2.5, "vocos": 3.5}
    chunks_all = io.chunk_plan(table["id"].tolist(), 8)
    for spec in agg.reported_systems(systems):
        for cond in spec.conditions:
            chunks = io.chunk_for_condition(chunks_all, io.condition_ids(table, cond))
            for group in agg.GROUPS_BY_KIND[spec.kind]:
                if (spec.name, cond, group) in skip:
                    continue
                for k, ids in enumerate(chunks):
                    if not ids:
                        continue
                    frame = io.base_columns(table, ids, cond)
                    n = len(ids)
                    if group == "utmos":
                        frame["utmos"] = quality[spec.name] + rng.normal(0, 0.1, n)
                    elif group == "signal":
                        frame["pesq_wb"] = quality[spec.name] / 2 + rng.normal(0, 0.1, n)
                        frame["mcd"] = 10 - quality[spec.name] + rng.normal(0, 0.1, n)
                        frame["mel_l1_8_12k"] = np.nan if spec.sr == 16000 else 0.3
                    elif group == "asr":
                        frame["hyp"] = "x"
                        frame["word_errors"] = rng.integers(0, 3, n).astype(float)
                        frame["word_ref_len"] = 5.0
                        frame["char_errors"] = 1.0
                        frame["char_ref_len"] = 25.0
                    else:
                        frame["f0_rmse_cents"] = 50 + 20 * rng.random(n)
                        frame["ema_r_mean"] = 0.9
                    frame["err"] = ""
                    io.write_parquet(paths.results_dir(spec.name, cond, group) / io.part_name(k, "parquet"), frame)
            for k, ids in enumerate(chunks):
                if ids and (spec.name, cond, "spk") not in skip:
                    emb = np.stack([np.eye(4)[zlib.crc32(uid.encode()) % 4] + 0.3 * rng.normal(size=4) for uid in ids])
                    io.write_npz(
                        paths.embeddings_dir(spec.name, cond, "ecapa") / io.part_name(k, "npz"),
                        {"ids": np.array(ids), "emb": emb.astype(np.float32), "err": np.array([""] * len(ids))},
                    )


def run_toy_aggregate(toy, monkeypatch, skip=(), overrides=()):
    set_env(monkeypatch, toy)
    cfg = toy_config(toy, "eval.spk_models=[ecapa]", "eval.aggregate.n_boot=200", *overrides)
    systems = load_systems(cfg)
    table = io.build_utterance_table(toy / "cache", SPLIT, sex={})
    paths = io.EvalPaths(Path(cfg.paths.eval_root), Path(cfg.paths.runs_root), SPLIT, None)
    fake_results(paths, systems, table, np.random.default_rng(0), skip)
    return cfg, paths, systems, table


def test_run_aggregate_writes_the_tables(toy, monkeypatch):
    cfg, paths, systems, table = run_toy_aggregate(toy, monkeypatch)
    summary = agg.run_aggregate(cfg, paths, systems, table, table["id"].tolist())
    out = paths.tables_dir()
    assert sorted(p.name for p in out.iterdir()) == ["meta.json", "paired.csv", "per_speaker.csv", "table.csv", "table.md"]
    wide = pd.read_csv(out / "table.csv")
    rows = wide[(wide["stratum"] == "all")]
    assert sorted(rows["system"].unique()) == ["ddsp", "enplus16", "gt", "hifigan", "vocos", "vocos_mel"]  # gt_shipped has no row
    assert set(rows[rows["system"] == "enplus16"]["condition"]) == {"T1", "T2"}
    hifigan_t1 = rows[(rows["system"] == "hifigan") & (rows["condition"] == "T1")].iloc[0]
    assert hifigan_t1["n_utt"] == len(table) and hifigan_t1["n_speakers"] == 5
    assert hifigan_t1["utmos"] == pytest.approx(3.0, abs=0.1) and hifigan_t1["utmos_lo"] < hifigan_t1["utmos"] < hifigan_t1["utmos_hi"]
    assert rows[(rows["system"] == "hifigan") & (rows["condition"] == "T2")].iloc[0]["n_utt"] == len(table) - 1  # speaker 305 has no T2
    assert set(wide[wide["condition"] == "T2"]["stratum"]) == {"all", "same_chapter", "diff_chapter"}
    t2 = wide[(wide["system"] == "ddsp") & (wide["condition"] == "T2")].set_index("stratum")
    assert t2.loc["same_chapter", "n_utt"] + t2.loc["diff_chapter", "n_utt"] == t2.loc["all", "n_utt"]
    gt_row = rows[rows["system"] == "gt"].iloc[0]
    assert np.isnan(gt_row["sim_same"]) and gt_row["sim_spk"] == gt_row["sim_spk"]  # gt: ceiling only
    assert np.isnan(rows[(rows["system"] == "enplus16") & (rows["condition"] == "T1")].iloc[0]["mel_l1_8_12k"])
    assert hifigan_t1["wer"] == pytest.approx(hifigan_t1["wer"]) and 0 <= hifigan_t1["wer"] <= 0.4

    paired = pd.read_csv(out / "paired.csv")
    row = paired[
        (paired["system_a"] == "hifigan") & (paired["system_b"] == "vocos") & (paired["condition_a"] == "T1") & (paired["metric"] == "utmos")
        & (paired["stratum"] == "all")
    ].iloc[0]
    assert row["diff"] == pytest.approx(-0.5, abs=0.1) and row["ci_excludes_zero"] and row["hi"] < 0 and row["frac_speakers_same_direction"] == 1.0
    assert set(paired["system_b"]) == {"ddsp", "vocos", "vocos_mel", "enplus16"} and "gt" not in set(paired["system_a"])

    per_speaker = pd.read_csv(out / "per_speaker.csv")
    assert {"system", "condition", "speaker", "n_utt", "utmos", "wer"} <= set(per_speaker.columns)
    assert len(per_speaker[(per_speaker["system"] == "gt")]) == 5

    meta = io.read_json(out / "meta.json")
    assert meta["missing"] == [] and meta["speaker_primary"] == "ecapa" and meta["n_boot"] == 200
    assert meta["systems"]["hifigan"]["checkpoint"] is None and "fork_commit" in meta
    markdown = (out / "table.md").read_text()
    assert "single training run" in markdown and "speaker-cluster bootstrap" in markdown
    assert "enplus16" in markdown and "16 kHz" in markdown and "Every group of every system and condition is complete" in markdown
    assert summary["table"].shape[0] == len(wide)


def test_missing_groups_and_systems_are_reported_not_dropped(toy, monkeypatch):
    skip = {("ddsp", c, g) for c in ("T1", "T2", "T3") for g in ("signal", "utmos", "asr", "prosody", "spk")}
    skip |= {("vocos", "T3", "asr"), ("hifigan", "T1", "prosody")}
    cfg, paths, systems, table = run_toy_aggregate(toy, monkeypatch, skip)
    # also truncate one chunk of an existing group to make it partial
    part = paths.results_dir("hifigan", "T2", "utmos") / io.part_name(0, "parquet")
    part.unlink()
    agg.run_aggregate(cfg, paths, systems, table, table["id"].tolist())
    out = paths.tables_dir()
    meta = io.read_json(out / "meta.json")
    missing = {(m["system"], m["condition"], m["group"]): m["status"] for m in meta["missing"]}
    assert missing[("ddsp", "T1", "signal")] == "missing" and missing[("ddsp", "T3", "spk:ecapa")] == "missing"
    assert missing[("vocos", "T3", "asr")] == "missing" and missing[("hifigan", "T1", "prosody")] == "missing"
    assert missing[("hifigan", "T2", "utmos")] == "partial"
    assert ("vocos", "T1", "asr") not in missing
    wide = pd.read_csv(out / "table.csv")
    ddsp = wide[(wide["system"] == "ddsp") & (wide["stratum"] == "all")]
    assert len(ddsp) == 3 and ddsp["utmos"].isna().all() and (ddsp["utmos_n"] == 0).all()  # a row of n/a, not an absent row
    assert (ddsp["n_utt"] > 0).all()
    partial = wide[(wide["system"] == "hifigan") & (wide["condition"] == "T2") & (wide["stratum"] == "all")].iloc[0]
    assert 0 < partial["utmos_n"] < partial["n_utt"]  # computed on the utterances that have it
    markdown = (out / "table.md").read_text()
    assert "ddsp T1 signal: missing" in markdown and "hifigan T2 utmos: partial" in markdown


def test_limit_restricts_every_cell_to_the_subset(toy, monkeypatch):
    cfg, paths, systems, table = run_toy_aggregate(toy, monkeypatch)
    subset = io.select_limit_ids(table["id"].tolist(), 6, seed=0)
    agg.run_aggregate(cfg, paths, systems, table, subset)
    wide = pd.read_csv(paths.tables_dir() / "table.csv")
    assert wide[(wide["system"] == "hifigan") & (wide["condition"] == "T1")].iloc[0]["n_utt"] == 6
    assert wide["n_utt"].max() <= 6


def test_nan_values_are_listed_as_gaps_not_hidden_in_the_mean(toy, monkeypatch):
    cfg, paths, systems, table = run_toy_aggregate(toy, monkeypatch)
    part = paths.results_dir("hifigan", "T1", "utmos") / io.part_name(0, "parquet")
    frame = pd.read_parquet(part)
    frame.loc[frame.index[0], "utmos"] = np.nan
    frame.loc[frame.index[0], "err"] = "RuntimeError: boom"
    io.write_parquet(part, frame)
    agg.run_aggregate(cfg, paths, systems, table, table["id"].tolist())
    meta = io.read_json(paths.tables_dir() / "meta.json")
    gaps = {(g["system"], g["condition"], g["stratum"], g["metric"]): (g["n"], g["n_utt"]) for g in meta["metric_gaps"]}
    n_utt = len(table)
    assert gaps[("hifigan", "T1", "all", "utmos")] == (n_utt - 1, n_utt)
    assert ("vocos", "T1", "all", "utmos") not in gaps  # complete cell
    # by design: not applicable, never a gap
    assert not [k for k in gaps if k[0] == "gt" and k[3] in ("pesq_wb", "mcd", "sim_same")]
    assert ("enplus16", "T1", "all", "mel_l1_8_12k") not in gaps
    markdown = (paths.tables_dir() / "table.md").read_text()
    assert "## Utterances without a value" in markdown and f"utmos {n_utt - 1}/{n_utt}" in markdown
