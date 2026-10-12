"""Ablation evaluation set: the ``eval_systems=ablations`` config group and the separate table directory (CPU)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from sparc.vocoders.eval import aggregate as agg
from sparc.vocoders.eval import io
from sparc.vocoders.eval.systems import compose_config, load_systems
from sparc.vocoders.eval.vocoder_loader import compose_vocoder_config
from test_aggregate import fake_results
from toy_split import SPLIT, build_toy_cache, set_env, toy_config

ABLATION_OVERRIDES = ("eval_systems=ablations",)
MAIN = ("gt", "gt_shipped", "vocos_mel", "enplus16", "hifigan", "ddsp", "vocos")


@pytest.fixture
def toy(tmp_path):
    build_toy_cache(tmp_path)
    return tmp_path


def test_default_systems_and_table_path_are_unchanged(toy, monkeypatch):
    set_env(monkeypatch, toy)
    cfg = toy_config(toy)
    assert tuple(load_systems(cfg)) == MAIN
    assert cfg.eval.table_name == "main" and cfg.eval.aggregate.pairs is None
    assert list(cfg.eval.aggregate.reference_systems) == ["vocos_mel", "enplus16"]
    paths = io.EvalPaths(Path(cfg.paths.eval_root), Path(cfg.paths.runs_root), SPLIT, None, str(cfg.eval.table_name))
    assert paths.tables_dir() == toy / "eval" / "tables" / SPLIT and paths.samples_dir() == toy / "eval" / "samples"


def test_ablation_group_defines_distinct_systems(toy, monkeypatch):
    set_env(monkeypatch, toy)
    cfg = toy_config(toy, *ABLATION_OVERRIDES)
    systems = load_systems(cfg)
    assert cfg.eval.table_name == "ablations" and list(cfg.eval.aggregate.reference_systems) == []
    assert systems["gt"].kind == "gt"
    vocoders = {n: s for n, s in systems.items() if s.kind == "vocoder"}
    assert len(vocoders) == 12 and not set(vocoders) & set(MAIN)  # no result directory is shared with the main systems
    assert all(s.conditions == ("T1", "T2", "T3") for s in vocoders.values())
    assert len({(s.experiment, s.vocoder) for s in vocoders.values()}) == 12
    assert {s.experiment for s in vocoders.values()} == {"ablation_100k", "abl_pcross0", "abl_native", "abl_msd"}
    for name, (experiment, vocoder) in {"a100k_vocos_a": ("ablation_100k", "vocos_a"), "msd_ddsp": ("abl_msd", "ddsp")}.items():
        assert (systems[name].experiment, systems[name].vocoder) == (experiment, vocoder)
    pairs = [tuple(p) for p in cfg.eval.aggregate.pairs]
    assert all(a in vocoders and b in vocoders for a, b in pairs) and len(set(pairs)) == len(pairs)
    for variant in ("a100k_vocos_a", "pcross0_hifigan", "native_hifigan", "msd_hifigan", "msd_vocos", "msd_ddsp"):
        assert any(a == variant for a, _ in pairs)  # every ablation is compared with a baseline
    paths = io.EvalPaths(Path(cfg.paths.eval_root), Path(cfg.paths.runs_root), SPLIT, None, str(cfg.eval.table_name))
    assert paths.tables_dir() == toy / "eval" / "tables" / SPLIT / "ablations"
    assert paths.samples_dir() == toy / "eval" / "samples_ablations"
    assert paths.results_dir("msd_ddsp", "T1", "asr") != paths.results_dir("ddsp", "T1", "asr")


def test_table_name_can_be_set_without_the_group(toy, monkeypatch):
    set_env(monkeypatch, toy)
    cfg = toy_config(toy, "eval.table_name=other")
    paths = io.EvalPaths(Path(cfg.paths.eval_root), Path(cfg.paths.runs_root), SPLIT, 7, str(cfg.eval.table_name))
    assert paths.tables_dir() == toy / "eval" / "smoke_7" / "tables" / SPLIT / "other"


@pytest.mark.parametrize(
    "experiment, vocoder, loss, p_cross",
    [("ablation_100k", "vocos_a", "mrd", 0.5), ("abl_pcross0", "hifigan", None, 0.0), ("abl_native", "hifigan", None, None),
     ("abl_msd", "ddsp", None, None)],
)
def test_ablation_experiments_compose(experiment, vocoder, loss, p_cross):
    cfg = compose_vocoder_config(experiment, vocoder)
    assert cfg.experiment_name == experiment and int(cfg.train.max_g_steps) == 100000
    assert cfg.run_dir.rstrip("/").endswith(f"{experiment}/{vocoder}")
    if p_cross is not None:
        assert float(cfg.speaker.p_cross) == p_cross


def test_paired_pairs_only_and_unknown_names(toy, monkeypatch):
    set_env(monkeypatch, toy)
    systems = load_systems(toy_config(toy))
    default = agg.paired_pairs(systems, [])
    assert {(a.name, b.name) for a, _, b, _ in default} == {("hifigan", "ddsp"), ("hifigan", "vocos"), ("ddsp", "vocos")}
    only = agg.paired_pairs(systems, [], [["vocos", "hifigan"]])
    assert [(a.name, c, b.name, d) for a, c, b, d in only] == [("vocos", c, "hifigan", c) for c in ("T1", "T2", "T3")]
    with pytest.raises(ValueError, match="unknown systems"):
        agg.paired_pairs(systems, [], [["vocos", "nope"]])


def test_ablation_aggregate_writes_a_separate_directory(toy, monkeypatch):
    set_env(monkeypatch, toy)
    cfg = toy_config(toy, "eval.spk_models=[ecapa]", "eval.aggregate.n_boot=100", *ABLATION_OVERRIDES)
    systems = load_systems(cfg)
    table = io.build_utterance_table(toy / "cache", SPLIT, sex={})
    paths = io.EvalPaths(Path(cfg.paths.eval_root), Path(cfg.paths.runs_root), SPLIT, None, str(cfg.eval.table_name))
    quality = {n: 3.0 + 0.1 * i for i, n in enumerate(systems)}
    quality["gt"] = 4.2
    fake_results(paths, systems, table, np.random.default_rng(0), quality=quality)
    main_dir = toy / "eval" / "tables" / SPLIT
    main_dir.mkdir(parents=True)
    (main_dir / "table.md").write_text("main table")
    agg.run_aggregate(cfg, paths, systems, table, table["id"].tolist())
    assert (main_dir / "table.md").read_text() == "main table"  # the main tables are never touched
    out = main_dir / "ablations"
    assert sorted(p.name for p in out.iterdir()) == ["meta.json", "paired.csv", "per_speaker.csv", "table.csv", "table.md"]
    meta = io.read_json(out / "meta.json")
    assert meta["table_name"] == "ablations" and meta["missing"] == []
    assert "(ablations)" in (out / "table.md").read_text().splitlines()[0]
    paired = pd.read_csv(out / "paired.csv")
    got = {(a, b) for a, b in zip(paired["system_a"], paired["system_b"])}
    assert got == {tuple(p) for p in cfg.eval.aggregate.pairs}
    row = paired[(paired["system_a"] == "pcross0_hifigan") & (paired["system_b"] == "a100k_hifigan") & (paired["condition_a"] == "T3")]
    assert not row.empty and set(row["condition_b"]) == {"T3"}
    wide = pd.read_csv(out / "table.csv")
    assert set(wide["system"]) == set(systems)
