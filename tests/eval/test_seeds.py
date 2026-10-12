"""Seed evaluation set: the ``eval_systems=seeds`` config group and the main_seed1 experiment (CPU)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from sparc.vocoders.eval import aggregate as agg
from sparc.vocoders.eval import io
from sparc.vocoders.eval.systems import load_systems
from sparc.vocoders.eval.vocoder_loader import compose_vocoder_config
from test_aggregate import fake_results
from toy_split import SPLIT, build_toy_cache, set_env, toy_config

SEED_OVERRIDES = ("eval_systems=seeds",)
VOCODERS = ("hifigan", "ddsp", "vocos")


@pytest.fixture
def toy(tmp_path):
    build_toy_cache(tmp_path)
    return tmp_path


def test_seed_set_composes(toy, monkeypatch):
    set_env(monkeypatch, toy)
    cfg = toy_config(toy, *SEED_OVERRIDES)
    systems = load_systems(cfg)
    assert cfg.eval.table_name == "seeds" and list(cfg.eval.aggregate.reference_systems) == []
    assert tuple(systems) == ("gt", *VOCODERS, *(f"{v}_s1" for v in VOCODERS))
    for v in VOCODERS:
        base, second = systems[v], systems[f"{v}_s1"]
        assert (base.experiment, base.vocoder) == ("main", v)  # the existing main results are reused under the same name
        assert (second.experiment, second.vocoder) == ("main_seed1", v)
        assert base.conditions == second.conditions == ("T1", "T2", "T3")
    pairs = [tuple(p) for p in cfg.eval.aggregate.pairs]
    assert len(set(pairs)) == len(pairs) and all(a in systems and b in systems and a != b for a, b in pairs)
    for v in VOCODERS:
        assert (f"{v}_s1", v) in pairs  # the seed-to-seed pairs
    paths = io.EvalPaths(Path(cfg.paths.eval_root), Path(cfg.paths.runs_root), SPLIT, None, str(cfg.eval.table_name))
    assert paths.tables_dir() == toy / "eval" / "tables" / SPLIT / "seeds"
    assert paths.results_dir("hifigan_s1", "T1", "asr") != paths.results_dir("hifigan", "T1", "asr")


@pytest.mark.parametrize("vocoder", VOCODERS)
def test_main_seed1_differs_from_main_only_in_seed_and_name(vocoder):
    main, second = compose_vocoder_config("main", vocoder), compose_vocoder_config("main_seed1", vocoder)
    assert int(main.seed) == 0 and int(second.seed) == 1
    assert main.experiment_name == "main" and second.experiment_name == "main_seed1"
    assert second.run_dir.rstrip("/").endswith(f"main_seed1/{vocoder}")
    assert int(second.train.max_g_steps) == 400000 and int(second.data.val_subset_seed) == int(main.data.val_subset_seed)
    a, b = (OmegaConf.to_container(c, resolve=True) for c in (main, second))
    for d in (a, b):
        d.pop("seed")
    # every other difference must be the run name inside a derived path or name
    b = json.loads(json.dumps(b).replace("main_seed1", "main"))
    diff = {k for k in a if a[k] != b[k]}
    assert not diff, diff


def test_seed_aggregate_has_seed_pairs(toy, monkeypatch):
    set_env(monkeypatch, toy)
    cfg = toy_config(toy, "eval.spk_models=[ecapa]", "eval.aggregate.n_boot=100", *SEED_OVERRIDES)
    systems = load_systems(cfg)
    table = io.build_utterance_table(toy / "cache", SPLIT, sex={})
    paths = io.EvalPaths(Path(cfg.paths.eval_root), Path(cfg.paths.runs_root), SPLIT, None, str(cfg.eval.table_name))
    quality = {n: 3.0 + 0.1 * i for i, n in enumerate(systems)}
    quality["gt"] = 4.2
    fake_results(paths, systems, table, np.random.default_rng(0), quality=quality)
    agg.run_aggregate(cfg, paths, systems, table, table["id"].tolist())
    out = toy / "eval" / "tables" / SPLIT / "seeds"
    paired = pd.read_csv(out / "paired.csv")
    got = {(a, b) for a, b in zip(paired["system_a"], paired["system_b"])}
    assert got == {tuple(p) for p in cfg.eval.aggregate.pairs}
    assert io.read_json(out / "meta.json")["missing"] == []
