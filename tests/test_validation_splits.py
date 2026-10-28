"""Validation / test speaker splits (docs/spkemb_validation_protocol.md)."""

import numpy as np
import pandas as pd
import pytest

from sparc.vocoders.eval import splits as S

SPLITS = {
    "primary_pool": {"val": "libritts_dev", "test": "libritts"},
    "cloning": {
        "val": {"libritts_dev": ["a", "b"], "vctk": ["p1", "p2"]},
        "test": {"libritts": ["c"], "vctk": ["p3", "p4"]},
    },
    "screen": {"val": {"heldout": ["libri_dev:a", "vctk:p1"]}, "test": {"heldout": ["libri_test:c"]}},
}


def test_halve_is_balanced_deterministic_and_seeded():
    ids = {"F": [f"p{i}" for i in range(10)], "M": [f"q{i}" for i in range(11)]}
    a = S.halve(ids, 0)
    assert a == S.halve(ids, 0)
    assert sum(v == "val" for k, v in a.items() if k.startswith("p")) == 5
    assert sum(v == "val" for k, v in a.items() if k.startswith("q")) == 5  # odd count: the extra speaker goes to test
    assert a != S.halve(ids, 1)


def test_globe_speaker_splits_and_leak_check():
    t = pd.DataFrame({"id": ["globe:dev-1", "globe:dev-2", "globe:test-1"], "speaker": ["x", "x", "y"]})
    assert S.globe_speaker_splits(t) == {"x": "val", "y": "test"}
    bad = pd.DataFrame({"id": ["dev-1", "test-1"], "speaker": ["x", "x"]})
    with pytest.raises(ValueError):
        S.globe_speaker_splits(bad)


def _items():
    rows = []
    for pool, spk in (("libritts_dev", ["a", "b"]), ("libritts", ["c"]), ("vctk", ["p1", "p2", "p3", "p4"])):
        for t in spk:
            for src in spk:
                rows.append({"pool": pool, "target": t, "source": src, "x": 1.0})
    return pd.DataFrame(rows)


def test_restrict_items_partitions_pools():
    it = _items()
    v, t = S.restrict_items(it, SPLITS, "val"), S.restrict_items(it, SPLITS, "test")
    assert set(v["pool"]) == {"libritts_dev", "vctk"} and set(t["pool"]) == {"libritts", "vctk"}
    assert set(v[v.pool == "vctk"]["target"]) == {"p1", "p2"} and set(t[t.pool == "vctk"]["target"]) == {"p3", "p4"}
    assert not (set(v[v.pool == "vctk"]["target"]) & set(t[t.pool == "vctk"]["target"]))
    assert S.restrict_items(it, SPLITS, "all") is it
    vs = S.restrict_items(it, SPLITS, "val", require_source=True)
    assert set(vs[vs.pool == "vctk"]["source"]) == {"p1", "p2"}


def test_restricted_summary_equals_mean_over_selected_speakers():
    from sparc.vocoders.eval.clone import speaker_means

    rng = np.random.default_rng(0)
    it = _items()
    it["m"] = rng.normal(size=len(it))
    v = S.restrict_items(it, SPLITS, "val")
    full = speaker_means(it[it.pool == "vctk"], "m")
    assert np.allclose(speaker_means(v[v.pool == "vctk"], "m").to_numpy(), full[["p1", "p2"]].to_numpy())


def test_screen_masks():
    spk = np.array(["libri_dev:a", "vctk:p1", "libri_test:c", "vctk:p9"])
    assert S.screen_masks(spk, SPLITS, "val", "heldout").tolist() == [True, True, False, False]
    assert S.screen_masks(spk, SPLITS, "test", "heldout").tolist() == [False, False, True, False]


def test_checked_in_splits_are_disjoint_and_cover_the_pools():
    sp = S.load()
    for kind in ("cloning", "screen"):
        for pool in set(sp[kind]["val"]) | set(sp[kind]["test"]):
            v, t = set(sp[kind]["val"].get(pool, [])), set(sp[kind]["test"].get(pool, []))
            assert not (v & t), (kind, pool)
    c = sp["cloning"]
    assert len(c["val"]["vctk"]) == len(c["test"]["vctk"]) == 20
    assert len(c["val"]["libritts_dev"]) == 36 and len(c["test"]["libritts"]) == 36
    assert len(c["val"]["globe_spk"]) == 400 and len(c["test"]["globe_spk"]) == 0
    assert len(c["test"]["globe_spk_test"]) == 400 and len(c["val"]["globe_spk_test"]) == 0
    assert not set(c["val"]["globe_spk"]) & set(c["test"]["globe_spk_test"])  # real-speaker pools: disjoint contributors


def test_globe_spk_test_pool_spec():
    from sparc.vocoders.eval.clone_pools import POOL_SPECS

    sp = POOL_SPECS["globe_spk_test"]
    assert sp.splits == ("test",) and sp.min_refs == 1 and sp.min_long == 2 and sp.per_speaker == 2
    assert POOL_SPECS["globe_spk"].splits == ("dev", "test") and POOL_SPECS["globe_spk"].min_long is None
