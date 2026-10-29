import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "spkemb" / "listening"))
from design import make_design  # noqa: E402


def toy_items():
    items = {}
    for pool, n_spk, per in (("vctk", 8, 2), ("globe", 4, 1), ("libritts", 3, 1)):
        for s in range(n_spk):
            for k in range(per):
                i = f"{pool}_{s}_{k}"
                items[i] = {"pool": pool, "speaker": f"{pool}{s}", "gender": "FM"[s % 2], "files": {r: f"{i}-{r}" for r in ("ref", "std", "fit")}}
    return items


def test_design_balance_and_checks():
    items = toy_items()
    pub, rows = make_design(items, 12, seed=1, n_vctk=8, n_globe=2, n_libritts=1)
    assert len(pub) == 12 and all(len(t) == 8 + 2 + 1 + 4 for t in pub.values())
    for lid, trials in pub.items():
        assert [t["trial_id"] for t in trials] == [f"{lid}-{k + 1:02d}" for k in range(len(trials))]
        assert all(set(t) == {"trial_id", "ref", "a", "b"} for t in trials)  # no arm labels in the public list
    test = [r for r in rows if r["check"] == ""]
    chk = [r for r in rows if r["check"] != ""]
    assert Counter(r["check"] for r in chk) == {"identical": 24, "mismatch": 24}
    assert min(int(r["pos"]) for r in chk) >= 3
    # items rated equally often (+-1) within a pool, fitted output is A about half the time per item and per listener
    cnt = Counter(r["item"] for r in test if r["pool"] == "vctk")
    assert max(cnt.values()) - min(cnt.values()) <= 1
    for i in cnt:
        a = sum(r["a_arm"] == "fit" for r in test if r["item"] == i)
        assert abs(2 * a - cnt[i]) <= 1
    for lid in pub:
        a = sum(r["a_arm"] == "fit" for r in test if r["listener"] == lid and r["pool"] == "vctk")
        assert abs(2 * a - 8) <= 1
    for r in chk:
        if r["check"] == "mismatch":
            assert items[r["item"]]["gender"] != items[r["other_item"]]["gender"]
        else:
            assert r["a_arm"] == r["b_arm"] == "std"
    for lid, trials in pub.items():
        for t in trials:  # a trial never repeats a stimulus pair of the same item arms except identical checks
            assert t["ref"] not in (t["a"], t["b"])


def test_design_deterministic():
    items = toy_items()
    a = make_design(items, 3, seed=5, n_vctk=4, n_globe=1, n_libritts=1)
    b = make_design(items, 3, seed=5, n_vctk=4, n_globe=1, n_libritts=1)
    assert a == b


def test_power_helpers():
    pytest.importorskip("scipy")
    from power import cluster_t, marginal_to_intercept  # noqa: E402

    assert abs(marginal_to_intercept(0.5, 0.7)) < 1e-6
    b = marginal_to_intercept(0.6, 0.0)
    assert abs(1 / (1 + np.exp(-b)) - 0.6) < 1e-6
    y = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    m, se = cluster_t(y, np.array([0, 0, 1, 1, 2, 2]), np.array([0, 1, 0, 1, 0, 1]))
    assert abs(m - y.mean()) < 1e-12 and se > 0
