"""Pure helpers of the Phase 8.1 channel / input-path diagnosis (:mod:`sparc.vocoders.eval.ood_channel`)."""

import numpy as np
import pandas as pd
import pytest

from sparc.vocoders.constants import F0_CHANNEL, LOUDNESS_CHANNEL, PERIODICITY_CHANNEL
from sparc.vocoders.eval import ood_channel as oc


def make_feats(t: int, f0_values: np.ndarray, loud: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = rng.normal(size=(15, t)).astype(np.float32)
    voiced = np.zeros(t, dtype=bool)
    voiced[: len(f0_values)] = True
    f[F0_CHANNEL] = 0.0
    f[F0_CHANNEL, voiced] = f0_values
    f[PERIODICITY_CHANNEL] = np.where(voiced, 0.8, 0.0)
    f[LOUDNESS_CHANNEL] = loud
    return f


def test_stretch_endpoints_and_length():
    x = np.array([1.0, 3.0, 5.0])
    y = oc.stretch(x, 9)
    assert len(y) == 9 and y[0] == 1.0 and y[-1] == 5.0
    assert np.all(np.diff(y) >= 0)
    assert np.allclose(oc.stretch(np.array([2.0]), 4), 2.0)


def test_swap_contour_keeps_voicing_and_articulators():
    target = make_feats(40, np.full(20, 100.0), 0.1, seed=1)
    other = make_feats(60, np.linspace(150.0, 250.0, 30), 0.3, seed=2)
    out = oc.swap_contour(target, other)
    assert np.array_equal(out[:12], target[:12])
    assert np.array_equal(out[PERIODICITY_CHANNEL], target[PERIODICITY_CHANNEL])
    v = target[F0_CHANNEL] > 0
    assert np.array_equal(out[F0_CHANNEL] > 0, v)
    assert out[F0_CHANNEL, v][0] == pytest.approx(150.0, rel=1e-4) and out[F0_CHANNEL, v][-1] == pytest.approx(250.0, rel=1e-4)
    assert np.allclose(out[LOUDNESS_CHANNEL], 0.3)


def test_swap_level_keeps_contour_shape():
    target = make_feats(40, np.linspace(100.0, 140.0, 20), 0.1, seed=1)
    other = make_feats(60, np.full(30, 200.0), 0.4, seed=2)
    out = oc.swap_level(target, other)
    v = target[F0_CHANNEL] > 0
    ratio = out[F0_CHANNEL, v] / target[F0_CHANNEL, v]
    assert np.allclose(ratio, ratio[0], rtol=1e-4)  # a pure scaling of the contour
    assert np.median(out[F0_CHANNEL, v]) == pytest.approx(200.0, rel=1e-3)
    assert np.allclose(out[LOUDNESS_CHANNEL][v], 0.4, rtol=1e-4)


def test_donor_is_other_speaker_and_deterministic():
    items = [{"own_id": f"i{k}", "speaker": f"s{k % 3}"} for k in range(12)]
    for o in items:
        d = oc.donor_item(items, o)
        assert d["speaker"] != o["speaker"] and d == oc.donor_item(items, o)


def _rows(n_spk: int = 6) -> pd.DataFrame:
    rows = []
    for s in range(n_spk):
        base = 0.4 + 0.02 * s
        r = {"speaker": f"s{s}", "n_held": 3}
        sims = {"std": base, "gt": base + 0.4, "chan_vocos": base + 0.3, "chan_vocos_lm": base + 0.25, "ctrl_contour": base - 0.03, "ctrl_level": base - 0.01,
                "swap_content": base - 0.2}
        for c, v in sims.items():
            r[f"{c}/utmos"] = 3.0
            for m in oc.MODELS:
                r[f"{c}/{m}/held"] = v
                r[f"{c}/{m}/all"] = v
        rows.append(r)
    return pd.DataFrame(rows)


def test_selection_and_rule():
    df = _rows()
    choice = oc.select_variants({"vctk": df})
    assert choice["channel"] == "chan_vocos" and choice["control"] == "ctrl_contour"
    res = oc.summarize_pool(df, n_boot=200)
    rule = oc.apply_rule(res, choice["channel"], choice["control"])
    assert rule["verdict"] == "mixed"  # (0.4+0.3)/(0.4+0.4) ~ 0.9 of ground truth
    assert rule["iii"]["moves"] and not rule["ii"]["moves"]
    assert rule["content_path_limits_gap"]
