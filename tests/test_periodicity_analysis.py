"""Tests of the pure helpers of sparc.vocoders.eval.periodicity_analysis (class mapping, frame masks, HNR, probes)."""

import numpy as np
import pytest

from sparc.vocoders.eval import periodicity_analysis as pa

SR = 24000


def test_phone_classes():
    expected = {
        "a": "vowels", "ɪ": "vowels", "iː": "vowels", "aj": "vowels", "əw": "vowels", "ɚ": "vowels",
        "n": "nasals", "m̩": "nasals", "ŋ": "nasals",
        "p": "stops", "kʰ": "stops", "dʲ": "stops", "ɡ": "stops", "ʔ": "stops",
        "tʃ": "affricates", "dʒ": "affricates",
        "z": "voiced_fricatives", "ð": "voiced_fricatives", "v": "voiced_fricatives",
        "s": "voiceless_fricatives", "ʃ": "voiceless_fricatives", "f": "voiceless_fricatives", "h": "h",
        "l": "approximants", "ɫ": "approximants", "ɹ": "approximants", "j": "approximants", "w": "approximants", "ɾ": "approximants",
        "": "silence", "sil": "silence", "sp": "silence",
        "spn": None,
    }
    for phone, cls in expected.items():
        assert pa.phone_class(phone) == cls, phone


def test_phone_voicing():
    expected = {"a": "voiced", "n": "voiced", "l": "voiced", "z": "voiced", "b": "voiced", "ɡ": "voiced", "dʒ": "voiced",
                "p": "voiceless", "kʰ": "voiceless", "s": "voiceless", "θ": "voiceless", "tʃ": "voiceless",
                "": "silence", "sp": "silence", "h": None, "ʔ": None, "spn": None}
    for phone, v in expected.items():
        assert pa.phone_voicing(phone) == v, phone


def test_phone_sets_and_boundaries():
    n = 50
    # silence, voiced vowel, voiceless s, h (breaks adjacency), voiced n, spn
    phones = [(0.0, 0.2, ""), (0.2, 0.4, "a"), (0.4, 0.6, "s"), (0.6, 0.7, "h"), (0.7, 0.8, "n"), (0.8, 1.0, "spn")]
    py = np.zeros(n)
    py[10:20] = 1.0
    m, _ = pa.class_masks(phones, n, np.zeros(n), py)
    assert np.array_equal(np.flatnonzero(m["phone_voiced"]), np.r_[10:20, 35:40])
    assert np.array_equal(np.flatnonzero(m["sonorants"]), np.r_[10:20, 35:40])
    assert np.array_equal(np.flatnonzero(m["phone_voiceless"]), np.arange(20, 30))
    assert np.array_equal(np.flatnonzero(m["h"]), np.arange(30, 35))
    # boundaries: silence | a at 200 ms (frames 8..11), a | s at 400 ms (frames 18..21); s | h and h | n are not boundaries
    assert np.array_equal(np.flatnonzero(m["change_phone"]), np.r_[8:12, 18:22])
    # pYIN changes at frames 10 (onset) and 20 (offset)
    assert np.array_equal(np.flatnonzero(m["change_pyin"]), np.r_[8:12, 18:22])


def test_class_masks_stops_and_transitions():
    n = 50  # 1 s of frames
    phones = [(0.0, 0.2, ""), (0.2, 0.3, "t"), (0.3, 0.6, "a"), (0.6, 0.62, "p"), (0.62, 0.8, "s"), (0.8, 1.0, "spn")]
    voiced = np.zeros(n)
    voiced[15:30] = 1.0  # voiced from frame 15 to 29
    m, info = pa.class_masks(phones, n, voiced)
    c = (np.arange(n) + 0.5) * 0.02
    assert m["silence"][:10].all() and not m["silence"][10:].any()
    # stop 200-300 ms: burst = last 40 ms (centres >= 260 ms), closure the rest
    assert np.array_equal(np.flatnonzero(m["stop_closure"]), np.arange(10, 13))
    assert np.array_equal(np.flatnonzero(m["stop_burst"]), np.array([13, 14] + [30]))  # 620 ms stop: shorter than 60 ms, all burst
    assert m["vowels"][15:30].all() and m["voiceless_fricatives"][31:40].all()
    assert info["unknown"] == 10 and info["labelled"] == 40
    # onset at frame 15 (between frames 14 and 15), offset at 30
    assert np.array_equal(np.flatnonzero(m["voicing_onset"]), np.arange(13, 17))
    assert np.array_equal(np.flatnonzero(m["voicing_offset"]), np.arange(28, 32))
    assert m["voicing_onset_offset"].sum() == 8
    assert c[13] > 0


def test_smooth_runs_removes_flicker():
    v = np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0], dtype=float)
    assert np.array_equal(pa.smooth_runs(v, 3), np.array([1.0] * 8 + [0.0] * 4))
    w = np.array([0, 1, 0, 0, 0, np.nan, np.nan], dtype=float)  # a first short run is kept, NaN runs are left alone
    out = pa.smooth_runs(w, 3)
    assert out[0] == 0 and np.all(out[1:5] == 0) and np.isnan(out[5:]).all()


def test_hnr_periodic_vs_noise():
    t = np.arange(SR) / SR
    harmonic = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 10))
    noise = np.random.default_rng(0).standard_normal(SR)
    centres = pa.frame_centres(SR // pa.HOP - 2)[5:30]
    h, nz = pa.hnr_db(harmonic, centres), pa.hnr_db(noise, centres)
    assert np.nanmedian(h) > 30.0
    assert np.nanmedian(nz) < 0.0
    mixed = harmonic / harmonic.std() + noise
    assert np.nanmedian(nz) < np.nanmedian(pa.hnr_db(mixed, centres)) < np.nanmedian(h)


def test_hf_band_db_selects_band():
    t = np.arange(SR) / SR
    low, high = np.sin(2 * np.pi * 1000 * t), np.sin(2 * np.pi * 8000 * t)
    centres = pa.frame_centres(20)[2:15]
    assert np.all(pa.hf_band_db(high, centres) - pa.hf_band_db(low, centres) > 60)


def test_probe_periodicity():
    per = np.array([0.0, 0.0, 0.9, 0.7, 0.8, 0.0, 0.0], dtype=np.float32)
    assert np.all(pa.probe_periodicity("probe_const", per, 0.437, 0.8) == np.float32(0.437))
    b = pa.probe_periodicity("probe_binary", per, 0.437, 0.8)
    assert np.allclose(b, [0, 0, 0.8, 0.8, 0.8, 0, 0])
    s = pa.probe_periodicity("probe_smooth", np.pad(per, (10, 10)), 0.437, 0.8)
    assert s.shape == per.shape[:0] + (27,) and s.max() < per.max() and s.sum() == pytest.approx(np.pad(per, (10, 10)).sum(), rel=1e-5)


def test_boot_and_paired():
    import pandas as pd

    a = pd.DataFrame({"id": list("abcd"), "speaker": ["s1", "s1", "s2", "s2"], "x": [1.0, 1.0, 3.0, 3.0]})
    b = a.assign(x=[0.0, 0.0, 2.0, 2.0])
    r = pa.paired(a, b, "x")
    assert r["diff"] == pytest.approx(1.0) and r["lo"] == pytest.approx(1.0) and r["hi"] == pytest.approx(1.0) and r["n_speakers"] == 2


def _pair(diff, half=0.001):
    return {"diff": diff, "lo": diff - half, "hi": diff + half, "a": 0.0, "b": 0.0, "n_speakers": 10}


def _gate_values(gate_frac_of_gap, guard=0.0):
    """Every quality metric has a no-periodicity gap of +0.1 against both seeds (seed spread 0.01); the gate sits at
    ``gate_frac_of_gap`` of that gap away from the reference."""
    vals = {}
    for m in pa.GATE_QUALITY:
        vals[m] = {"none_ref": _pair(0.1), "none_s1": _pair(0.09), "s1_ref": _pair(-0.01), "gate_ref": _pair(0.1 * gate_frac_of_gap)}
    for cls, col in pa.GATE_GUARDS:
        vals[f"{cls}/{col}"] = {"none_ref": _pair(-0.02), "none_s1": _pair(-0.02), "s1_ref": _pair(0.002), "gate_ref": _pair(guard)}
    return vals


def test_gate_decision_rule():
    assert pa.gate_decision(_gate_values(0.2))["suffices"]  # recovers 80 % of the gap
    assert not pa.gate_decision(_gate_values(0.8))["suffices"]  # recovers 20 %
    assert pa.gate_decision(_gate_values(0.5))["suffices"]  # exactly half passes
    assert pa.gate_decision(_gate_values(0.2, guard=-0.002))["suffices"]  # within the seed spread
    d = pa.gate_decision(_gate_values(0.2, guard=-0.01))
    assert not d["suffices"] and not d["guards"]["change_pyin/py_agree"]["pass"]
    assert pa.gate_decision(_gate_values(0.5))["borderline"] == list(pa.GATE_QUALITY)  # at the threshold
    assert pa.gate_decision(_gate_values(0.2))["borderline"] == []
    assert pa.gate_decision(_gate_values(0.2, guard=-0.01))["guard_only_failure"]
    assert not pa.gate_decision(_gate_values(0.8))["guard_only_failure"]
    v = _gate_values(0.9)
    for m in pa.GATE_QUALITY:  # a gap inside the seed spread does not enter the rule
        v[m]["none_ref"] = _pair(0.005)
        v[m]["none_s1"] = _pair(0.004)
    d = pa.gate_decision(v)
    assert not any(r["enters"] for r in d["metrics"].values()) and not d["suffices"]  # nothing to recover: no claim


def test_average_systems_per_item_mean():
    import pandas as pd

    a = pd.DataFrame({"id": ["u1", "u2", "u3"], "speaker": ["s", "s", "t"], "system": "gate", "utmos": [4.0, 3.0, np.nan], "hyp": ["x", "y", "z"]})
    b = pd.DataFrame({"id": ["u3", "u2", "u1"], "speaker": ["t", "s", "s"], "system": "gate_s1", "utmos": [3.0, 5.0, 2.0], "hyp": ["q", "q", "q"]})
    m = pa.average_systems(a, b, "gate_mean", ["id"]).set_index("id")
    assert m.loc["u1", "utmos"] == 3.0 and m.loc["u2", "utmos"] == 4.0
    assert np.isnan(m.loc["u3", "utmos"])  # missing in one seed stays missing
    assert set(m["system"]) == {"gate_mean"} and m.loc["u1", "hyp"] == "x"
    fa = pd.DataFrame({"id": ["u1", "u1"], "class": ["all", "vowels"], "speaker": "s", "system": "gate", "mel_l1": [1.0, 2.0]})
    fb = pd.DataFrame({"id": ["u1"], "class": ["vowels"], "speaker": "s", "system": "gate_s1", "mel_l1": [4.0]})
    out = pa.average_systems(fa, fb, "gate_mean", ["id", "class"])
    assert len(out) == 1 and out["mel_l1"].iloc[0] == 3.0  # only rows present in both seeds


def _frames(system, vals):
    import pandas as pd

    return pd.DataFrame([{"system": system, "id": f"u{k}", "speaker": spk, "class": "all", "mel_l1": v} for k, (spk, v) in enumerate(vals)])


def test_two_seed_mean_rule():
    """The rule on the mean of two gate seeds: the mean's gate - ref is the mean of the seeds' differences, and the verdict follows
    the mean (seed 0 alone passes, seed 1 alone fails, the mean sits on the half-gap threshold)."""
    spk = ["a", "a", "b", "b", "c", "c", "d", "d"]
    ref = _frames("ref", [(s, 4.0) for s in spk])
    s0 = _frames("gate", [(s, 4.02) for s in spk])  # seed 0: gate - ref = +0.02
    s1 = _frames("gate_s1", [(s, 4.08) for s in spk])  # seed 1: +0.08
    mean = pa.average_systems(s0, s1, pa.GATE_MEAN, ["id"])
    d = pa.paired(mean, ref, "mel_l1")
    assert d["diff"] == pytest.approx(0.05)
    d0, d1 = pa.paired(s0, ref, "mel_l1"), pa.paired(s1, ref, "mel_l1")
    assert (d0["diff"] + d1["diff"]) / 2 == pytest.approx(d["diff"])

    def decide(gate_diff):  # no-periodicity gap 0.1, spread 0.01: half-gap threshold 0.05
        v = _gate_values(0.0)
        for m in pa.GATE_QUALITY:
            v[m]["gate_ref"] = _pair(gate_diff)
        return pa.gate_decision(v)

    assert decide(d0["diff"])["suffices"] and not decide(d1["diff"])["suffices"]
    dm = decide(d["diff"])
    assert dm["suffices"] and dm["borderline"] == list(pa.GATE_QUALITY)


def test_gate_split_lines_two_seed_columns():
    v = _gate_values(0.2)
    for key in v:
        v[key]["gate_none"] = _pair(-0.05)
        v[key]["only_gate_ref"], v[key]["only_gate_s1_ref"] = _pair(0.01), _pair(0.03)
    text = "\n".join(pa.gate_split_lines("dev", v, pa.gate_decision(v), ["gate", "gate_s1"]))
    assert "mean of the two gate seeds" in text and "training seed 1 - ref" in text and "DECIDING" in text
