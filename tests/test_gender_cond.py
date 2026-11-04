"""Gender-conditioning study: draws, label plumbing, zero-init FiLM, training-loop equality, ridge statistics, rule logic."""

import json

import numpy as np
import pytest
import torch

from sparc.compression import gender_cond as gc
from sparc.compression import head_screen
from sparc.compression.probe import Ridge
from sparc.compression.seqheads import CascadeHead

SPK = gc.SPEAKERS


# ---------------------------------------------------------------- draws

def test_draws_deterministic():
    assert json.dumps(gc.make_draws(), sort_keys=True) == json.dumps(gc.make_draws(), sort_keys=True)


@pytest.mark.parametrize("h,n3", [("usc_M1", 4), ("usc_F1", 8), ("usc_M3", 4), ("usc_F5", 8), ("5emo_jn", 10), ("5emo_jr", 10), ("5emo_kf", 4)])
def test_a3_candidates_mixed_and_corpus_matched(h, n3):
    c = gc.a3_candidates(h)
    assert len(c) == n3
    a2 = gc.a2_set(h)
    k = sum(gc.CORPUS[s] == gc.CORPUS[h] for s in a2)
    for sub in c:
        assert h not in sub and len(sub) == len(a2) and set(sub) != set(a2)
        same = sum(gc.GENDER[s] == gc.GENDER[h] for s in sub)
        assert 0 < same < len(sub)  # truly mixed
        assert sum(gc.CORPUS[s] == gc.CORPUS[h] for s in sub) == k


def test_a4_candidates_exclude_true_partition_and_keep_group_sizes():
    d = gc.make_draws()
    for h in SPK:
        c = gc.a4_candidates(h)
        assert len(c) == (14 if gc.GENDER[h] == "M" else 19)
        real_group = {s for s in SPK if gc.GENDER[s] == gc.GENDER[h]}
        for S in c:
            assert set(S) | {h} != real_group
        for dr in d["folds"][h]["A4"]:
            lab = dr["labels"]
            assert lab[h] == 1 and sum(lab.values()) == len(real_group)  # h's group has the real gender group's size
            assert {s for s in SPK if lab[s] == lab[h]} != real_group
        assert len({tuple(x["group_with_h"]) for x in d["folds"][h]["A4"]}) == gc.D_A4  # without replacement
        assert d["folds"][h]["A4c_labels"][h] == 1


def test_a3_draws_use_every_candidate_before_repeating():
    d = gc.make_draws()
    for h in SPK:
        n = d["folds"][h]["A3_n_candidates"]
        assert d["folds"][h]["A3_n_unique"] == min(n, gc.D_A3)
        assert all(x["n_corpus_mates"] == sum(gc.CORPUS[s] == gc.CORPUS[h] for s in gc.a2_set(h)) for x in d["folds"][h]["A3"])


def test_specs_cover_draws_and_keys_unique():
    d = json.loads(json.dumps(gc.make_draws()))
    keys = set()
    for sp in gc.all_specs(d):
        k = (sp.arm, sp.key)
        assert k not in keys
        keys.add(k)
        assert sp.fold not in sp.train
    assert sum(sp.arm == "A4" for sp in gc.all_specs(d)) == 7 * gc.D_A4


# ---------------------------------------------------------------- head and training loop

def _toy(n=6, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    X = [rng.standard_normal((int(rng.integers(20, 40)), dim)).astype(np.float32) for _ in range(n)]
    Y = [rng.standard_normal((len(x), 12)) for x in X]
    return X, Y


def test_film_zero_init_equals_cascade_head():
    torch.manual_seed(3)
    a = CascadeHead(16, "gru_conv", kernel=9, smooth=False).eval()
    torch.manual_seed(3)
    b = gc.FiLMGRUConv(16).eval()
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sb) - set(sa) == {"film_front", "film_rnn"}
    assert all(torch.equal(sa[k], sb[k]) for k in sa)  # the random stream is untouched by the tables
    x = torch.randn(3, 25, 16)
    mask = torch.zeros(3, 25, dtype=torch.bool)
    mask[1, 20:] = True
    with torch.no_grad():
        assert torch.equal(a(x, mask), b(x, mask, torch.tensor([0, 1, 1])))
        assert torch.equal(a(x, mask), b(x, mask))


def test_film_table_changes_output_only_for_its_label():
    torch.manual_seed(0)
    b = gc.FiLMGRUConv(16).eval()
    with torch.no_grad():
        b.film_front[1] += 0.5
        x = torch.randn(2, 20, 16)
        y0 = b(x, None, torch.tensor([0, 0]))
        y1 = b(x, None, torch.tensor([1, 1]))
        ym = b(x, None, torch.tensor([0, 1]))
    assert not torch.allclose(y0, y1)
    assert torch.allclose(ym[0], y0[0], atol=1e-6) and torch.allclose(ym[1], y1[1], atol=1e-6)


def test_param_groups_film_never_decayed_and_others_as_head_screen():
    h = gc.FiLMGRUConv(16)
    g = gc.param_groups_g(h)
    nodecay = {id(p) for p in g[1]["params"]}
    assert id(h.film_front) in nodecay and id(h.film_rnn) in nodecay
    ref = head_screen.param_groups(CascadeHead(16, "gru_conv", kernel=9, smooth=False))
    assert len(g[0]["params"]) == len(ref[0]["params"]) and len(g[1]["params"]) == len(ref[1]["params"]) + 2


def test_train_one_g_without_labels_equals_head_screen_train_one():
    X, Y = _toy()
    Yall = np.concatenate(Y)
    ym, ys = Yall.mean(0), Yall.std(0)
    ref, v_ref = head_screen.train_one(gc.HEAD, False, 1e-3, {"train": (X, Y), "valid": (X[:2], Y[:2])}, ym, ys, "cpu", seed=1, epochs=2)
    new, v_new, info = gc.train_one_g(False, 1e-3, {"train": (X, Y, None), "valid": (X[:2], Y[:2], None)}, ym, ys, "cpu", seed=1, epochs=2)
    assert v_ref == v_new and info["epochs"] >= 1
    for k, t in ref.state_dict().items():
        assert torch.equal(t, new.state_dict()[k]), k


def test_train_one_g_labelled_first_step_matches_unconditioned_and_then_learns_tables():
    X, Y = _toy()
    Yall = np.concatenate(Y)
    ym, ys = Yall.mean(0), Yall.std(0)
    g = [0, 1, 0, 1, 0, 1]
    head, v, info = gc.train_one_g(True, 1e-3, {"train": (X, Y, g), "valid": (X[:2], Y[:2], g[:2])}, ym, ys, "cpu", seed=1, epochs=2)
    assert isinstance(head, gc.FiLMGRUConv) and torch.isfinite(torch.tensor(v))
    assert head.film_front.abs().sum() > 0


def test_predict_g_label_order_and_batching():
    torch.manual_seed(0)
    head = gc.FiLMGRUConv(16).eval()
    with torch.no_grad():
        head.film_rnn[1] += 0.3
        head.film_front[0, 1] += 0.2
    X, _ = _toy(n=5)
    g = [1, 0, 1, 1, 0]
    ym, ys = np.zeros(12), np.ones(12)
    both = gc.predict_g(head, X, ym, ys, "cpu", g, bs=2)
    for i in range(5):
        alone = gc.predict_g(head, [X[i]], ym, ys, "cpu", [g[i]])[0]
        assert np.allclose(both[i], alone, atol=1e-4)
    swapped = gc.predict_g(head, X, ym, ys, "cpu", [1 - v for v in g], bs=2)
    assert not np.allclose(both[0], swapped[0])


def test_subset_and_labels_stay_aligned():
    split = {"X": [np.zeros((3, 2)), np.ones((4, 2)), np.zeros((5, 2))], "Y": [1, 2, 3], "stems": ["a", "b", "c"], "spk": ["usc_M1", "usc_F1", "usc_M1"]}
    sub = gc.subset(split, ["usc_M1"])
    assert sub["stems"] == ["a", "c"] and [len(x) for x in sub["X"]] == [3, 5] and sub["Y"] == [1, 3]
    lab = gc.utt_labels(sub, gc.gender_labels())
    assert lab == [1, 1] and len(lab) == len(sub["X"])
    assert gc.utt_labels(sub, None) is None


# ---------------------------------------------------------------- ridge from sufficient statistics

def test_suffridge_matches_probe_ridge_plain_and_interaction():
    rng = np.random.default_rng(0)
    d = 7
    spk = ["a", "b", "c"]
    Xs = {s: rng.standard_normal((60 + 10 * i, d)) + i for i, s in enumerate(spk)}
    Ys = {s: rng.standard_normal((len(Xs[s]), 12)) + 0.3 * i for i, s in enumerate(spk)}
    stats = {s: gc.speaker_stats([Xs[s]], [Ys[s]], "cpu") for s in spk}
    g = {"a": 1, "b": 0, "c": 1}
    for blocks, feat in (([{s: 1 for s in spk}], lambda s: Xs[s]),
                         ([{s: 1 for s in spk}, g], lambda s: np.concatenate([Xs[s], g[s] * Xs[s]], 1))):
        st = gc.combine(stats, spk, blocks)
        r = gc.SuffRidge(st)
        ref = Ridge([feat(s) for s in spk], [Ys[s] for s in spk], "cpu")
        for alpha in (0.5, 30.0):
            W, b = r.weights(alpha)
            W2, b2 = ref.weights(alpha)
            assert torch.allclose(W, W2, atol=1e-8) and torch.allclose(b, b2, atol=1e-8)
            P = np.concatenate([feat(s) @ W.numpy() + b.numpy() for s in spk])
            direct = np.sqrt(((P - np.concatenate([Ys[s] for s in spk])) ** 2).mean(0)).mean()
            assert abs(gc.stats_rmse(st, W, b) - direct) < 1e-9


def test_ridge_predict_blocks():
    rng = np.random.default_rng(1)
    X = [rng.standard_normal((9, 4))]
    W = torch.from_numpy(rng.standard_normal((8, 12)))
    b = torch.from_numpy(rng.standard_normal(12))
    p1 = gc.ridge_predict(X, [1, 1], W, b, "cpu")[0]
    p0 = gc.ridge_predict(X, [1, 0], W, b, "cpu")[0]
    assert np.allclose(p1, X[0] @ (W[:4] + W[4:]).numpy() + b.numpy())
    assert np.allclose(p0, X[0] @ W[:4].numpy() + b.numpy())


# ---------------------------------------------------------------- rule logic

def M(r, p, c=None):
    return {"rmse_z": r, "pcc": p, "rmse_centered": r if c is None else c}


TAU = {"rmse_z": 0.015, "pcc": 0.006}


def synthetic(a1_shift, ctrl_shift, rep_shifts, centered_shift=None):
    """A0 = 0.8 / 0.7 in every fold; A1 and the pseudo-gender runs shifted by the given rmse_z gain (PCC gain = 0.4 x)."""
    def arm(shift):
        return M(0.8 - shift, 0.7 + 0.4 * shift, 0.8 - (shift if centered_shift is None else centered_shift))

    V = {"A0": {h: {"mean": arm(0.0)} for h in SPK}, "A1": {h: {"mean": arm(a1_shift)} for h in SPK},
         "A4": {h: {"reps": [arm(s) for s in rep_shifts], "ctrl": arm(ctrl_shift)} for h in SPK},
         "A4c": {h: {"mean": arm(0.0)} for h in SPK}}
    return V


def test_primary_helps_when_clear_and_beats_every_replicate():
    V = synthetic(0.06, 0.0, [0.0, 0.01, -0.01, 0.02, 0.0, -0.02, 0.01, 0.0])
    for G in ("M", "F"):
        r = gc.analyze_primary(V, TAU, G)
        assert r["verdict"] == "helps" and r["rank_above_all"]


def test_primary_not_helps_if_a_replicate_matches_or_centered_gate_fails():
    V = synthetic(0.06, 0.0, [0.0, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert gc.analyze_primary(V, TAU, "M")["verdict"] != "helps"
    V = synthetic(0.06, 0.0, [0.0] * 8, centered_shift=-0.01)  # gain only in the per-utterance offset
    assert gc.analyze_primary(V, TAU, "F")["verdict"] != "helps"


def test_primary_hurts_mirror_and_equivalence_and_inconclusive():
    V = synthetic(-0.06, 0.0, [0.0, 0.01, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert gc.analyze_primary(V, TAU, "M")["verdict"] == "hurts"
    V = synthetic(0.004, 0.0, [0.0] * 8)
    assert gc.analyze_primary(V, TAU, "F")["verdict"] == "no detectable effect"
    V = synthetic(0.03, 0.0, [0.0] * 8)
    V["A1"]["usc_M1"]["mean"] = M(0.8, 0.7)  # one male fold without gain: 2 of 3 still a majority; a gain between tau and 2 tau
    V = synthetic(0.012, 0.0, [0.0] * 8)  # below tau on rmse_z: no wins, but not equivalent either (> 0 and PCC near tau)
    r = gc.analyze_primary(V, TAU, "M")
    assert r["verdict"] in ("no detectable effect", "inconclusive")


def test_majority_and_judge_counts():
    assert gc.majority(3) == 2 and gc.majority(4) == 3
    gains = {"a": {"rmse_z": 0.03, "pcc": 0.01, "rmse_centered": 0.0}, "b": {"rmse_z": 0.03, "pcc": 0.001, "rmse_centered": 0.0},
             "c": {"rmse_z": 0.03, "pcc": 0.01, "rmse_centered": 0.0}}
    j = gc.judge(gains, TAU, "M")
    assert j["wins"] == 2 and j["losses"] == 0 and not j["equivalent"]


def test_sign_flip_minimum_p():
    assert gc.sign_flip_p([1.0] * 7) == pytest.approx(2 / 128)
    assert gc.sign_flip_p([1.0] * 3) == pytest.approx(0.25)


def test_tau_floor_and_gain_signs():
    V = {"A0": {h: {"seed": [M(0.8, 0.7)] * 3} for h in SPK}}
    tau, _ = gc.tau_from(V)
    assert tau == {"rmse_z": 0.005, "pcc": 0.003}
    g = gc.gain(M(0.7, 0.72), M(0.8, 0.7))
    assert g["rmse_z"] == pytest.approx(0.1) and g["pcc"] == pytest.approx(0.02)


def test_calibration_counts_passes_of_fake_treatments():
    V = synthetic(0.0, 0.0, [0.0, 0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0])
    cal = gc.calibration(V, TAU)
    assert cal["M"]["replicates"] == 8 and cal["M"]["pass_contrasts_1_2"] == 2


# ---------------------------------------------------------------- end-to-end analysis on a synthetic result tree

@pytest.mark.skipif(not gc.STORED_HEADS.exists(), reason="stored head-screen result not available")
def test_analyze_on_synthetic_tree(tmp_path):
    rng = np.random.default_rng(0)
    stored = json.loads(gc.STORED_HEADS.read_text())
    draws = json.loads(json.dumps(gc.make_draws()))

    def rec(base, shift=0.0):
        return {"rmse_z": base["rmse_z"] - shift + rng.normal(0, 0.003), "pcc": base["pcc"] + 0.4 * shift + rng.normal(0, 0.002),
                "rmse_centered": base["rmse_z"] * 0.8 - shift}

    for sp in gc.all_specs(draws):
        base = {m: float(np.mean([stored[sp.fold]["seeds"][s]["test_unseen"][m] for s in "012"])) for m in ("rmse_z", "pcc")}
        shift = 0.05 if (sp.arm == "A1" and gc.GENDER[sp.fold] == "M") else 0.0
        seeds = {str(s): {"valid_rmse_z": 0.7, "epochs": 5, "steps": 100, "unseen": rec(base, shift),
                          "seen": {s2: rec(base) for s2 in gc.SPEAKERS if s2 != sp.fold},
                          **({"wrong": rec(base)} if sp.arm == "A1" else {})} for s in sp.seeds}
        d = {"arm": sp.arm, "fold": sp.fold, "key": sp.key, "train": list(sp.train), "shift": stored[sp.fold]["shift"],
             "lr": stored[sp.fold]["lr"], "seeds": seeds, "secs": 1.0}
        p = tmp_path / sp.arm / f"{sp.key}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d))
    res = gc.analyze(tmp_path)
    assert res["pipeline_check"]["ok"]
    assert res["primary"]["M"]["verdict"] == "helps"
    assert res["primary"]["F"]["verdict"] in ("no detectable effect", "inconclusive")
    assert (tmp_path / "summary.md").exists() and (tmp_path / "results.json").exists()
    assert gc.missing(tmp_path, ridge=False) == []
    assert len(gc.missing(tmp_path)) == 14


def test_failed_seed_is_incomplete_and_wrong_label_mean_skips_it():
    spec = type("S", (), {"seeds": (0, 1, 2)})()
    ok = {m: 1.0 for m in gc.METRICS}
    rec = {"seeds": {"0": {"wrong": ok}, "1": {"failed": True}, "2": {"wrong": ok}}}
    assert not gc.is_complete(rec, spec)
    assert gc.mean_metrics_wrong(rec) == gc.mean_metrics([ok, ok])
    rec["seeds"]["1"] = {"wrong": ok}
    assert gc.is_complete(rec, spec)
