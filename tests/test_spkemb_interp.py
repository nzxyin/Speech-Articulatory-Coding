"""Speaker-embedding Phase 3 evaluation harness (``sparc.vocoders.eval.interp``): pure functions with fakes."""

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from sparc.vocoders.constants import F0_CHANNEL, N_FEATURES, PERIODICITY_CHANNEL, SAMPLE_RATE
from sparc.vocoders.eval import interp
from sparc.vocoders.models.speaker import LayerPoolSpeakerEncoder, SpeakerFFN

A = np.array(interp.ALPHAS)


def fake_table(n_speakers: int = 12, per_speaker: int = 14) -> tuple[pd.DataFrame, dict[str, str]]:
    rows, sex = [], {}
    rng = np.random.default_rng(0)
    for s in range(n_speakers):
        spk = str(100 + s)
        sex[spk] = "F" if s % 2 == 0 else "M"
        for u in range(per_speaker):
            n24 = int(rng.integers(2, 10) * SAMPLE_RATE)
            rows.append({"id": f"{spk}_1_{u:06d}_000000", "speaker": spk, "n24": n24})
    return pd.DataFrame(rows), sex


# ------------------------------------------------------------------------------------------------------ plan


def test_plan_is_deterministic_and_structured():
    table, sex = fake_table()
    kw = dict(seed=3, n_same=5, n_cross=5, n_swap=12, n_prior=9)
    p1 = interp.build_plan(table, sex, **kw)
    p2 = interp.build_plan(table.sample(frac=1.0, random_state=1), sex, **kw)
    assert p1 == p2
    assert interp.build_plan(table, sex, **{**kw, "seed": 4}) != p1
    pairs = p1["pairs"]
    assert len(pairs) == 10 and sum(p["same_gender"] for p in pairs) == 5
    assert len({frozenset((p["a"], p["b"])) for p in pairs}) == 10
    for p in pairs:
        assert (sex[p["a"]] == sex[p["b"]]) == p["same_gender"]
        assert p["content_speaker"] not in (p["a"], p["b"])
        n24 = int(table.set_index("id").loc[p["content_id"], "n24"])
        assert 3 * SAMPLE_RATE <= n24 <= 8 * SAMPLE_RATE
        assert p["content_id"].startswith(p["content_speaker"])
    for s, ids in p1["refs"].items():
        assert ids == sorted(ids) and len(ids) <= 10
        assert (table.set_index("id").loc[ids, "n24"] >= 3 * SAMPLE_RATE).all()


def test_plan_swaps_prior_and_real():
    table, sex = fake_table()
    plan = interp.build_plan(table, sex, seed=0, n_same=3, n_cross=3, n_swap=20, n_prior=15)
    refs = {u for r in plan["refs"].values() for u in r}
    assert len(plan["swaps"]) == 20
    for k, s in enumerate(plan["swaps"]):
        assert s["donor"] != s["target_speaker"] and s["target_id"] not in refs
        assert (sex[s["donor"]] == sex[s["target_speaker"]]) == (k % 2 == 0) == s["same_gender"]
    assert len(plan["prior"]) == 15 and len(plan["real_ids"]) == 15
    first = [i.split("_")[0] for i in plan["real_ids"][:12]]
    assert len(set(first)) == 12  # round robin over distinct speakers first


def test_ineligible_speakers_are_excluded():
    table, sex = fake_table(6, 14)
    table.loc[table.speaker == "100", "n24"] = SAMPLE_RATE  # all too short
    del sex["101"]
    plan = interp.build_plan(table, sex, n_same=1, n_cross=1, n_swap=2, n_prior=2)
    assert plan["speakers"] == ["102", "103", "104", "105"]


def test_prior_codes_seeded():
    z = interp.prior_codes(5, seed=1)
    assert z.shape == (5, 64) and np.array_equal(z, interp.prior_codes(5, seed=1))
    assert not np.array_equal(z, interp.prior_codes(5, seed=2))


def test_select_references():
    ids = ["b", "a", "d", "c"]
    n24 = [100000, 10, 100000, 100000]
    assert interp.select_references(ids, n24, n_refs=2) == ["b", "c"]


# ------------------------------------------------------------------------------------------------------ F0


def test_f0_scale_factor_endpoints_and_geometric_mean():
    assert interp.f0_scale_factor(0.0, 100.0, 200.0, 100.0) == pytest.approx(1.0)
    assert interp.f0_scale_factor(1.0, 100.0, 200.0, 100.0) == pytest.approx(2.0)
    assert interp.f0_scale_factor(0.5, 100.0, 400.0, 100.0) == pytest.approx(2.0)  # geometric, not arithmetic
    assert interp.f0_scale_factor(0.5, 100.0, 400.0, 200.0) == pytest.approx(1.0)


def test_scale_f0_only_voiced_frames():
    f = np.zeros((N_FEATURES, 4), np.float32)
    f[F0_CHANNEL] = [100, 120, 130, 140]
    f[PERIODICITY_CHANNEL] = [0.9, 0.0, 0.5, 0.0]
    g = interp.scale_f0(f, 2.0)
    assert g[F0_CHANNEL].tolist() == [200, 120, 260, 140]
    assert f[F0_CHANNEL].tolist() == [100, 120, 130, 140]  # input untouched
    assert np.array_equal(g[:F0_CHANNEL], f[:F0_CHANNEL])


def test_median_voiced_f0():
    a = np.zeros((3, N_FEATURES))
    a[:, F0_CHANNEL] = [100, 999, 300]
    a[:, PERIODICITY_CHANNEL] = [1, 0, 1]
    b = np.zeros((1, N_FEATURES))
    b[:, F0_CHANNEL], b[:, PERIODICITY_CHANNEL] = 200, 1
    assert interp.median_voiced_f0([a, b]) == 200.0
    assert math.isnan(interp.median_voiced_f0([np.zeros((2, N_FEATURES))]))


def test_build_items_counts_and_factors():
    table, sex = fake_table()
    plan = interp.build_plan(table, sex, n_same=2, n_cross=2, n_swap=6, n_prior=4)
    med = {s: 100.0 + 10 * i for i, s in enumerate(plan["speakers"])}
    items = interp.build_items(plan, med, ("interp", "swap", "prior"))
    assert len([i for i in items if i.kind == "interp"]) == 4 * 2 * 9
    assert len([i for i in items if i.kind == "swap"]) == 6 * 2
    assert len([i for i in items if i.kind == "prior"]) == 4
    assert len({i.key for i in items}) == len(items)
    p = plan["pairs"][0]
    sc = {i.spec["alpha"]: i for i in items if i.key.startswith(f"interp/{p['pair_id']}_scaled")}
    assert sc[0.0].f0_factor == pytest.approx(med[p["a"]] / med[p["content_speaker"]])
    assert sc[1.0].f0_factor == pytest.approx(med[p["b"]] / med[p["content_speaker"]])
    uns = [i for i in items if i.key.startswith(f"interp/{p['pair_id']}_unscaled")]
    assert all(i.f0_factor == 1.0 for i in uns)
    assert [i.meta["asr"] for i in uns] == [k in (0, 4, 8) for k in range(9)]
    assert not [i for i in interp.build_items(plan, med, ("prior",), vae=False)]
    assert len(interp.build_items(plan, med, ("interp",), variants=("scaled",))) == 4 * 9


# ------------------------------------------------------------------------------------------------------ metrics


def test_ideal_linear_path():
    d = -1.0 + 2.0 * A  # SIM_B - SIM_A moves evenly from -1 to +1
    m = interp.interp_metrics(d)
    assert m["spearman"] == pytest.approx(1.0)
    assert m["frac_increasing"] == 1.0
    assert m["evenness"] == pytest.approx(1 / 8)
    assert m["range"] == pytest.approx(2.0)
    assert m["crossing"] == pytest.approx(0.0)
    assert interp.interp_score(m["spearman"], m["evenness"], 0.0) == pytest.approx(1.0)


def test_step_path_is_uneven_and_off_center():
    d = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1], float)  # all the change in one step
    m = interp.interp_metrics(d)
    assert m["evenness"] == pytest.approx(1.0)
    assert m["crossing"] == pytest.approx(9 / 16 - 0.5)
    # a* lies between a = 4/8 and 5/8 (midpoint 9/16)
    assert interp.crossing_point(d) == pytest.approx(9 / 16)
    s = interp.interp_score(m["spearman"], m["evenness"], 0.0)
    assert s == pytest.approx((m["spearman"] + 0.0 + 1.0) / 3)


def test_flat_decreasing_and_no_crossing():
    assert math.isnan(interp.interp_metrics(np.full(9, 0.3))["spearman"])
    m = interp.interp_metrics(np.linspace(1, -1, 9))
    assert m["spearman"] == pytest.approx(-1.0) and m["frac_increasing"] == 0.0 and math.isnan(m["evenness"])
    pos = interp.interp_metrics(np.linspace(0.2, 1.0, 9))
    assert math.isnan(pos["crossing"]) and pos["evenness"] == pytest.approx(1 / 8)
    assert interp.interp_score(math.nan, math.nan, 0.0) == pytest.approx(1 / 3)  # failed path: two zero terms
    assert math.isnan(interp.interp_score(1.0, 0.125, math.nan))
    assert all(math.isnan(v) for v in interp.interp_metrics([0, math.nan, 1, 1, 1, 1, 1, 1, 1]).values())


def test_crossing_exact_zero_and_interpolation():
    assert interp.crossing_point([-1, -0.5, 0.0, 0.5, 1, 1, 1, 1, 1]) == pytest.approx(0.25)
    assert interp.crossing_point([-2, -1, 1, 2, 3, 4, 5, 6, 7]) == pytest.approx(0.1875)


def test_quality_dip_and_score_clipping():
    assert interp.quality_dip([4, 3, 2, 3, 4]) == pytest.approx(-2.0)
    assert interp.quality_dip([3, 3.5, 4]) == pytest.approx(3 - 3.5)  # path min is the endpoint
    assert interp.quality_dip([3, 4, 5, 3.5]) == pytest.approx(3 - 3.25)
    assert interp.interp_score(1.0, 0.125, -2.0) == pytest.approx((1 + 1 + 0) / 3)  # 1 + dip clipped at 0
    assert interp.interp_score(1.0, 5.0, 0.5) == pytest.approx((1 + 0 + 1) / 3)  # evenness term clipped at 0


def test_spearman_ties_and_rank():
    assert interp.spearman([1, 2, 3], [10, 20, 15]) == pytest.approx(0.5)
    assert interp.spearman([1, 1, 2, 2], [1, 2, 3, 4]) == pytest.approx(0.8944, abs=1e-3)


def test_bootstrap_ci():
    rng = np.random.default_rng(0)
    x = rng.normal(2.0, 1.0, 40)
    ci = interp.bootstrap_ci(x, n_boot=2000, seed=1)
    assert ci == interp.bootstrap_ci(x, n_boot=2000, seed=1)
    assert ci["mean"] == pytest.approx(x.mean()) and ci["lo"] < ci["mean"] < ci["hi"] and ci["n"] == 40
    assert ci["hi"] - ci["lo"] == pytest.approx(2 * 1.96 / math.sqrt(40), rel=0.3)
    with_nan = interp.bootstrap_ci(np.r_[x, np.nan], n_boot=500)
    assert with_nan["n"] == 40 and with_nan["mean"] == pytest.approx(x.mean())
    assert interp.bootstrap_ci([math.nan, math.nan])["n"] == 0
    const = interp.bootstrap_ci([1.0] * 5, n_boot=100)
    assert const["lo"] == const["hi"] == 1.0


def test_rate_ci_is_corpus_level():
    r = interp.rate_ci([1, 0, 4], [10, 10, 20], n_boot=500)
    assert r["mean"] == pytest.approx(5 / 40)
    assert math.isnan(interp.rate_ci([], [])["mean"])


def test_diversity_and_novelty():
    e = np.eye(4)
    assert interp.pairwise_diversity(e) == pytest.approx(1.0)
    assert interp.pairwise_diversity(np.ones((3, 4))) == pytest.approx(0.0)
    assert interp.pairwise_diversity(e, np.array([0, 0, 1])) == pytest.approx(1.0)  # the repeated row's self pair is dropped
    ci = interp.diversity_ci(e, n_boot=50, seed=0)
    assert ci["mean"] == pytest.approx(1.0) and ci["lo"] <= 1.0
    means = np.array([[1.0, 0], [0, 1.0]])
    out = np.array([[1.0, 1.0], [2.0, 0.0]])
    assert interp.novelty_scores(out, means) == pytest.approx([math.sqrt(0.5), 1.0])
    assert interp.novelty_scores(out, means, exclude=[0, 0]) == pytest.approx([math.sqrt(0.5), 0.0])


# ------------------------------------------------------------------------------------------------------ summarize


def make_fake_scores(n_pairs: int = 4):
    """Manifest and fake scores where a code that interpolates linearly gives an ideal path (SIM = cosine of a rotation)."""
    table, sex = fake_table()
    plan = interp.build_plan(table, sex, n_same=n_pairs // 2, n_cross=n_pairs // 2, n_swap=6, n_prior=5)
    med = {s: 120.0 for s in plan["speakers"]}
    items = interp.build_items(plan, med, ("interp", "swap", "prior"), variants=("scaled",))
    rows = [{"key": i.key, "kind": i.kind, "content_id": i.content_id, **i.meta} for i in items]
    man = pd.DataFrame(rows)
    man["ref_text"] = "the cat sat"
    speakers = plan["speakers"]
    rng = np.random.default_rng(0)
    dim = 8
    means = {s: rng.standard_normal(dim) for s in speakers}
    ref_means = {m: means for m in interp.SV_MODELS}
    emb = np.zeros((len(man), dim))
    for r, it in enumerate(items):
        if it.kind == "interp":
            a = it.spec["alpha"]
            emb[r] = (1 - a) * means[it.spec["a"]] + a * means[it.spec["b"]]
        elif it.kind == "swap":
            emb[r] = means[it.spec["donor"]]
        else:
            emb[r] = rng.standard_normal(dim)
    utmos = np.full(len(man), 3.5)
    text = ["the cat sat" if r % 2 == 0 else "the cat sat down" for r in range(len(man))]
    wer = lambda ref, hyp: (abs(len(ref.split()) - len(hyp.split())), len(ref.split()))  # noqa: E731
    real = rng.standard_normal((10, dim))
    return man, {m: emb for m in interp.SV_MODELS}, utmos, text, ref_means, real, [speakers[i % len(speakers)] for i in range(10)], wer


def test_summarize_on_fake_scores():
    man, emb, utmos, text, ref_means, real, real_spk, wer = make_fake_scores()
    s = interp.summarize(man, emb, utmos, text, ref_means, real, real_spk, wer, n_boot=200)
    res = s["interp"]["scaled"]
    assert res["n_pairs"] == 4 and res["n_incomplete_pairs"] == 0
    met = res["metrics"]["all"]
    assert met["ecapa/spearman"]["mean"] > 0.9 and met["ecapa/range"]["mean"] > 0
    assert met["ecapa/evenness"]["mean"] < 0.6
    assert met["quality_dip"]["mean"] == pytest.approx(0.0)
    assert set(res["metrics"]) == {"all", "same_gender", "cross_gender"}
    assert set(res["wer"]) == {"a0", "a4", "a8"}
    sw = s["swap"]["scaled"]
    # outputs equal the donor's reference mean: SIM to the target (donor) is 1, to the source is lower
    assert sw["metrics"]["all"]["ecapa/sim_target"]["mean"] == pytest.approx(1.0)
    assert sw["metrics"]["all"]["ecapa/sim_diff"]["mean"] > 0
    assert s["prior"]["n"] == 5 and 0 <= s["prior"]["diversity_generated"]["mean"] <= 2
    assert "novelty_real_other_speakers" in s["prior"]
    flat = interp.flatten_summary(s)
    assert "interp/scaled/ecapa/score" in flat and "swap/scaled/same_gender/wer" in flat and "prior/utmos" in flat
    md = interp.render_markdown({**s, "meta": {"run": "r", "g_step": 1, "seed": 0}})
    assert "interp/scaled" not in md and "ecapa/spearman" in md
    json_safe = interp.sanitize(s)
    import json

    json.dumps(json_safe, allow_nan=False)


def test_summarize_skips_incomplete_pairs_and_compares():
    man, emb, utmos, text, ref_means, real, real_spk, wer = make_fake_scores()
    drop = man.index[(man["kind"] == "interp") & (man["alpha_idx"] == 3) & (man["pair_id"] == "p00")]
    keep = man.index.difference(drop)
    man2 = man.loc[keep].reset_index(drop=True)
    s = interp.summarize(
        man2, {m: e[keep] for m, e in emb.items()}, utmos[keep], [text[i] for i in keep], ref_means, None, None, wer, n_boot=100
    )
    assert s["interp"]["scaled"]["n_pairs"] == 3 and s["interp"]["scaled"]["n_incomplete_pairs"] == 1
    table = interp.render_compare([s, s], ["a", "b"])
    assert table.splitlines()[0] == "| metric | a | b |" and "interp/scaled/ecapa/score" in table


def test_nan_embeddings_do_not_break_pairs():
    man, emb, utmos, text, ref_means, real, real_spk, wer = make_fake_scores()
    row = man.index[(man["kind"] == "interp") & (man["pair_id"] == "p00") & (man["alpha_idx"] == 2)][0]
    for m in emb:
        emb[m][row] = np.nan
    s = interp.summarize(man, emb, utmos, text, ref_means, real, real_spk, wer, n_boot=100)
    assert s["interp"]["scaled"]["metrics"]["all"]["ecapa/spearman"]["n"] == 3


# ------------------------------------------------------------------------------------------------------ speaker codes


class FakeStore:
    def __init__(self, vectors: dict[str, np.ndarray]):
        self.vectors = vectors
        self.names = list(vectors)

    def index_of(self, utt_id: str) -> int:
        return self.names.index(utt_id)

    def speaker_vector(self, utt: int) -> np.ndarray:
        return self.vectors[self.names[utt]]


def test_speaker_code_ffn_is_mean_of_references():
    torch.manual_seed(0)
    ffn = SpeakerFFN().eval()
    vecs = {f"u{i}": np.random.default_rng(i).standard_normal(1024).astype(np.float32) for i in range(3)}
    z = interp.speaker_code(SimpleNamespace(speaker=ffn), FakeStore(vecs), list(vecs), "cpu")
    with torch.no_grad():
        expect = ffn(torch.from_numpy(np.stack(list(vecs.values())))).mean(0).numpy()
    assert np.allclose(z, expect, atol=1e-5)


def test_speaker_code_vae_is_product_of_gaussians():
    torch.manual_seed(0)
    enc = LayerPoolSpeakerEncoder(n_layers=3, hidden_dim=16, vae=True).eval()
    vecs = {f"u{i}": np.random.default_rng(i).standard_normal((2, 3, 1024)).astype(np.float32) for i in range(4)}
    z = interp.speaker_code(SimpleNamespace(speaker=enc), FakeStore(vecs), list(vecs), "cpu")
    with torch.no_grad():
        out = enc.per_reference(torch.from_numpy(np.stack(list(vecs.values()))))
        mu_k, logvar_k = out[:, :64], out[:, 64:].clamp(-12, 8)
        prec = 1 + torch.exp(-logvar_k).sum(0)
        expect = (mu_k * torch.exp(-logvar_k)).sum(0) / prec
    assert np.allclose(z, expect.numpy(), atol=1e-5)
    one = interp.speaker_code(SimpleNamespace(speaker=enc), FakeStore(vecs), ["u0"], "cpu")
    assert not np.allclose(one, z)  # more references change the posterior mean


def test_parser_defaults():
    args = interp.build_parser().parse_args(["synth", "--run", "r"])
    assert (args.parts, args.n_swap, args.n_prior, args.f0, args.n_same, args.n_cross) == ("interp,swap,prior", 200, 200, "both", 20, 20)
    assert interp.build_parser().parse_args(["compare", "--outs", "a", "b", "--labels", "x", "y"]).labels == ["x", "y"]
