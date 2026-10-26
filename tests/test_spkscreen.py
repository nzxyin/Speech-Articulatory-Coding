"""Speaker-encoder screen harness on synthetic frames (no WavLM, no cache)."""

import json

import numpy as np
import pytest
import torch

from sparc.vocoders.spkscreen import metrics as M
from sparc.vocoders.spkscreen import pooling, report, train
from sparc.vocoders.spkscreen.data import FrameSet


def synth(n_spk, n_utt, seed, dim=1024, layers=(6,), target_dim=12):
    """Frames whose layer features carry a per-speaker offset plus per-utterance noise; targets follow the offset."""
    rng = np.random.default_rng(seed)
    proj = rng.normal(size=(8, target_dim))
    hidden = {k: [] for k in layers}
    aux, lens, spk, tg = [], [], [], []
    for s in range(n_spk):
        code = rng.normal(size=8)
        base = np.zeros(dim)
        base[:8] = code * 3
        for u in range(n_utt):
            n = int(rng.integers(60, 150))
            for k in layers:
                hidden[k].append((base + rng.normal(size=(n, dim))).astype(np.float16))
            per = (rng.random(n) > 0.4).astype(np.float32)
            aux.append(np.stack([per, rng.normal(size=n), rng.random(n), 100 + 20 * rng.random(n)], 1).astype(np.float32))
            lens.append(n)
            spk.append(f"s{s}")
            tg.append(code @ proj + 0.3 * rng.normal(size=target_dim))
    fs = FrameSet({k: torch.from_numpy(np.concatenate(v)) for k, v in hidden.items()},
                  torch.from_numpy(np.concatenate(aux)), None, np.array(lens), np.array(spk), np.array(tg),
                  np.array([np.log(100.0 + 5 * t[0]) for t in tg]))
    return fs


def test_batch_matches_slices():
    fs = synth(3, 3, 0)
    b = fs.batch(np.array([4, 1]), np.array([5, 0]), np.array([20, 10]), (6,))
    o = fs.offsets[4]
    np.testing.assert_allclose(b.hidden[0, 0, :20].numpy(), fs.hidden[6][o + 5 : o + 25].float().numpy())
    assert b.mask.sum(1).tolist() == [20, 10]
    assert float(b.hidden[1, 0, 10:].abs().sum()) == 0.0
    np.testing.assert_allclose(b.periodicity[0, :20].numpy(), fs.aux[o + 5 : o + 25, 0].numpy())


def test_baseline_pooling_and_fallback():
    fs = synth(2, 2, 1)
    enc = pooling.build("baseline")
    enc.setup(fs)
    b = fs.full(np.array([0, 1]), enc.layers)
    got = enc.pooled(b)
    o, n = fs.offsets[0], fs.lens[0]
    w = fs.aux[o : o + n, 0].double()
    want = (fs.hidden[6][o : o + n].double() * w[:, None]).sum(0) / w.sum()
    np.testing.assert_allclose(got[0].double().numpy(), want.numpy(), atol=1e-4)
    b.periodicity.zero_()  # no voiced frames: uniform over real frames
    uni = enc.pooled(b)[0].double()
    np.testing.assert_allclose(uni.numpy(), fs.hidden[6][o : o + n].double().mean(0).numpy(), atol=1e-4)
    assert enc(b).shape == (2, 64)
    assert enc.n_params() == 1024 * 1024 + 1024 + 1024 * 64 + 64


def test_registry():
    pooling.build("baseline")
    with pytest.raises(KeyError):
        pooling.build("no_such_variant")
    with pytest.raises(ValueError):
        pooling.register("baseline")(lambda: None)


def test_eer_and_metrics():
    assert M.equal_error_rate(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 0.0
    assert abs(M.equal_error_rate(np.arange(100.0), np.arange(100.0)) - 0.5) < 0.02
    h = synth(12, 8, 2)
    spk = h.speakers
    is_test = np.isin(spk, [f"s{i}" for i in range(8, 12)])
    good = M.evaluate(h.targets[:, :8] + 0.0, h.targets, h.logf0, spk, is_test)  # embedding = informative
    rng = np.random.default_rng(0)
    bad = M.evaluate(rng.normal(size=(len(spk), 8)), h.targets, h.logf0, spk, is_test)
    assert good["m1_rmse"] < bad["m1_rmse"]
    assert good["id_acc"] > 0.9 and good["eer"] < bad["eer"]
    for k in ("m1_rmse", "id_acc", "eer", "m3_leak_pcc", "f0_pcc"):
        assert np.isfinite(good[k]) and good[k + "_se"] >= 0


def test_train_runs_and_resumes(tmp_path):
    tr, he = synth(10, 4, 3), synth(15, 8, 4)
    he.is_test = np.isin(he.speakers, [f"s{i}" for i in range(10, 15)])
    cfg = {"steps": 40, "batch": 8, "warmup": 5, "crop_max": 80}
    r = train.train_variant("baseline", 0, tmp_path, device="cpu", cfg=cfg, train=tr, heldout=he, quiet=True)
    assert np.isfinite(r["m1_rmse"]) and r["encoder_params"] > 1e6 and (tmp_path / "runs/baseline/seed0/embeddings.npz").exists()
    assert r["log"][-1]["aam"] < r["log"][0]["aam"]
    again = train.train_variant("baseline", 0, tmp_path, device="cpu", cfg=cfg, train=tr, heldout=he, quiet=True)
    assert again["m1_rmse"] == r["m1_rmse"]


def test_report_rule(tmp_path):
    def put(name, seed, m1, eer, m3):
        d = tmp_path / "runs" / name / f"seed{seed}"
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({"seed": seed, "m1_rmse": m1, "eer": eer, "m3_leak_pcc": m3, "id_acc": 0.9,
                                                   "f0_pcc": 0.9, "encoder_params": 1}))

    for s, off in enumerate((0.0, 0.001, -0.001)):
        put("baseline", s, 0.85 + off, 0.10, 0.5)
        put("better", s, 0.80 + off + 0.0005 * s, 0.10, 0.52)
        put("leaky", s, 0.80 + off, 0.10, 0.60)
        put("noise", s, 0.85 + off + 0.003 * (-1) ** s, 0.10, 0.5)
    text = report.summarize(tmp_path, "baseline")
    assert "Promising, ranked by M1 gain then EER gain: better" in text
