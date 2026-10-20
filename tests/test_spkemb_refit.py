import numpy as np
import torch

from sparc.spkemb import ffn_refit as fr


def test_pick_ref_never_self():
    rng = np.random.default_rng(0)
    for n in (2, 3, 10):
        for j in range(n):
            assert fr.pick_ref(n, j) != j
            assert all(fr.pick_ref(n, j, rng) != j for _ in range(50))
    assert {fr.pick_ref(4, 2, rng) for _ in range(200)} == {0, 1, 3}


def test_vc_target_and_heldout():
    sp = ["a", "b", "c"]
    assert [fr.vc_target(sp, i) for i in range(3)] == [1, 2, 0]
    names = [f"vctk:p{i}" for i in range(109)]
    h = fr.heldout_speakers(names)
    assert len(h) == 10 and h == fr.heldout_speakers(list(reversed(names)))


def test_wer_corpus():
    rows = [{"wer": 0.5, "n_words": 10}, {"wer": 0.0, "n_words": 30}, {"wer": None, "n_words": 5}]
    assert abs(fr.wer_corpus(rows) - 12.5) < 1e-9


def test_mrstft():
    x = torch.randn(2, 8000)
    assert fr.mrstft_loss(x, x).item() < 1e-5
    assert fr.mrstft_loss(x, torch.randn(2, 8000)).item() > 0.5
