"""Clean validation scoring of the OOD screen (Amendment 2 of docs/spkemb_ood_screen.md): probe-fit speakers are disjoint
from the validation and test speakers, VCTK scored speakers are the cloning speakers, centering groups never mix splits."""

import pathlib

import numpy as np
import pandas as pd
import pytest

from sparc.vocoders.eval import splits as S
from sparc.vocoders.spkscreen.ood_embed import center_groups

SPLITS = {
    "cloning": {"val": {"vctk": ["p1"]}, "test": {"vctk": ["p3"]}},
    "screen": {"val": {"heldout": ["libri_dev:a"], "globe_spk": ["globe_spk:g1"]},
               "test": {"heldout": ["libri_test:c"], "globe_spk": []}},
}


def test_screen_scored_uses_cloning_vctk_and_probe_fit_is_disjoint():
    spk = np.array(["libri_dev:a", "libri_dev:z", "vctk:p1", "vctk:p3", "vctk:p9", "libri_test:c"])
    m = S.screen_scored(spk, SPLITS)
    assert m["val"].tolist() == [True, False, True, False, False, False]
    assert m["test"].tolist() == [False, False, False, True, False, True]
    fit = S.probe_fit_mask(spk, SPLITS)
    assert fit.tolist() == [False, True, False, False, True, False]
    assert not (fit & m["val"]).any() and not (fit & m["test"]).any() and not (m["val"] & m["test"]).any()


def test_checked_in_probe_fit_speakers_are_disjoint_from_validation_and_test():
    man = pathlib.Path("/data/user_data/xoy/spkemb/encoder_screen/heldout/manifest.parquet")
    if not man.exists():
        pytest.skip("encoder-screen manifest not available")
    sp = S.load()
    names = pd.read_parquet(man)["speaker"].astype(str).unique()
    m = S.screen_scored(names, sp)
    fit = set(names[S.probe_fit_mask(names, sp)])
    val, test = set(names[m["val"]]), set(names[m["test"]])
    assert fit and val and test
    assert not (fit & val) and not (fit & test) and not (val & test)
    assert {n for n in val if n.startswith("vctk:")} == {f"vctk:{s}" for s in sp["cloning"]["val"]["vctk"]}
    assert {n for n in test if n.startswith("vctk:")} == {f"vctk:{s}" for s in sp["cloning"]["test"]["vctk"]}


def test_centering_groups_split_by_corpus_and_split():
    spk = np.array(["libri_dev:a", "libri_dev:z", "vctk:p1", "vctk:p3", "vctk:p9", "libri_test:c"])
    g = center_groups(spk, SPLITS, "heldout")
    assert g.tolist() == ["libri/val", "libri/fit", "vctk/val", "vctk/test", "vctk/fit", "libri/test"]
    gg = center_groups(np.array(["globe_spk:g1", "globe_spk:g2"]), SPLITS, "globe_spk")
    assert gg.tolist() == ["globe_spk/val", "globe_spk/fit"]
