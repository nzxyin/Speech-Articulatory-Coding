"""Phase 5 filtered packed subsets: rows, offsets, speaker arrays and datamodule compatibility."""

import numpy as np
import pandas as pd
import pytest

from sparc.vocoders.data.dataset import CropDataset, PackedStore
from sparc.vocoders.features import subset_pack as sp
from test_data import build_split

ROWS = [(f"u{i}", f"s{i % 2}", f"c{i}", 1.0 + 0.5 * i) for i in range(8)]


def test_runs():
    assert list(sp.runs(np.array([0, 1, 2, 5, 7, 8]))) == [(0, 3), (5, 6), (7, 9)]
    assert list(sp.runs(np.array([], dtype=int))) == []


def test_subset_rows_offsets_and_arrays(tmp_path):
    src = build_split(tmp_path, "src", ROWS)
    keep = {"u1", "u2", "u3", "u6", "missing"}
    rep = sp.subset_split(src, keep, tmp_path / "out" / "src")
    assert rep["kept"] == 4 and rep["keep_ids_not_in_source"] == 1 and rep["source_rows"] == 8
    a, b = PackedStore([src]), PackedStore([tmp_path / "out" / "src"])
    assert b.ids.tolist() == ["u1", "u2", "u3", "u6"]
    assert b.offset.tolist() == np.concatenate([[0], np.cumsum(b.T)[:-1]]).tolist()
    for u in b.ids:
        i, j = a.index_of(u), b.index_of(u)
        assert np.array_equal(a.frames(i, 0, int(a.T[i]))[0], b.frames(j, 0, int(b.T[j]))[0])
        assert np.array_equal(a.frames(i, 0, int(a.T[i]))[1], b.frames(j, 0, int(b.T[j]))[1])
        assert a.speakers[a.speaker_codes[i]] == b.speakers[b.speaker_codes[j]]
        assert a.wav_paths[i] == b.wav_paths[j] and a.peak24[i] == b.peak24[j] and a.spk_wsum[i] == b.spk_wsum[j]
    for layer in ("l0", "l6", "enplus64"):
        sa, sb = PackedStore([src], layer), PackedStore([tmp_path / "out" / "src"], layer)
        for u in b.ids:
            assert np.array_equal(sa.speaker_vector(sa.index_of(u)), sb.speaker_vector(sb.index_of(u)))
    ds = CropDataset([tmp_path / "out" / "src"], crop_frames=16, n_refs=(1, 3))
    sample = ds[(0, 0, 0)]
    assert sample["spk_raw"].shape[0] == 3


def test_empty_keep_raises(tmp_path):
    src = build_split(tmp_path, "src", ROWS)
    with pytest.raises(ValueError):
        sp.subset_split(src, {"zzz"}, tmp_path / "out")


def test_kept_ids_reads_keep_column(tmp_path):
    pd.DataFrame({"id": ["a", "b", "c"], "keep": [True, False, True]}).to_parquet(tmp_path / "x.parquet")
    assert sp.kept_ids(tmp_path / "x.parquet") == {"a", "c"}
