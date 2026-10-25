"""Phase 5 VCTK training set: trimming, duration filter, evaluation-speaker exclusion, resumable 24 kHz preparation."""

import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from sparc.vocoders.constants import SAMPLE_RATE
from sparc.vocoders.features import vctk_train as vt

RATE = 48000


def tone(seconds: float, lead: float = 0.0, tail: float = 0.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    sig = 0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * rng.standard_normal(len(t)))
    return np.concatenate([np.zeros(int(lead * RATE)), sig, np.zeros(int(tail * RATE))]).astype(np.float32)


def make_corpus(tmp_path):
    root = tmp_path / "VCTK-Corpus"
    (root / "speaker-info.txt").parent.mkdir(parents=True)
    (root / "speaker-info.txt").write_text(
        "ID  AGE  GENDER  ACCENTS  REGION\n225  23  F  English  Southern England\n226  22  M  Scottish  Surrey\n"
        "227  22  F  American  NY\n"
    )
    spec = {
        "p225": [("p225_001", tone(2.0)), ("p225_002", tone(1.5))],  # eval speaker
        "p226": [("p226_001", tone(2.0, lead=1.0, tail=1.5)), ("p226_002", tone(0.3, lead=0.5)), ("p226_003", tone(16.0))],
        "p227": [("p227_001", tone(3.0, lead=0.5)), ("p227_002", tone(1.0, 0.0, 0.0))],
    }
    for spk, utts in spec.items():
        (root / "wav48" / spk).mkdir(parents=True)
        (root / "txt" / spk).mkdir(parents=True)
        for utt, wav in utts:
            sf.write(root / "wav48" / spk / f"{utt}.wav", wav, RATE)
            if spk != "p227":  # a speaker without transcripts is kept
                (root / "txt" / spk / f"{utt}.txt").write_text(f"text of {utt}\n")
    sf.write(root / "wav48" / "p226" / "p226_001_mic2.wav", tone(2.0), RATE)  # skipped like in clone_pools
    eval_table = tmp_path / "table.parquet"
    pd.DataFrame({"id": ["p225_001", "p225_002"], "speaker": ["p225", "p225"]}).to_parquet(eval_table)
    return root, eval_table


def test_trim_bounds_cut_silence_keep_padding_and_clip_to_the_file():
    wav = tone(2.0, lead=1.0, tail=1.5)
    lo, hi = vt.trim_bounds(wav, RATE)
    assert abs(lo - (1.0 - vt.PAD_S) * RATE) < 0.05 * RATE
    assert abs(hi - (3.0 + vt.PAD_S) * RATE) < 0.05 * RATE
    assert vt.trim_bounds(tone(1.0), RATE)[0] == 0
    assert vt.trim_bounds(np.zeros(RATE, np.float32), RATE) == (0, RATE)


def test_drop_reason_limits_are_inclusive():
    assert vt.drop_reason(0.99) == "too_short" and vt.drop_reason(1.0) is None
    assert vt.drop_reason(15.0) is None and vt.drop_reason(15.01) == "too_long"


def test_list_excludes_eval_speakers_and_reports_drops(tmp_path):
    root, eval_table = make_corpus(tmp_path)
    table, report = vt.build_list(root, eval_table, workers=1)
    assert set(table["speaker"]) == {"p227", "p226"} and not table["id"].str.contains("mic2").any()
    assert sorted(table["id"]) == ["p226_001", "p227_001", "p227_002"]
    assert report["eval_speakers_excluded"] == ["p225"] and report["train_speakers"] == 2
    assert report["per_speaker"]["p226"]["dropped"] == {"unreadable": 0, "too_short": 1, "too_long": 1}
    assert report["drop_reasons"]["too_short"] == 1 and report["drop_reasons"]["too_long"] == 1
    assert report["speakers_without_text"] == ["p227"]
    row = table.set_index("id").loc["p226_001"]
    assert row["native_sr"] == RATE and 0 < row["trim_start"] < row["trim_end"] <= row["n_native"]
    assert abs(row["dur_trimmed"] - (row["trim_end"] - row["trim_start"]) / RATE) < 1e-9
    assert row["gender"] == "M" and row["accent"] == "Scottish" and row["text"] == "text of p226_001"


def test_prepare_writes_trimmed_24k_wavs_and_is_resumable(tmp_path):
    root, eval_table = make_corpus(tmp_path)
    out = tmp_path / "out"
    table = vt.run_list(out, root, eval_table)
    for shard in range(2):
        assert vt.run_prepare(out, shard, 2) == 0
    for r in table.itertuples():
        path = vt.wav24_path(out, r.speaker, r.id)
        info = sf.info(str(path))
        assert info.samplerate == SAMPLE_RATE and info.channels == 1 and abs(info.frames - r.n24) <= 1
    path = vt.wav24_path(out, "p227", "p227_001")
    mtime = path.stat().st_mtime_ns
    assert vt.run_prepare(out, 0, 1) == 0 and path.stat().st_mtime_ns == mtime
    with pytest.raises(ValueError):
        vt.run_prepare(out, 2, 2)


def test_kept_table_removes_dropped_ids_and_hooks_are_disabled(tmp_path):
    root, eval_table = make_corpus(tmp_path)
    out = tmp_path / "out"
    vt.run_list(out, root, eval_table)
    with pytest.raises(FileNotFoundError):
        vt.kept_table(out)
    paths = vt.out_paths(out)
    paths["leak"].mkdir()
    paths["dropped"].write_text("p227_001\n")
    assert sorted(vt.kept_table(out)["id"]) == ["p226_001", "p227_002"]
    with pytest.raises(NotImplementedError):
        vt.run_extract(out)
    assert json.loads(paths["report"].read_text())["kept"] == 3
