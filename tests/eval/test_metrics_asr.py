"""Text normalization, edit counts and the batching logic of the Whisper wrapper (no model is loaded here)."""

from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

import sparc.conf
from sparc.vocoders.eval.metrics import asr

CONF = Path(sparc.conf.__file__).parent


@pytest.fixture(scope="module")
def asr_cfg() -> DictConfig:
    return OmegaConf.load(CONF / "eval_metrics" / "default.yaml").asr


@pytest.fixture(scope="module")
def normalizer(asr_cfg):
    try:
        norm = asr.load_normalizer(asr_cfg)
    except Exception as err:  # normalizer.json of the pinned revision is not in the hub cache and cannot be fetched
        pytest.skip(f"normalizer.json unavailable: {err}")
    asr.set_default_normalizer(asr_cfg)
    return norm


def test_config_pins_whisper_large_v3(asr_cfg):
    assert asr_cfg.model_id == "openai/whisper-large-v3"
    assert len(asr_cfg.revision) == 40 and asr_cfg.language == "en" and asr_cfg.task == "transcribe"
    assert asr_cfg.num_beams == 1 and asr_cfg.short_form_max_s == 30.0


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Mr. Smith arrived.", "mister smith arrived"),
        ("He paid twenty-one dollars.", "he paid $21"),
        ("The colour of the neighbour's harbour", "the color of the neighbor is harbor"),  # British spelling map
        ("  Hello,   WORLD!  ", "hello world"),
        ("Dr. Jones", "doctor jones"),
        ("", ""),
        ("[YOU THOUGHT I HAD FORGOTTEN]", "you thought i had forgotten"),  # spoken words in brackets are kept
        ("He played (like this) the cornet.", "he played like this the cornet"),
    ],
)
def test_normalize_text_examples(normalizer, text, expected):
    assert asr.normalize_text(text) == expected


def test_normalization_is_idempotent_and_makes_forms_equal(normalizer):
    ref = "Mr. Brown's colour was twenty-one."
    hyp = "mister brown's color was 21"
    assert asr.normalize_text(ref) == asr.normalize_text(hyp)
    once = asr.normalize_text(ref)
    assert asr.normalize_text(once) == once


def levenshtein(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


@pytest.mark.parametrize(
    "ref, hyp, expected",
    [
        ("the cat sat", "the cat sat", (0, 3, 0, 11)),
        ("the cat sat", "the bat sat on", (2, 3, 4, 11)),  # one substitution and one insertion; "c"->"b" plus " on"
        ("a b c d", "a x c", (2, 4, 3, 7)),  # b->x, d deleted; chars: b->x, " d" deleted
        ("a b", "", (2, 2, 3, 3)),
        ("", "a b", (2, 0, 3, 0)),
        ("", "", (0, 0, 0, 0)),
    ],
)
def test_edit_counts_on_toy_strings(ref, hyp, expected):
    counts = asr.edit_counts(ref, hyp)
    assert list(counts) == ["word_errors", "word_ref_len", "char_errors", "char_ref_len"]
    assert tuple(counts.values()) == expected


def test_edit_counts_agree_with_a_plain_levenshtein_dp_and_ignore_extra_spaces():
    rng = np.random.default_rng(0)
    vocabulary = ["the", "a", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and"]
    for _ in range(50):
        ref = " ".join(rng.choice(vocabulary, size=rng.integers(1, 12)))
        hyp = " ".join(rng.choice(vocabulary, size=rng.integers(0, 12)))
        counts = asr.edit_counts(ref, hyp)
        assert counts["word_errors"] == levenshtein(ref.split(), hyp.split())
        assert counts["char_errors"] == levenshtein(list(ref), list(hyp))
        assert counts["word_ref_len"] == len(ref.split()) and counts["char_ref_len"] == len(ref)
    assert asr.edit_counts("  a   b ", "a b") == asr.edit_counts("a b", "a b")


def test_corpus_wer_is_the_ratio_of_sums_not_the_mean_of_ratios():
    pairs = [("one two three four five six seven eight nine ten", "one two three four five six seven eight nine ten"), ("a", "b")]
    counts = [asr.edit_counts(r, h) for r, h in pairs]
    wer = sum(c["word_errors"] for c in counts) / sum(c["word_ref_len"] for c in counts)
    assert wer == pytest.approx(1 / 11)
    assert wer != pytest.approx(np.mean([0.0, 1.0]))


def test_read_reference_reads_the_normalized_text_next_to_the_wav(tmp_path):
    (tmp_path / "1_2_000003_000004.wav").write_bytes(b"")
    (tmp_path / "1_2_000003_000004.normalized.txt").write_text("Hello there, Mr. Smith.\n", encoding="utf-8")
    (tmp_path / "1_2_000003_000004.original.txt").write_text("other", encoding="utf-8")
    assert asr.read_reference(tmp_path / "1_2_000003_000004.wav") == "Hello there, Mr. Smith."
    assert asr.read_reference(str(tmp_path / "1_2_000003_000004.wav")) == "Hello there, Mr. Smith."


class FakeWhisper(asr.WhisperASR):
    """Whisper wrapper with the model replaced by a recorder, to test batching, routing and error handling."""

    def __init__(self, batch_size=3, limit=100, fail_on=()):
        self.cfg = OmegaConf.create({"batch_size": batch_size})
        self.short_limit = limit
        self.fail_on = set(fail_on)
        self.calls: list[tuple[str, list[int]]] = []

    def _decode_short(self, wavs):
        self.calls.append(("short", [len(w) for w in wavs]))
        if self.fail_on & {len(w) for w in wavs}:
            raise RuntimeError("boom")
        return [f"len{len(w)}" for w in wavs]

    def _decode_long(self, wav):
        self.calls.append(("long", [len(wav)]))
        return f"long{len(wav)}"


def test_transcribe_sorts_by_length_batches_and_restores_order():
    lengths = [50, 10, 300, 30, 70, 20, 90]
    fake = FakeWhisper(batch_size=3)
    out = fake.transcribe([np.zeros(n, dtype=np.float32) for n in lengths])
    assert out == [f"len{n}" if n <= 100 else f"long{n}" for n in lengths]
    assert fake.calls == [("short", [10, 20, 30]), ("short", [50, 70, 90]), ("long", [300])]


def test_transcribe_reports_errors_per_utterance_when_asked():
    waves = [np.zeros(n, dtype=np.float32) for n in (10, 20, 30, 40)]
    fake = FakeWhisper(batch_size=4, fail_on={20})
    with pytest.raises(RuntimeError, match="boom"):
        fake.transcribe(waves)
    fake = FakeWhisper(batch_size=4, fail_on={20})
    errors: list[str] = []
    out = fake.transcribe(waves, errors)
    assert out == ["len10", "", "len30", "len40"]
    assert errors[0] == "" and errors[1].startswith("RuntimeError") and errors[2:] == ["", ""]
    assert fake.transcribe([]) == []
