"""Model-based metrics (UTMOS, Whisper, speaker embeddings) on real test.clean audio, on CPU.

The model tests are marked ``slow``: they need the pinned models in the hub cache (``HF_HUB_CACHE``, ``TORCH_HOME``)
and the packed test.clean split (``SPARC_VOC_CACHE``) and take minutes. The helpers at the top are plain unit tests.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

import sparc.conf
from sparc.vocoders.constants import HOP
from sparc.vocoders.data.dataset import PackedStore
from sparc.vocoders.data.gain import apply_gain, gain_factor
from sparc.vocoders.eval.metrics import asr, speaker
from sparc.vocoders.eval.metrics.signal import resample

CONF = Path(sparc.conf.__file__).parent
EVAL_GAIN_DB = -3.0


@pytest.fixture(scope="module")
def cfg() -> DictConfig:
    return OmegaConf.load(CONF / "eval_metrics" / "default.yaml")


# --- plain unit tests ---------------------------------------------------------------------------------------------


def test_cosine_and_normalization():
    a = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
    b = np.array([[2.0, 0.0], [1.0, -1.0], [0.0, -1.0]])
    np.testing.assert_allclose(speaker.cosine(a, b), [1.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(speaker.cosine(a, np.array([0.0, 3.0])), [0.0, 1 / math.sqrt(2), 1.0], atol=1e-12)
    assert speaker.cosine(np.array([1.0, 2.0]), np.array([2.0, 4.0])) == pytest.approx(1.0)
    unit = speaker.l2_normalize(a)
    np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1.0)


def test_mean_embedding_is_the_mean_of_normalized_rows_and_skips_nan():
    rows = np.array([[3.0, 0.0], [0.0, 5.0], [np.nan, np.nan]])
    np.testing.assert_allclose(speaker.mean_embedding(rows), [0.5, 0.5])
    assert np.isnan(speaker.mean_embedding(np.full((2, 3), np.nan))).all()


def test_unknown_speaker_model_is_rejected(cfg):
    with pytest.raises(ValueError, match="one of"):
        speaker.SpeakerEmbedder("resemblyzer", cfg.speaker, "cpu")


def test_config_pins_every_model(cfg):
    assert len(cfg.asr.revision) == 40
    assert len(cfg.speaker.ecapa.revision) == 40 and len(cfg.speaker.wavlm_sv.revision) == 40
    assert cfg.utmos.hub_repo == "tarepan/SpeechMOS:v1.2.0" and len(cfg.utmos.checkpoint_sha256) == 64
    assert cfg.speaker.ecapa.dim == 192 and cfg.speaker.wavlm_sv.dim == 512


# --- real audio ---------------------------------------------------------------------------------------------------


class Corpus:
    """A few test.clean utterances prepared like ``FullUtteranceDataset`` (first ``480 T`` samples times the eval gain)."""

    def __init__(self, packed_dir: Path):
        self.store = PackedStore([packed_dir])

    def audio16(self, utt: int) -> np.ndarray:
        store = self.store
        audio = store.read_audio(utt, 0, HOP * int(store.T[utt]))
        audio = apply_gain(audio, gain_factor(store.peak24[utt], EVAL_GAIN_DB))
        return resample(audio, 24000, 16000)

    def reference(self, utt: int) -> str:
        return asr.read_reference(self.store.wav_paths[utt])

    def pick(self, low_s: float, high_s: float, speakers: int, per_speaker: int) -> list[list[int]]:
        """Utterance indices by speaker (sorted by id), durations within ``[low_s, high_s]``."""
        store = self.store
        groups: dict[str, list[int]] = {}
        for utt in np.argsort(store.ids):
            if low_s <= store.T[utt] / 50.0 <= high_s:
                groups.setdefault(str(store.speakers[store.speaker_codes[utt]]), []).append(int(utt))
        chosen = [g[:per_speaker] for _, g in sorted(groups.items()) if len(g) >= per_speaker]
        return chosen[:speakers]


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    root = os.environ.get("SPARC_VOC_CACHE")
    packed = Path(root) / "packed" / "test.clean" if root else None
    if packed is None or not (packed / "index.parquet").is_file():
        pytest.skip("packed test.clean split not available (SPARC_VOC_CACHE)")
    return Corpus(packed)


@pytest.fixture(scope="module")
def three_speakers(corpus) -> list[list[int]]:
    groups = corpus.pick(3.0, 6.0, speakers=3, per_speaker=2)
    assert len(groups) == 3
    return groups


@pytest.mark.slow
def test_utmos_scores_speech_high_and_noise_low(cfg, corpus, three_speakers):
    from sparc.vocoders.eval.metrics.utmos import UTMOS

    model = UTMOS(cfg.utmos, "cpu")
    assert model.provenance()["checkpoint_sha256"] == cfg.utmos.checkpoint_sha256
    utts = [g[0] for g in three_speakers]
    speech = [corpus.audio16(u) for u in utts]
    noise = [np.random.default_rng(0).standard_normal(len(w)).astype(np.float32) * 0.05 for w in speech]
    s_speech, s_noise = model.score(speech), model.score(noise)
    print("UTMOS gt", s_speech, "white noise", s_noise)
    assert all(3.3 < s < 5.0 for s in s_speech)
    assert all(1.0 <= s < 2.0 for s in s_noise)
    assert model.score(speech[:1]) == pytest.approx(s_speech[:1], abs=1e-4)  # deterministic, independent of the list
    # zero padding changes the score, which is why utterances are scored one at a time
    padded = np.concatenate([speech[0], np.zeros(16000, dtype=np.float32)])
    print("UTMOS padded with 1 s of zeros:", model.score([padded])[0], "unpadded:", s_speech[0])
    errors: list[str] = []
    out = model.score([speech[0], np.zeros(100, dtype=np.float32)], errors)
    assert math.isfinite(out[0]) and math.isnan(out[1]) and errors[0] == "" and errors[1].startswith("ValueError")


@pytest.mark.slow
def test_whisper_transcribes_gt_close_to_the_reference_text(cfg, corpus, three_speakers):
    model = asr.WhisperASR(cfg.asr, "cpu")
    assert model.dtype.is_floating_point and str(model.dtype) == "torch.float32"
    utts = [three_speakers[0][0], three_speakers[1][0], three_speakers[2][0]]
    texts = model.transcribe([corpus.audio16(u) for u in utts])
    totals = {"word_errors": 0, "word_ref_len": 0, "char_errors": 0, "char_ref_len": 0}
    for u, text in zip(utts, texts):
        ref, hyp = asr.normalize_text(corpus.reference(u)), asr.normalize_text(text)
        counts = asr.edit_counts(ref, hyp)
        print(f"{corpus.store.ids[u]}\n  REF {ref}\n  HYP {hyp}\n  {counts}")
        for key in totals:
            totals[key] += counts[key]
    wer, cer = totals["word_errors"] / totals["word_ref_len"], totals["char_errors"] / totals["char_ref_len"]
    print(f"corpus WER {wer:.4f}  CER {cer:.4f}")
    assert wer < 0.10 and cer < 0.06
    # batching does not change the transcript: two utterances in one batch against one at a time
    pair = [corpus.audio16(utts[0]), corpus.audio16(utts[1])]
    assert model.transcribe(pair) == [model.transcribe([w])[0] for w in pair]


@pytest.mark.slow
def test_whisper_long_form_for_inputs_over_30_seconds(cfg, corpus):
    store = corpus.store
    long_utts = [int(u) for u in np.argsort(store.T) if store.T[u] / 50.0 > 31.0]
    assert long_utts, "no test.clean utterance longer than 31 s"
    utt = long_utts[0]
    wav = corpus.audio16(utt)
    assert len(wav) > 30 * 16000
    model = asr.WhisperASR(cfg.asr, "cpu")
    text = model.transcribe([wav])[0]
    ref, hyp = asr.normalize_text(corpus.reference(utt)), asr.normalize_text(text)
    counts = asr.edit_counts(ref, hyp)
    print(f"long-form {store.ids[utt]} ({len(wav) / 16000:.1f} s)\n  REF {ref}\n  HYP {hyp}\n  {counts}")
    assert counts["word_errors"] / counts["word_ref_len"] < 0.10


@pytest.mark.slow
@pytest.mark.parametrize("name", ["ecapa", "wavlm_sv"])
def test_speaker_embeddings_separate_speakers(cfg, corpus, three_speakers, name):
    model = speaker.SpeakerEmbedder(name, cfg.speaker, "cpu")
    assert model.dim == cfg.speaker[name].dim
    utts = [u for group in three_speakers for u in group]  # speaker 0 twice, speaker 1 twice, speaker 2 twice
    emb = model.embed([corpus.audio16(u) for u in utts])
    assert emb.shape == (6, model.dim) and emb.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    sims = speaker.cosine(emb[:, None, :], emb[None, :, :])
    same = [sims[2 * k, 2 * k + 1] for k in range(3)]
    different = [sims[i, j] for i in range(6) for j in range(6) if i // 2 != j // 2]
    print(f"{name}: same-speaker cosine {np.round(same, 3)}, different-speaker mean {np.mean(different):.3f} "
          f"(max {np.max(different):.3f})")
    # the partner is each utterance's nearest neighbour (WavLM-SV has a high floor: unrelated speakers reach 0.9, so a
    # global threshold between same and different pairs does not exist for it, but the ranking per utterance holds)
    off_diagonal = np.where(np.eye(6, dtype=bool), -np.inf, sims)
    assert (off_diagonal.argmax(axis=1) // 2 == np.arange(6) // 2).all()
    assert np.mean(same) > np.mean(different) + 0.1
    # the same audio gives the same embedding
    again = model.embed([corpus.audio16(utts[0])])
    assert speaker.cosine(emb[0], again[0]) == pytest.approx(1.0, abs=1e-5)
    errors: list[str] = []
    rows = model.embed([corpus.audio16(utts[0]), np.zeros(1000, dtype=np.float32)], errors)
    assert np.isfinite(rows[0]).all() and np.isnan(rows[1]).all() and errors[0] == "" and errors[1].startswith("ValueError")
