"""Corpus definitions: file discovery, utterance metadata, sentence segmentation, per-speaker frame facts.

USC-TIMIT EMA (Data/<spk>/{mat,wav,trans}; speakers M1, F1, M3, F5)
    Each .mat holds five consecutive TIMIT sentences (e.g. ..._001_005). The .wav is identical to the
    embedded AUDIO. .trans rows are "start,end,phone,word,sentence text,sentence index" on the same clock.
    M3's transcripts are named usctimit_mri_m3_* and belong to the MRI session: a lip-aperture vs /p b m/
    check gives no correlation for them (vs a consistent peak for M1/F1/F5), so M3 is segmented from the
    audio (pauses) instead. EMA ends 70-110 ms before the audio in most F1/F5 and some M3 files; the same
    check shows the streams are start-aligned (the EMA just stops early).
    F5 sentences 001-065 are offset by ~12 mm (all sensors, mostly lateral, 3-5 mm anterior) from 066-460,
    a content-independent frame change between two recording blocks, so F5 gets two normalization groups.

USC EMA_5EMO (jn, jr, kf; 5 emotions x 8 sentences, neutral repetitions, 2 passages, normal/fast rate)
    pssg_short/ holds the passages cut into phrases (passage 1: 10, passage 2: 5). They are exact excerpts
    (audio and EMA) of the full passage files, so the phrases are used and the full passages are skipped.
    The .wav files are peak-normalized copies of the embedded AUDIO. dummy* files are skipped.
    jn's coordinates are not aligned to the occlusal plane (README: "The articulatory data of F1 was not
    aligned"; jn is the only speaker whose frame differs): its lip line (LL->UL) and tongue line (TD->TT)
    are rotated by 48.7 and 51.6 degrees relative to the median of the six other speakers, so jn's x/y are
    rotated by JN_ROTATION_DEG. Sessions are emotion blocks and carry 2-6 mm common-mode offsets, but
    session and emotion are confounded (each emotion was one block), so they are not corrected; per-file
    offset estimates go into the manifest.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DB_ROOT = Path("/data/group_data/UTD-NAS/Databases")
USC_ROOT = DB_ROOT / "USC-TIMIT" / "EMA" / "Data"
USC_SENTENCES = DB_ROOT / "USC-TIMIT" / "list_of_sentences.txt"
EMO_ROOT = DB_ROOT / "EMA_5EMO_NSF"

USC_SPEAKERS = ("M1", "F1", "M3", "F5")
EMO_SPEAKERS = ("jn", "jr", "kf")
EMOTIONS = {"neu": "neutral", "ang": "anger", "sad": "sadness", "hap": "happiness", "fea": "fear"}
JN_ROTATION_DEG = 50.2  # mean of the lip-line (48.7) and tongue-line (51.6) estimates

EMO_SENTENCES = {
    1: "9 1 5 2 6 9 5 1 6 2",
    2: "MaMaMa MaMaMa MaMaMaMa",
    3: "John bought five black cats at the store.",
    4: "The Leopard, skunk and peacock are wild animals.",
    5: "Charlie, did you think to measure the tree?",
    6: "The queen said the KNIGHT is a MONSTER.",
    7: "Pam said bat that FAT cat at that mat.",
    8: "Hickory dickory dock, the mouse ran up the clock. Hickory dickory dock.",
}


@dataclass
class Segment:
    """One output utterance: a time span of a source .mat file."""
    corpus: str
    speaker: str  # corpus-prefixed, e.g. usc_M1, 5emo_jn
    utt_id: str
    mat: Path
    start: float  # seconds into the file (audio clock)
    end: float
    text: str
    norm_group: str
    meta: dict = field(default_factory=dict)


def rotation_deg(speaker):
    return JN_ROTATION_DEG if speaker == "5emo_jn" else 0.0


# --- USC-TIMIT -------------------------------------------------------------------------------

def usc_sentence_texts():
    out = {}
    for line in USC_SENTENCES.read_text().splitlines():
        m = re.match(r"\s*(\d+)\s*:\s*(.*)", line)
        if m:
            out[int(m.group(1))] = m.group(2).strip()
    return out


def usc_norm_group(spk, sent_id):
    if spk == "F5":
        return "usc_F5_s1" if sent_id <= 65 else "usc_F5_s2"
    return f"usc_{spk}"


def read_trans(path):
    """[(start, end, phone, word, sentence_index or None)] from a USC-TIMIT .trans file."""
    rows = []
    for line in path.read_text().splitlines():
        p = line.split(",")
        if len(p) < 3:
            continue
        idx = p[5].strip() if len(p) > 5 else ""
        rows.append((float(p[0]), float(p[1]), p[2].strip(), p[3].strip() if len(p) > 3 else "",
                     int(idx) if idx.isdigit() else None))
    return rows


def spans_from_trans(rows, total_dur, pad=0.2):
    """Sentence spans from phone rows: first-to-last phone of each sentence index, padded into the
    neighbouring silences by up to `pad` s but never past the middle of a shared silence."""
    sents = {}
    for s, e, ph, w, idx in rows:
        if idx is not None:
            a, b = sents.get(idx, (s, e))
            sents[idx] = (min(a, s), max(b, e))
    order = sorted(sents, key=lambda i: sents[i][0])
    spans = {}
    for k, i in enumerate(order):
        s, e = sents[i]
        prev_end = sents[order[k - 1]][1] if k > 0 else 0.0
        next_start = sents[order[k + 1]][0] if k + 1 < len(order) else total_dur
        lo = max(s - pad, (prev_end + s) / 2 if k > 0 else 0.0)
        hi = min(e + pad, (e + next_start) / 2 if k + 1 < len(order) else total_dur)
        spans[i] = (lo, hi)
    return spans


def usc_expected_durations(exclude=()):
    """Typical spoken duration of each TIMIT sentence (first to last phone), from the transcribed speakers,
    each speaker's durations divided by that speaker's mean rate factor, then the median across speakers."""
    per_spk = {}
    for spk in USC_SPEAKERS:
        if spk in exclude or spk == "M3":  # M3's transcripts belong to its MRI session
            continue
        d = {}
        for mat, first, last, trans in usc_files(spk):
            if trans is None:
                continue
            sents = {}
            for s, e, ph, w, idx in read_trans(trans):
                if idx is not None:
                    a, b = sents.get(idx, (s, e))
                    sents[idx] = (min(a, s), max(b, e))
            d.update({i: b - a for i, (a, b) in sents.items()})
        per_spk[spk] = d
    common = set.intersection(*[set(d) for d in per_spk.values()])
    ref = {i: np.median([per_spk[s][i] for s in per_spk]) for i in common}
    out = {}
    for s, d in per_spk.items():
        rate = np.median([d[i] / ref[i] for i in common])
        for i, v in d.items():
            out.setdefault(i, []).append(v / rate)
    return {i: float(np.median(v)) for i, v in out.items()}


def _speech_mask(audio, sr):
    hop = int(sr * 0.01)
    frames = len(audio) // hop
    e = 10 * np.log10((audio[: frames * hop].reshape(frames, hop) ** 2).mean(1) + 1e-12)
    lo, hi = np.percentile(e, 10), np.percentile(e, 95)
    speech = e > lo + 0.35 * (hi - lo)
    k = 5  # close small gaps inside words, drop isolated blips
    return np.convolve(speech.astype(float), np.ones(k) / k, mode="same") > 0.5


def _pad_spans(bounds, total, pad):
    spans = []
    for k, (s, e) in enumerate(bounds):
        prev_end = bounds[k - 1][1] if k > 0 else 0.0
        next_start = bounds[k + 1][0] if k + 1 < len(bounds) else total
        spans.append((max(s - pad, (prev_end + s) / 2 if k > 0 else 0.0),
                      min(e + pad, (e + next_start) / 2 if k + 1 < len(bounds) else total)))
    return spans


def spans_from_pauses_expected(audio, sr, expected, min_pause=0.08, pad=0.2, sigma=0.25):
    """Split a recording of len(expected) sentences at internal pauses chosen by dynamic programming.

    Score of a choice of n-1 pauses = sum of log pause lengths - sum over sentences of
    (log(duration / (rate * expected)))^2 / (2 sigma^2), with the speaker's rate factor estimated from the
    file's total speech span. Returns [(start, end)] or None if there are too few pauses."""
    n = len(expected)
    speech = _speech_mask(audio, sr)
    idx = np.flatnonzero(speech)
    if len(idx) == 0:
        return None
    first, last = idx[0], idx[-1] + 1
    pauses = []  # (start_frame, end_frame) of internal silent runs
    i = first
    while i < last:
        if not speech[i]:
            j = i
            while j < last and not speech[j]:
                j += 1
            if (j - i) * 0.01 >= min_pause:
                pauses.append((i, j))
            i = j
        else:
            i += 1
    if len(pauses) < n - 1:
        return None
    exp = np.asarray(expected, float)
    rate = (last - first) * 0.01 / exp.sum()  # pauses between sentences inflate this slightly; fine
    starts = [first] + [b for a, b in pauses]  # possible sentence starts (frames)
    ends = [a for a, b in pauses] + [last]  # possible sentence ends
    P = len(pauses)
    NEG = -1e18
    # dp[k][p]: best score with sentence k ending at end option p (p = pause index, or P = file end)
    dp = np.full((n, P + 1), NEG)
    back = np.zeros((n, P + 1), int)
    dur_cost = lambda s, e, k: (np.log(max((e - s) * 0.01, 1e-3) / (rate * exp[k]))) ** 2 / (2 * sigma**2)
    for p in range(P + 1):  # first sentence: starts at `first`
        if p == P and n > 1:
            continue
        dp[0][p] = -dur_cost(first, ends[p], 0) + (np.log((pauses[p][1] - pauses[p][0]) * 0.01) if p < P else 0)
    for k in range(1, n):
        for p in range(k, P + 1):
            if p == P and k != n - 1:
                continue
            for q in range(k - 1, p):  # previous sentence ended at pause q; this one starts after it
                if dp[k - 1][q] <= NEG / 2:
                    continue
                s = pauses[q][1]
                sc = dp[k - 1][q] - dur_cost(s, ends[p], k)
                if p < P:
                    sc += np.log((pauses[p][1] - pauses[p][0]) * 0.01)
                if sc > dp[k][p]:
                    dp[k][p], back[k][p] = sc, q
    if dp[n - 1][P] <= NEG / 2:
        return None
    chosen = [P]
    for k in range(n - 1, 0, -1):
        chosen.append(back[k][chosen[-1]])
    chosen = chosen[::-1]  # end option of each sentence
    bounds, s = [], first
    for p in chosen:
        e = ends[p]
        bounds.append((s * 0.01, e * 0.01))
        if p < P:
            s = pauses[p][1]
    return _pad_spans(bounds, len(audio) / sr, pad)


def _mfcc(audio, sr, hop_s=0.01):
    import librosa

    y = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
    m = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=13, hop_length=int(16000 * hop_s), n_fft=400)
    m = np.vstack([m, librosa.feature.delta(m)])
    return (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-8)


def cuts_by_dtw(audio, sr, references, snap=0.3):
    """Internal sentence cut times for a recording whose sentences are also in `references`.

    references: list of (ref_audio, ref_sr, ref_cut_times) for recordings of the same sentences with known
    cuts (e.g. midpoints of the between-sentence silences in their transcripts). Each reference is aligned
    to the target with DTW on MFCC + delta features; each reference cut maps to the median target time on
    the path; the median over references is then snapped to the middle of the nearest pause within `snap` s.
    """
    import librosa

    X = _mfcc(audio, sr)
    mapped = []
    for ra, rsr, rcuts in references:
        Y = _mfcc(ra, rsr)
        _, wp = librosa.sequence.dtw(X=Y, Y=X, metric="cosine")
        wp = wp[::-1]  # (ref_frame, target_frame), increasing
        mapped.append([float(np.median(wp[wp[:, 0] == min(int(round(c / 0.01)), wp[-1, 0]), 1])) * 0.01
                       for c in rcuts])
    mapped = np.array(mapped)
    cuts = np.median(mapped, axis=0)
    spread = mapped.max(axis=0) - mapped.min(axis=0)  # disagreement between references, seconds
    speech = _speech_mask(audio, sr)
    out = []
    for c in cuts:
        lo, hi = int(max(0, (c - snap) / 0.01)), int(min(len(speech), (c + snap) / 0.01))
        best, best_d = c, None
        i = lo
        while i < hi:  # silent runs in the window: snap to the middle of the one nearest to c
            if not speech[i]:
                j = i
                while j < hi and not speech[j]:
                    j += 1
                mid = (i + j) / 2 * 0.01
                if best_d is None or abs(mid - c) < best_d:
                    best, best_d = mid, abs(mid - c)
                i = j
            else:
                i += 1
        out.append(best)
    return out, spread.tolist()


def spans_from_cuts(audio, sr, cuts, pad=0.2):
    """Sentence spans given internal cut times: speech extent within each interval, padded."""
    speech = _speech_mask(audio, sr)
    total = len(audio) / sr
    edges = [0.0] + list(cuts) + [total]
    bounds = []
    for a, b in zip(edges[:-1], edges[1:]):
        idx = np.flatnonzero(speech[int(a / 0.01):int(b / 0.01)])
        if len(idx) == 0:
            bounds.append((a, b))
        else:
            bounds.append((a + idx[0] * 0.01, a + (idx[-1] + 1) * 0.01))
    return _pad_spans(bounds, total, pad)


def spans_from_pauses(audio, sr, n, min_pause=0.15, pad=0.2):
    """Split a recording of n sentences at its n-1 longest internal pauses (energy-based voice activity).

    Returns [(start, end)] in order, or None if fewer than n-1 internal pauses are found. Kept as the
    baseline that spans_from_pauses_expected improves on."""
    hop = int(sr * 0.01)
    frames = len(audio) // hop
    e = 10 * np.log10((audio[: frames * hop].reshape(frames, hop) ** 2).mean(1) + 1e-12)
    lo, hi = np.percentile(e, 10), np.percentile(e, 95)
    speech = e > lo + 0.35 * (hi - lo)
    # close small gaps inside words, drop isolated blips
    k = 5
    speech = np.convolve(speech.astype(float), np.ones(k) / k, mode="same") > 0.5
    idx = np.flatnonzero(speech)
    if len(idx) == 0:
        return None
    first, last = idx[0], idx[-1]
    pauses = []  # (length, start_frame, end_frame) of internal silent runs
    i = first
    while i <= last:
        if not speech[i]:
            j = i
            while j <= last and not speech[j]:
                j += 1
            if (j - i) * 0.01 >= min_pause:
                pauses.append((j - i, i, j))
            i = j
        else:
            i += 1
    if len(pauses) < n - 1:
        return None
    cuts = sorted(pauses, key=lambda p: -p[0])[: n - 1]
    cuts = sorted(cuts, key=lambda p: p[1])
    bounds = [(first * 0.01, None)]
    for _, a, b in cuts:
        bounds[-1] = (bounds[-1][0], a * 0.01)
        bounds.append((b * 0.01, None))
    bounds[-1] = (bounds[-1][0], (last + 1) * 0.01)
    total = len(audio) / sr
    spans = []
    for k, (s, e) in enumerate(bounds):
        prev_end = bounds[k - 1][1] if k > 0 else 0.0
        next_start = bounds[k + 1][0] if k + 1 < len(bounds) else total
        spans.append((max(s - pad, (prev_end + s) / 2 if k > 0 else 0.0),
                      min(e + pad, (e + next_start) / 2 if k + 1 < len(bounds) else total)))
    return spans


def usc_files(spk):
    for mat in sorted((USC_ROOT / spk / "mat").glob("*.mat")):
        m = re.search(r"_(\d{3})_(\d{3})\.mat$", mat.name)
        first, last = int(m.group(1)), int(m.group(2))
        trans = USC_ROOT / spk / "trans" / (mat.stem + ".trans")
        yield mat, first, last, trans if trans.exists() else None


TRANSCRIBED = ("M1", "F1", "F5")  # speakers whose .trans files match their EMA-session audio


def _transcript_cuts(mat, trans, ids, read_audio):
    audio, sr = read_audio(mat)
    sp = spans_from_trans(read_trans(trans), len(audio) / sr, pad=0.0)
    return audio, sr, [(sp[ids[k]][1] + sp[ids[k + 1]][0]) / 2 for k in range(len(ids) - 1)]


def usc_segments(spk, read_audio):
    """Segments of one USC-TIMIT speaker. read_audio(mat) -> (audio, sr).

    M1, F1, F5: sentence spans from their transcripts. M3 (transcripts from the MRI session): sentence cuts
    transferred from M1, F1 and F5's recordings of the same sentences by DTW (cuts_by_dtw); in a
    leave-one-speaker-out check on M1/F1/F5 (two references) 2-4% of transferred cuts fell outside the true
    between-sentence silence, so `cut_spread_s` (reference disagreement) is recorded to flag doubtful ones."""
    texts = usc_sentence_texts()
    ref_files = {} if spk in TRANSCRIBED else \
        {r: {(f, l): (m, t) for m, f, l, t in usc_files(r)} for r in TRANSCRIBED}
    for mat, first, last, trans in usc_files(spk):
        ids = list(range(first, last + 1))
        audio, sr = read_audio(mat)
        spread = [0.0] * (len(ids) - 1)
        if spk in TRANSCRIBED:
            if trans is None:
                print(f"skip {mat.name}: no transcript")
                continue
            spans = spans_from_trans(read_trans(trans), len(audio) / sr)
            method = "transcript"
            missing = [i for i in ids if i not in spans]
            if missing:
                raise ValueError(f"{trans}: sentences {missing} not in transcript")
            spans = [spans[i] for i in ids]
        else:
            refs = [_transcript_cuts(*ref_files[r][(first, last)], ids, read_audio)
                    for r in TRANSCRIBED if (first, last) in ref_files[r] and ref_files[r][(first, last)][1]]
            if not refs:
                print(f"skip {mat.name}: no reference recording of sentences {first}-{last}")
                continue
            cuts, spread = cuts_by_dtw(audio, sr, refs)
            spans = spans_from_cuts(audio, sr, cuts)
            method = "dtw"
        for k, (sid, (s, e)) in enumerate(zip(ids, spans)):
            # a sentence's boundary uncertainty: the larger spread of its two neighbouring cuts
            nbrs = ([spread[k - 1]] if k > 0 else []) + ([spread[k]] if k < len(spread) else [])
            sp = max(nbrs) if nbrs else 0.0
            yield Segment(corpus="usc_timit", speaker=f"usc_{spk}", utt_id=f"usc_{spk}_{sid:03d}", mat=mat,
                          start=s, end=e, text=texts.get(sid, ""), norm_group=usc_norm_group(spk, sid),
                          meta={"sentence_id": sid, "segmentation": method, "cut_spread_s": round(float(sp), 3)})


# --- EMA_5EMO --------------------------------------------------------------------------------

EMO_NAME = re.compile(r"ema_5emo_(?P<spk>\w\w)_(?P<emo>[a-z]{3})_(?P<rate>norm|fast)_"
                      r"(?P<kind>sent|pssg)(?P<num>\d+)(?:_rep(?P<rep>\d+))?_utt(?P<utt>\d+)\.mat$")


def perceived_emotion_codes():
    """Raw perceived-emotion code per file stem from <spk>_best_emo (code meaning undocumented)."""
    out = {}
    for spk in EMO_SPEAKERS:
        f = EMO_ROOT / spk / f"{spk}_best_emo"
        if f.exists():
            for line in f.read_text().splitlines():
                p = line.split()
                if len(p) >= 2:
                    out[p[0]] = p[1]
    return out


def passage_phrase_texts():
    """Phrase texts of pssg_short, from passage_brief.txt (abbreviated with '~' in the source)."""
    out, block = {}, 0
    for line in (EMO_ROOT / "pssg_short" / "passage_brief.txt").read_text().splitlines():
        m = re.match(r"\s*\((\d+)\)\s*(.*)", line)
        if m:
            if int(m.group(1)) == 1:
                block += 1
            out[(block, int(m.group(1)))] = m.group(2).strip()
    return out


PASSAGE_PHRASES = {1: 10, 2: 5}
PASSAGE_TEXT = {
    1: "My grandfather: You wished to know all about my grandfather. Well, he is nearly ninety-three years old; "
       "he dresses himself in an ancient black frock coat, usually minus several buttons; yet he still thinks as "
       "swiftly as ever. A long, flowing beard clings to his chin, giving those who observe him a pronounced "
       "feeling of the utmost respect. When he speaks, his voice is just a bit cracked and quivers a trifle. "
       "Twice each day he plays skillfully and with zest upon our small organ. Except in the winter when the ooze "
       "or snow or ice prevents, he slowly takes a short walk in the open air each day. We have often urged him "
       "to walk more and smoke less, but he always answers, \"Banana oil!\" Grandfather likes to be modern in his "
       "language.",
    2: "The North Wind and the Sun were disputing which was the stronger, when a traveler came along wrapped in a "
       "warm cloak. They agreed that the one who first succeeded in making the traveler take his cloak off should "
       "be considered stronger than the other. Then the North Wind blew as hard as he could, but the more he blew "
       "the more closely did the traveler fold his cloak around him, and at last the North Wind gave up the "
       "attempt. Then the Sun shone out warmly, and immediately the traveler took off his cloak. And so the North "
       "Wind was obliged to confess that the Sun was the stronger of the two.",
}


def emo_segments(read_audio=None):
    """EMA_5EMO utterances: sentences, passage phrases (pssg_short) and full passages not covered by phrases.

    Most pssg_short phrases are exact excerpts of a full passage recording (same audio and EMA); those are
    located (exact audio match), which fixes their passage and phrase order regardless of file names, and the
    full passage they cover is then not emitted. Phrases that are not excerpts of any full passage (jn fear:
    a different take of passage 1 and 2) are kept as their own recordings, ordered by utterance number
    (10 phrases of passage 1, then 5 of passage 2) and flagged located=False; the full passages of that
    speaker/emotion/rate are then emitted as long utterances since they are separate recordings."""
    if read_audio is None:
        raise ValueError("passage phrases need read_audio to be located in their passages")
    codes = perceived_emotion_codes()
    fulls = {}  # (spk, emo, rate) -> {passage: (match, mat)}
    for spk in EMO_SPEAKERS:
        for mat in sorted((EMO_ROOT / spk / "wav_mat").glob("ema_5emo_*.mat")):
            m = EMO_NAME.match(mat.name)
            if m is None:
                print(f"skip unparsed {mat.name}")
                continue
            if m["kind"] == "pssg":
                fulls.setdefault((spk, m["emo"], m["rate"]), {})[int(m["num"])] = (m, mat)
                continue
            yield _emo_segment(m, mat, EMO_SENTENCES[int(m["num"])], codes, read_audio,
                               {"prompt": f"sent{m['num']}"})
    phrase_text = passage_phrase_texts()
    groups = {}
    for mat in sorted((EMO_ROOT / "pssg_short" / "wav_mat").glob("ema_5emo_*.mat")):
        m = EMO_NAME.match(mat.name)
        if m is None:
            print(f"skip unparsed {mat.name}")
            continue
        groups.setdefault((m["spk"], m["emo"], m["rate"]), []).append((m, mat))
    covered = set()  # (spk, emo, rate, passage) of full passages whose content is in located phrases
    for key, items in sorted(groups.items()):
        full_audio = {p: read_audio(mat)[0] for p, (_, mat) in fulls.get(key, {}).items()}
        items = sorted(items, key=lambda x: int(x[0]["utt"]))
        hits = []
        for m, mat in items:
            audio = read_audio(mat)[0]
            hits.append(next(((p, off) for p, fa in full_audio.items() if (off := locate(audio, fa)) is not None),
                             None))
        if all(h is None for h in hits):
            raise ValueError(f"{key}: no phrase is an excerpt of a full passage; cannot assign passages")
        # unlocated phrases join the passage of the nearest located phrase before them (else after them); one
        # between located phrases of two different passages goes to the passage short of its canonical count
        passages = []
        for i, h in enumerate(hits):
            if h is not None:
                passages.append(h[0])
                continue
            before = [hits[j][0] for j in range(i - 1, -1, -1) if hits[j] is not None]
            after = [hits[j][0] for j in range(i + 1, len(hits)) if hits[j] is not None]
            if before and after and before[0] != after[0]:
                # phrases each passage still lacks: canonical count - assigned so far - located ones still to come
                need = {p: PASSAGE_PHRASES[p] - passages.count(p)
                        - sum(1 for j in range(i + 1, len(hits)) if hits[j] is not None and hits[j][0] == p)
                        for p in (before[0], after[0])}
                passages.append(max(need, key=lambda p: (need[p], p == before[0])))
            else:
                passages.append(before[0] if before else after[0])
        for passage in sorted(set(passages)):
            idx = [i for i in range(len(items)) if passages[i] == passage]
            offs = [hits[i][1] for i in idx if hits[i] is not None]
            if offs != sorted(offs):
                raise ValueError(f"{key} passage {passage}: utterance order and positions in the recording disagree")
            canonical = len(idx) == PASSAGE_PHRASES[passage]
            covered.add((*key, passage)) if offs else None
            for k, i in enumerate(idx):
                m, mat = items[i]
                meta = {"prompt": f"pssg{passage}_phrase{k + 1:02d}", "located": hits[i] is not None,
                        "phrases_in_passage": len(idx)}
                if hits[i] is not None:
                    meta["passage_offset_samples"] = hits[i][1]
                # phrase texts follow passage_brief.txt only where the phrase split is the canonical one
                text = phrase_text.get((passage, k + 1), "") if canonical else ""
                yield _emo_segment(m, mat, text, codes, read_audio, meta)
    for key, ps in sorted(fulls.items()):
        for passage, (m, mat) in sorted(ps.items()):
            if (*key, passage) not in covered:
                yield _emo_segment(m, mat, PASSAGE_TEXT[passage], codes, read_audio,
                                   {"prompt": f"pssg{passage}_full"})


def locate(short, full, probe=4096):
    """Sample offset of `short` inside `full` if it is an exact excerpt, else None."""
    n = min(probe, len(short))
    if len(full) < len(short) or n == 0:
        return None
    from scipy.signal import correlate

    # probe with the most energetic window: a near-silent probe (phrases often start in silence) has no
    # distinctive correlation peak
    energy = np.convolve(short**2, np.ones(n), mode="valid")
    p0 = int(np.argmax(energy))
    c = correlate(full, short[p0:p0 + n], mode="valid", method="fft")
    for pos in np.argsort(c)[::-1][:10]:
        off = int(pos) - p0
        if off < 0:
            continue
        seg = full[off:off + len(short)]
        if len(seg) == len(short) and np.allclose(seg, short, atol=1e-9):
            return off
    return None


def _emo_segment(m, mat, text, codes, read_audio, extra):
    spk = m["spk"]
    if read_audio is not None:
        audio, sr = read_audio(mat)
        end = len(audio) / sr
    else:
        end = None
    utt = f"5emo_{spk}_{m['emo']}_{m['rate']}_{extra['prompt']}" + (f"_rep{m['rep']}" if m["rep"] else "") \
        + f"_utt{int(m['utt']):03d}"
    meta = {"emotion": EMOTIONS[m["emo"]], "rate": "normal" if m["rate"] == "norm" else "fast",
            "repetition": int(m["rep"] or 0), "source_utt": int(m["utt"]),
            "perceived_emotion_code": codes.get(mat.stem, ""), **extra}
    return Segment(corpus="ema_5emo", speaker=f"5emo_{spk}", utt_id=utt, mat=mat, start=0.0, end=end,
                   text=text, norm_group=f"5emo_{spk}", meta=meta)
