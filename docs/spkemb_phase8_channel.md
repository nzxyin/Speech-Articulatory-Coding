# Phase 8.1: channel ceiling and input path (2026-10-29)

Pre-registration: `docs/speaker_embedding_plan.md` "Phase 8", 8.1 (multispeaker-eval branch, commit bbb9601). Code:
`src/sparc/vocoders/eval/ood_channel.py`, tests `tests/test_spkemb_ood_channel.py`. Results and per-item rows:
`/data/user_data/xoy/spkemb/phase8/channel/{summary.md,summary.json,choice.json,val,test}`. Model: the 24 kHz HiFi-GAN run `p5_fv`
(filtered LibriTTS-R + GLOBE + VCTK, up to 8 references, speaker-relative pitch, 100k steps). No training.

## Design as run

Own-speaker items of the cloning plans (K = 5 references; GLOBE: its 2), scored against the speaker's other ground-truth
utterances (held-out references where they exist; GLOBE uses all). Validation: LibriTTS-R dev-clean (100 items, 30 speakers), the
VCTK validation half (48 items, 19 speakers), GLOBE real-speaker dev pool `globe_spk` (100 items, 100 speakers). Test: LibriTTS-R
test-clean (100 items, 29 speakers), VCTK test half (52 items, 18 speakers), GLOBE pseudo-speaker test half (9 items, 8 speakers;
descriptive). Verifiers ECAPA, WavLM-SV, CAM++; the primary similarity is the mean of the three per speaker. Intervals: speaker-level
bootstrap, 4000 draws.

Conditions:
- standard resynthesis; ground-truth recording;
- (i) channel ceiling: the recording through a **separate mel-input vocoder, Vocos mel copy synthesis** (`charactr/vocos-mel-24khz`,
  the reference system of the vocoder comparison, pinned revision, SHA-256 checked). A mel-input vocoder exists, so the band-limiting
  fallback was not used. Variants: `chan_vocos` (recording at the evaluation level) and `chan_vocos_lm` (recording scaled to the RMS
  of the standard resynthesis);
- (ii) the target's own content features with F0 and loudness from another utterance of the same speaker. Variants: `ctrl_contour`
  (that utterance's voiced ln-F0 sequence and loudness curve stretched linearly over the target; voicing and articulators kept) and
  `ctrl_level` (target contours, median F0 and loudness level of the other utterance);
- (iii) content features of an utterance of a different speaker (same pool and split), F0 range matched to the target speaker's
  references, the target speaker's standard code.

Selection on validation (all three pools, pools weighted equally): channel variant = highest fraction of ground truth; control
variant = lowest similarity. Chosen: `chan_vocos` (0.9934 vs 0.9932 of ground truth) and `ctrl_contour` (similarity 0.632 vs 0.708).
Both selection criteria were fixed by me before the validation numbers existed and are the choices least favourable to the "channel"
reading and to "(ii) does not move". Test rendered and reported only the chosen variants, once.

## Results (mean of the three verifiers; similarity, fraction of ground truth, UTMOS)

| condition | VCTK validation (19 spk) | VCTK test (18 spk) | LibriTTS-R test (29 spk) | GLOBE real-speaker validation (100 spk) |
|---|---|---|---|---|
| standard resynthesis | 0.661 / 0.789 / 3.65 | 0.692 / 0.808 / 3.59 | 0.820 / 0.982 / 3.82 | 0.657 / 0.817 / 2.94 |
| ground truth | 0.837 / 1.000 / 3.94 | 0.856 / 1.000 / 3.98 | 0.835 / 1.000 / 4.26 | 0.804 / 1.000 / 3.01 |
| (i) Vocos copy of the recording | 0.833 / **0.995** [0.991, 0.999] / 3.78 | 0.852 / **0.994** [0.991, 0.998] / 3.76 | 0.832 / 0.996 / 3.98 | 0.795 / 0.989 / 2.56 |
| (ii) other utterance's F0 and loudness contour | 0.598 / 0.714 / 2.08 | 0.626 / 0.731 / 1.99 | 0.728 / 0.872 / 1.75 | 0.579 / 0.720 / 1.64 |
| (iii) other speaker's content | 0.581 / 0.694 / 3.70 | 0.605 / 0.706 / 3.56 | 0.764 / 0.915 / 3.76 | see summary.md |

Difference to standard as a fraction of the standard gap (G - A), VCTK validation: (iii) -0.45 [-0.62, -0.29]; (ii) contour
-0.36 [-0.45, -0.26]; (ii) level 0.00 [-0.02, 0.02]. VCTK test: (iii) -0.53 [-0.69, -0.40]; (ii) contour -0.40 [-0.55, -0.28].
Per verifier on VCTK validation (standard / channel ceiling as fraction of ground truth): ECAPA 0.709 / 0.992, CAM++ 0.642 /
0.996, WavLM-SV 0.977 / 0.997 (nearly saturated). LibriTTS-R is already at 0.98 of ground truth with the standard code (validation
0.984), which is why the gap is an out-of-domain one. GLOBE test (9 items, 8 speakers, descriptive): standard 0.850, ceiling 0.996.
UTMOS of the Vocos copy is below the recording (3.78 vs 3.94 on VCTK validation) and above the standard resynthesis.
GLOBE rows and all verifiers with speaker-level intervals are in `summary.md`.

## Rule outcome

- **Channel versus voice (VCTK validation half): mainly voice.** The channel-ceiling similarity is 0.995 [0.991, 0.999] of ground truth, far
  above the 0.95 line, and the same on the test half (0.994), LibriTTS-R (0.989) and GLOBE. A perfect voice passed through a
  mel-input vocoder at this bandwidth and level keeps the speaker-similarity score on all three verifiers. The 0.79 of ground truth
  the standard resynthesis reaches is therefore not a limit of the channel for these verifiers. This is a ceiling for Vocos at 24
  kHz, not for every possible channel (recording-specific noise, reverberation and codecs beyond what Vocos passes are not
  covered, and VCTK ground truth is the original recording itself).
- **Content path limits the gap: no, by the rule as pre-registered with the selected control.** Condition (iii) differs from
  standard by -0.45 of the gap (CI excludes 0), but (ii) with the selected `ctrl_contour` also does (-0.36, CI excludes 0), so the
  "(ii) does not" clause fails; test agrees (-0.53 and -0.40). Sensitivity, reported for transparency: with the other control
  variant `ctrl_level`, (ii) does not move, and the literal rule would read "content path limits the gap". Direction matters
  there: similarity **falls** when the content comes from another speaker, so the standard score is helped, not limited, by the
  target's own content features (they carry identity of the target utterance's own speech, partly through prosody, since replacing
  only the contour already costs 0.36 of the gap). This is the opposite of the reading the rule was written to detect (alternative
  features raising similarity), and no raising condition exists in the data.
- Caveat on `ctrl_contour`: UTMOS falls to about 2.0-2.1, so part of its drop is an unnatural stretched contour, not the speaker's
  prosody; it shows that the standard score depends on the utterance's own prosody, not by how much the speaker.

## Consequence for the remaining gap

With the channel ceiling at about 0.99, the standard resynthesis' deficit (VCTK about 0.19-0.21 of ground truth on the mean verifier,
about 0.28 on ECAPA) sits in the voice rendering of the generator and its code, as far as these verifiers show, not in what
a mel vocoder does to the recording. The content path does not explain it either (it carries the target's own identity, so swapping
it costs similarity). That leaves the speaker code and the generator's conditioning, which are what 8.2 (code fitting) and the
conditioning-mechanism work address.
