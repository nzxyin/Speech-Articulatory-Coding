# Phase 8 (speaker-embedding out-of-domain follow-up): results

Pre-registration: `docs/speaker_embedding_plan.md` "Phase 8" (branch `multispeaker-eval`, commit bbb9601). Section 8.1 (channel ceiling)
is in `docs/spkemb_phase8_channel.md`; the conditioning study is in `docs/spkemb_conditioning_screen.md`. This file holds section 8.2.
Rule for every section: choices on validation splits, test run once for the chosen settings (`docs/spkemb_validation_protocol.md`).

## 8.2 Constrained speaker-code fitting (2026-10-29)

Code `src/sparc/vocoders/eval/ood_svfit_p8.py` (builds on `ood_svfit.py`), tests `tests/test_ood_svfit_p8.py`. Model: the 24 kHz HiFi-GAN
run `p5_fv` (filtered LibriTTS-R + GLOBE + VCTK, up to 8 references, speaker-relative pitch, 100k), no training. Results
`/data/user_data/xoy/spkemb/phase8/svfit/` (`choice.json`, `sweep_table.md`, `summary.{json,md}`, per-item rows under `sweep/`, `test/`,
`test_swap/`), synthesized audio `/data/user_data/xoy/spkemb/phase8/svfit/audio/{own,swap}/<pool>/<arm>/` (own: `std`, `unc`, `l2`,
`nat`, named by own-speaker item id; swap: `std`, `fit`, named by swap id; 24 kHz PCM_16, 242 MB).

### Method

The 64-d speaker code is fitted with Adam on a 4 s content crop of the item per step, through the frozen generator and the
differentiable ECAPA-TDNN:

`loss = -cos(ECAPA(output), e_refs) + lam * |dz|^2 / |z0|^2 + nat * D_mel(output, standard output)`

`e_refs` is the mean ECAPA embedding of the speaker's code-reference utterances (K = 5; GLOBE all 2 to 7; swaps: the target's 5
references, or its single reference); the target utterance's audio is never used (the content features of the item are the input
being rendered). Arms: standard code; `l2` (`nat = 0`); `nat` (`lam > 0`, `nat > 0`); `unc` (`lam = nat = 0`, the unconstrained
reference).

Naturalness term, fixed before tuning: **log-mel statistics**. `D_mel` is the mean squared difference between the per-band mean and
per-band standard deviation over time of the natural-log mel spectrogram (24 kHz, 80 bands, n_fft 1024, hop 480, 0 to 12 kHz) of the
fitted output and of the standard-code output on the same crop with the same noise seed. WavLM features were not used.

Verifiers: ECAPA is the loss only. Selection uses CAM++ only. WeSpeaker ResNet34 is held out from loss and selection; the advance
rule is judged on it (CAM++ closure reported next to it). **No third verifier was found**: no speaker-verification model with
documented non-VoxCeleb training data exists offline (`/data/hf_cache`, `/data/models` and the spkemb hf caches hold ECAPA, CAM++,
WeSpeaker ResNet34, pyannote embedding and WavLM-base-plus-sv, all VoxCeleb-trained). All three judges share a training family, so
"verifier-measured" does not exclude a shared verifier bias.

### Validation (LibriTTS-R dev-clean 12 items, VCTK validation half 19, `globe_spk` 12; one item per speaker)

Grid: lr {0.01, 0.03, 0.1}; `unc` 3 configs, `l2` lam {0.3, 1, 3, 10} (12 configs), `nat` lam {0.3, 3} x nat {1, 3, 10, 30} (24
configs); snapshots at 25, 50, 100, 200, 400 steps (5 x 39 = 195 settings). Selection: highest mean CAM++ similarity over the three
pools, subject to UTMOS drop <= 0.05 and pooled WER rise <= 0.5 points against the standard code in every pool.

| arm | chosen setting | CAM++ dev-clean / VCTK / GLOBE | UTMOS drop (L / V / G) | guard-passing settings |
|---|---|---|---|---|
| standard code | - | 0.757 / 0.504 / 0.568 | - | - |
| ECAPA fit + L2 | lr 0.1, lam 0.3, 100 steps | 0.797 / 0.620 / 0.630 | 0.02 / 0.03 / 0.04 | 36 of 60 |
| ECAPA fit + L2 + naturalness | lr 0.03, lam 0.3, nat 3, 400 steps | 0.808 / 0.615 / 0.655 | -0.08 / 0.04 / 0.00 | 52 of 120 |
| unconstrained (reference) | lr 0.03, 400 steps (no setting passes the guard: best overall) | 0.805 / 0.645 / 0.666 | 0.20 / 0.22 / 0.21 | 0 of 15 |

The setting judged on test is the better (CAM++, 0.693 vs 0.682) of the two constrained arms: **L2 + naturalness**. The selected
step count (400) is the edge of the widened grid, so longer fits were not explored. The validation sets are small (one item per
speaker) and UTMOS/WER guards are noisy at that size; several guard-passing settings are within noise of the choice.

### Test, run once

Own-speaker items (resynthesis of an utterance of the speaker from a code fitted on other utterances; similarity to the speaker's
held-out references; speaker-level bootstrap 95% intervals): LibriTTS-R test-clean 36 items / 20 speakers, VCTK test half 34 items /
18 speakers, GLOBE pseudo-speaker test half 8 items / 8 speakers (descriptive: the real-speaker GLOBE test pool has one reference and
no own items). Swap items (the chosen setting vs standard, 80 seeded swaps per pool; inference variant `k5_f0_loud`): LibriTTS-R test-clean
(35 targets), VCTK test half (20), GLOBE real test speakers `globe_spk_test` (80, one reference each).

Closure = fraction of the standard-to-ground-truth gap closed (ground truth = the original recording of the held-out utterance).

| pool | arm | WeSpeaker sim (held out) | closure | CAM++ sim | closure | UTMOS | WER % |
|---|---|---|---|---|---|---|---|
| VCTK | standard | 0.533 | - | 0.562 | - | 3.56 | 3.7 |
| | unconstrained (reference) | 0.672 | 0.475 [0.417, 0.533] | 0.712 | 0.566 [0.497, 0.636] | 3.38 | 2.8 |
| | L2 | 0.633 | 0.341 [0.265, 0.412] | 0.676 | 0.430 [0.358, 0.506] | 3.52 | 3.0 |
| | **L2 + naturalness (judged)** | 0.631 | **0.333 [0.268, 0.391]** | 0.676 | 0.429 [0.371, 0.488] | 3.54 | 3.9 |
| | ground truth | 0.825 | - | 0.827 | - | 3.97 | 2.2 |
| LibriTTS-R | standard | 0.745 | - | 0.745 | - | 3.82 | 2.3 |
| | L2 + naturalness | 0.780 | (gap tiny: GT 0.784) | 0.786 | (GT 0.780) | 3.84 | 1.9 |
| GLOBE pseudo (8 spk) | standard | 0.580 | - | 0.598 | - | 3.36 | 3.9 |
| | L2 + naturalness | 0.663 | 0.434 [0.306, 0.686] | 0.689 | 0.471 [0.325, 0.794] | 3.25 | 2.6 |

On the fitted verifier (ECAPA) the VCTK closure of the judged setting is 0.766 [0.700, 0.842]: the held-out verifiers see less than half
of what the loss verifier sees. The naturalness term does not beat the L2 penalty on test (VCTK closure 0.333 vs 0.341, UTMOS 3.54 vs 3.52); on
GLOBE it keeps UTMOS and WER better (3.25 / 2.6 vs 3.17 / 3.9) at lower closure. The unconstrained fit closes more (0.475) but costs
0.18 UTMOS on VCTK and 0.48 on GLOBE.

Swap items (judged setting minus standard; speaker-level bootstrap over targets):

| pool | to target, WeSpeaker | to target, CAM++ | to source, WeSpeaker | to source, CAM++ | UTMOS std -> fit | WER % std -> fit |
|---|---|---|---|---|---|---|
| LibriTTS-R | +0.050 [0.043, 0.059] | +0.058 | -0.021 | -0.013 | 3.70 -> 3.66 | 3.0 -> 3.4 |
| VCTK | +0.119 [0.105, 0.132] | +0.124 | -0.011 | -0.018 | 3.63 -> 3.59 | 3.8 -> 4.9 |
| GLOBE real test speakers (1 reference) | +0.108 [0.095, 0.121] | +0.114 | -0.008 | -0.007 | 2.65 -> 2.55 | 6.8 -> 9.3 |

### Rule outcome: advances under the pre-registered rule (verifier-measured); quality costs on swap items

Correction after review (2026-10-29): the pre-registration marks GLOBE as descriptive, and the swap-item guards below
were added after the test numbers were seen. Applied as written (the GLOBE row excluded as descriptive, no post-hoc
guards), every check passes, so the constrained fit advances to the listening test. The swap-item WER and UTMOS costs
are reported as real quality costs, not as rule failures. The gain is verifier-measured until the listening test.

| check | outcome |
|---|---|
| VCTK closure lower bound >= 0.25 on each held-out verifier | pass (WeSpeaker 0.268; CAM++, the selection verifier, 0.371) |
| LibriTTS-R own-speaker drop <= 0.01 | pass (similarity rises: -0.035 WeSpeaker, -0.042 CAM++) |
| swap similarity to the source rises <= 0.02 | pass (it falls in every pool) |
| WER <= +0.5 points, own-speaker items | pass (LibriTTS -0.4, VCTK +0.2, GLOBE -1.3) |
| UTMOS drop <= 0.1, own-speaker items, every pool | **fail**: GLOBE pseudo-speaker pool 0.111 (LibriTTS -0.02, VCTK 0.02) |
| UTMOS drop <= 0.1, swap items | **fail**: GLOBE real test speakers 0.107 |
| WER <= +0.5 points, swap items | **fail**: VCTK +1.05, GLOBE +2.48 (LibriTTS +0.43) |

The last two rows were added to the evaluation after the first summary showed swap-pool WER and UTMOS (the pre-registration says
"in every pool"; swap items are the same pools); they can only add failures, and the outcome is unchanged without them because the
own-speaker GLOBE UTMOS guard already fails. The own-speaker GLOBE pool has 8 items and is descriptive by the pre-registration; with
it excluded the own-speaker guards all pass, but the swap-item GLOBE and VCTK guards still fail. Per the protocol the failures are
recorded, not re-tuned.

### Reading

- Fitting the code on ECAPA from the speaker's other utterances raises held-out-verifier similarity everywhere, including to unseen
  voices in swaps (+0.05 to +0.12 WeSpeaker), without making the output more similar to the source speaker. On VCTK the judged setting
  closes about a third of the gap to the recording on WeSpeaker and 0.43 on CAM++, against 0.77 on the fitted verifier. This is
  verifier-measured (all three verifiers are VoxCeleb-trained; no third verifier exists offline) and not perceptual: 8.3 (listening
  test) decides whether it is audible.
- The quality cost is small on LibriTTS-R and VCTK own-speaker items (UTMOS within 0.02) but visible on GLOBE speakers (UTMOS -0.1,
  WER up on swaps), which is where the code has the most room and the generator the least robustness.
- The naturalness term (log-mel statistics) did not add anything measurable over the L2 penalty on VCTK; its validation advantage came
  from a longer, smaller-step fit on small validation pools.
- Not run (no result exists): a third non-VoxCeleb verifier, a longer-than-400-step fit, mel statistics versus WavLM features for the
  naturalness term, and a perceptual test.

### Acceptance on the 1-reference GLOBE test pool

`clone.calibration_stats` (and so `item_table`) now returns a NaN threshold when no speaker has two references, and acceptance is then
NaN ("not available"), not 0 (`x >= nan` is False, which silently reported 0 % acceptance). Not calibrated from the validation pool
(`globe_spk`): its speakers have two references and a reference-mean similarity of a different scale than a single reference, so the
threshold would not transfer. Test `test_acceptance_is_not_available_without_a_threshold`.

## 8.3 Listening test: preparation (2026-10-29; no listeners recruited, nothing published)

Files: `scripts/spkemb/listening/{build_items,design,make_trials,analyze,power,simulate_responses}.py`, page draft
`artifacts/listening/{index.html,listening.js,test_listening.js}`. Data: `/data/user_data/xoy/spkemb/phase8/listening/`
(`items.json` private key, `manifest.json` public, `stimuli/`, `loudness.csv`, `trials/`).

**Items, fixed now.** VCTK test half, all 34 own-speaker items (18 speakers; primary); GLOBE 8 pseudo-speaker items (descriptive);
LibriTTS-R 6 control items (one per speaker, seeded). Each item: a real recording of a different utterance of the speaker
(reference), the standard-code output, and the 8.2 `nat`-arm output (same content utterance). VCTK and LibriTTS-R references are
held-out recordings (not among the five the code was fitted on, not the content utterance, 2.5-10 s); GLOBE pseudo-speakers have only
two references and both were used by the fit, so their reference was in the fit (flagged `ref_in_fit`).

**Loudness.** All stimuli (references and outputs) get one linear gain to -26 LUFS (ITU-R BS.1770-4 K-weighted, gated;
`torchaudio.functional.loudness`, refined until within 0.01 LU), 16-bit wav at 24 kHz; the measured loudness after scaling is within
-26.00 to -25.99 LUFS and no stimulus clips (peak <= 0.97). Files are named by a salted hash (`<hash>.wav`), so neither the file name nor
the manifest reveals the arm.

**Trials.** 34 per listener: 24 VCTK, 4 GLOBE, 2 LibriTTS-R test trials plus 4 attention checks (2 identical pairs, 2 obvious
mismatches: the same-speaker standard output against a standard output of the opposite gender), about 20-25 minutes. Items are rated
equally often (+-1) across listeners, the fitted output is A half the time per item and per listener (+-1), order is shuffled and checks sit
at spread positions after the second trial. Pilot: 5 listeners (seed 1); main: parametrized (`make_trials.py --n-listeners`).
Per trial: similarity to the reference and naturalness (1-5) for A and B, forced choice of the output more similar to the reference.

**Analysis (`analyze.py`).** Attention checks exclude a listener who fails at least half of the checks. Primary pool VCTK:
preference by a logistic mixed model with listener and speaker random intercepts (variational Bayes) and a two-way cluster
bootstrap; ratings by linear mixed models of the within-trial difference (plus a cumulative-link sensitivity and Wilcoxon on listener
means). The rule uses both preference intervals (both must exclude 0.5) and the naturalness lower bound against a margin
(`--nat-margin`, placeholder 0.25 points; to be fixed with the listener count before the main test). The analysis was tested on simulated
responses (careless listeners excluded; a true conditional preference of 0.70 recovered as 0.685 [0.643, 0.724] with 22 kept listeners).

**Power (`power.py`).** Simulation of the actual main design with the real speaker structure, marginal preference rate 0.60 vs 0.50,
cluster-robust test (type I error 0.04-0.07 in the simulation). With 18 VCTK speakers the speaker variance caps the power: with
listener sd 0.63 and speaker sd 0.52 (the upper 80 % values of a simulated pilot, illustrative only) 60 listeners reach 80 %, with
sd 0.37 / 0.44 fewer. The real number comes from the real pilot (`power.py --pilot-analysis`). Listeners cannot make up for few speakers; if
the pilot shows large speaker variance, more speakers are needed, not more listeners.

**Run.** Page: serve `artifacts/listening/` with `manifest.json`, `trials/<set>.json` and `stimuli/` next to it (or open the page and choose the
two JSON files), `index.html?set=pilot&listener=L01`; responses export as JSON (`session.export()`, optional `LISTENING_HOOKS`).
Analysis: `analyze.py --dir <dir> --set pilot --responses '<files>' --out <out>` in `/data/user_data/xoy/venvs/p8-listening`.
