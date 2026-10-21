# CLAUDE.md

Living summary of the project's current direction and findings. Changes go in `CHANGELOG.md`.

## Current direction (2026-10)

Compare three vocoders that synthesize 24 kHz speech from SPARC articulatory features: HiFi-GAN (the fork's
generator, retrained), a DDSP harmonic-plus-noise vocoder (after DDSP-Articulatory-Vocoder and RT-VC) and Vocos
adapted to articulatory input. All three use the same 15 features per 50 Hz frame (12 EMA channels, pitch, loudness,
periodicity), the same speaker conditioning (FiLM from a SPARC speaker embedding), the same data (filtered LibriTTS-R)
and the same training recipe and budget. The EMA comes from the per-utterance MNGU0 refit of the linear AAI head
(`linear_aai_mngu0_refit_peralign.pkl`), not from the head embedded in the `en+` checkpoint.

Status (2026-10-20): all three phases and the follow-ups (test.other, second seeds with their probes, X-Y probes, 800k runs of HiFi-GAN
and Vocos) are done; no runs are pending. The three primary runs (400k generator steps, fp32, one GPU each) and 12
ablation runs (100k steps) are trained, and the Phase 3 evaluation on test.clean is complete (code `sparc.vocoders.eval`,
contract `docs/vocoders/EVALUATION.md`). The write-up is `docs/vocoders/RESULTS.md`; tables are in
`docs/vocoders/results/`. Large outputs (checkpoints, audio, per-utterance results, samples, failure cases) are under
`/data/user_data/xoy/sparc-vocoders/` (`runs/`, `eval/`).

Work built on this branch elsewhere (2026-10-20): speaker-embedding Phase 3 on branch `spkemb-phase3` trains a
resynthesis-oriented, interpolatable speaker encoder (all-layer WavLM statistics, VAE latent, cross-utterance
references) with this branch's HiFi-GAN recipe; six 100k-step screening arms are running (issue #31; design in
`docs/speaker_embedding_plan.md` on branch `multispeaker-eval`). Its runs, caches and evaluation outputs are under
`/data/user_data/xoy/spkemb/phase3/`, separate from `sparc-vocoders/`.

## Main findings (test.clean; robust = same sign and supported in both training seeds and larger than the seed spread)

- HiFi-GAN leads on quality and speaker similarity (robust, also on test.other): UTMOS 4.17 / 4.16 (two seeds) against
  3.90 / 3.94 (Vocos) and 3.83 / 3.82 (DDSP); ground truth 4.21. On test.other HiFi-GAN equals ground-truth UTMOS (4.03).
- WER does not separate the vocoders in either seed or on test.other (about 2.5 percent on test.clean, 5.0 percent on
  test.other; ground truth 1.9 and 3.4 percent).
- Prosody failures differ (robust): Vocos has the largest F0 error (about 35 cents RMSE); DDSP has the largest voicing and
  loudness errors. The HiFi-GAN-DDSP F0 ordering and the Vocos-DDSP UTMOS ordering are not robust.
- F0 and loudness edits are followed closely; single EMA channel edits are not (gain about 0.1). Both replicate in the
  second seed (RESULTS.md section 5.2); HiFi-GAN and Vocos follow EMA edits slightly more than DDSP (by about 0.02, robust). Pairing the X and Y
  edits of a sensor does not help; time-varying edits and the direction within a sensor do (best about 0.57); the X-Y
  findings also replicate in the second seed (section 5.3).
- Compute: Vocos uses 11.5 times fewer FLOPs than HiFi-GAN and is 7-10 times faster at inference, but only 1.5 times
  faster to train; the SPARC feature extractor dominates the end-to-end cost.
- Training length (one 800k run each of HiFi-GAN and Vocos, against both 400k seeds; RESULTS.md section 6.1.3): HiFi-GAN
  followed the projection (dev mel L1 -0.0075; UTMOS 4.207, equal to the ground truth; every metric except WER, CER and
  Sim spk robustly better) and is still about 3 percent above its fitted asymptote. Vocos gained less than projected and
  mostly within its seed spread (UTMOS 3.97; within about 1 percent of its asymptote). DDSP had converged at 400k by
  projection and was not run longer. The HiFi-GAN lead holds at equal steps (UTMOS +0.24 at 800k) and at about equal
  training compute (Vocos 800k, 46 GPU-hours, against HiFi-GAN 400k, about 34: UTMOS +0.19). Vocos at 800k is ahead of
  DDSP at 400k on UTMOS by about 0.14 (supported against both DDSP seeds; single run).
- Ablations at 100k: Vocos option B beats A; MSD instead of MRD does not change the ranking; the native SPARC recipe
  raises UTMOS but worsens mel L1, PESQ and WER; at about 6 M parameters the HiFi-GAN lead over DDSP is not supported.

## Export (2026-10-21)

`sparc-export` (docs/vocoders/EXPORT.md) writes inference exports (generator + speaker FFN, about 60 MB, or 28 MB for
DDSP) and uploads them to the Hugging Face Hub only with `--upload`. Recommended checkpoints: HiFi-GAN and Vocos
`main_800k` step 800k, DDSP `main` step 400k (seed 0); all three verified exact against their checkpoints, and the
documented WAV-to-input path reproduces the cached inputs to within 1.2e-4. Local exports: `$SV_ROOT/exports/<vocoder>`.
Uploaded 2026-10-21 as private Hub repositories (MIT, with the refit inversion head): `nzxyin/sparc-hifigan-24k`,
`nzxyin/sparc-vocos-24k`, `nzxyin/sparc-ddsp-24k`; loading them by repo id reproduces the local exports exactly.

## Measurements so far

- fp32 train step at the main shape (batch 16 × 64 frames, RTX 6000 Ada, `cudnn.benchmark` off): HiFi-GAN 304 ms,
  DDSP 236 ms, Vocos 214 ms (bf16: about 120-160 ms). A 400k-step run is about 24-34 GPU-hours in fp32; the 800k-step
  runs took 69.7 (HiFi-GAN) and 45.9 (Vocos) GPU-hours of wall time including restarts.
- Feature extraction: about 144 s per audio hour on an L40 (batch size 1), about 23 GPU-hours for all 572 h.
- Overfit regime (4 utterances, 4k steps): final mel L1 HiFi-GAN 0.18, DDSP 0.29 (plateaus from about 2k steps),
  Vocos 0.08; HiFi-GAN output about 3 dB quieter than the reference at that point.
- "fp32" runs use cuDNN TF32 convolutions (PyTorch default); the feature extractor disables TF32.
- RTX A6000 nodes (21 % of `preempt` GPUs) run this fp32 training about 4 times slower (HiFi-GAN 1.35 s/step);
  submit with `SBATCH_CONSTRAINT="L40S|L40|6000Ada|RTX_PRO_6000|A100_80GB|A100_80G|A100_40GB|H100|H200"`.
- Evaluation of one vocoder on test.clean (3 conditions): about 1.5 GPU-hours plus about 45 CPU-minutes (signal
  metrics, one process per system and condition).

## Findings that constrain the implementation

The legacy 16 kHz bugs below (#8-#16) are fixed on `main`, which was merged into this branch on 2026-10-20. The
feature cache and every run here predate the merge; the vocoder code path never used the affected functions in the
buggy way (see each item), so they stay valid. Checked on 5 cached test.clean utterances (one L40S): the merged and
pre-merge `SparcFeatureExtractor` give bit-identical arrays, and both match the cache within 1.1e-5 (loudness exactly).
- `load_model("en+", linear_model_path=...)` kept the checkpoint's own head (#11; now an explicit path overrides it and
  is exposed as `model.linear_model_path`). The cache loads the refit head through `linear_model_state_dict` and
  asserts the loaded weights; keep that.
- With `normalize: true`, every waveform is z-scored per utterance before WavLM, CREPE and the loudness convolution
  (`src/sparc/speech.py`), so the cached loudness does not change with gain. Gain augmentation needs an
  un-normalized loudness, cached ungained and multiplied by the gain.
- torchcrepe adds random dither to pitch (#13). The cache seeds numpy per utterance itself (seed `crc32(id)`); since the
  merge, `encode(seed=...)` and `sparc-encode`'s `deterministic_pitch` do the same for the legacy API.
- Batched `encode()` corrupted short utterances and very short utterances could not be encoded (#12; fixed by per-
  utterance filtering). The cache was extracted with batch size 1 and its contract keeps one utterance per call.
- Feature length for 24 kHz input is `T = floor(L24/480) - 1 + [L24 mod 480 >= 119]`; frame j of loudness, pitch and
  the WavLM CNN is centred at 24 kHz samples 480j, 480j + 120 and 480j + 300.
- The `en+` speaker embedding pools WavLM `hidden_states[6]`; `training/prepare_spk_raw.py` pools layer 0 with
  different periodicity settings unless run with `--en-plus-compatible` (#13). The cache pools layers 0 and 6 itself.
- The legacy training module (`sparc.training`, 16 kHz) is now preemption-safe, counts batches, supports DDP, and has
  the fixed lr schedule and MSD pooling (#9, #10, #15). The vocoder comparison uses its own `sparc.vocoders.training`
  module and ported, fixed discriminators throughout, so none of its results depend on these fixes.
