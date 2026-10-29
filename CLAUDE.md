# CLAUDE.md

Living summary of the project's current direction and findings. Changes go in `CHANGELOG.md`.

## Current direction (2026-10)

Compare three vocoders that synthesize 24 kHz speech from SPARC articulatory features: HiFi-GAN (the fork's
generator, retrained), a DDSP harmonic-plus-noise vocoder (after DDSP-Articulatory-Vocoder and RT-VC) and Vocos
adapted to articulatory input. All three use the same 15 features per 50 Hz frame (12 EMA channels, pitch, loudness,
periodicity), the same speaker conditioning (FiLM from a SPARC speaker embedding), the same data (filtered LibriTTS-R)
and the same training recipe and budget. The EMA comes from the per-utterance MNGU0 refit of the linear AAI head
(`linear_aai_mngu0_refit_peralign.pkl`), not from the head embedded in the `en+` checkpoint.

Status (2026-10-19): all three phases and the follow-ups (test.other, second seeds, X-Y probes, 800k runs of HiFi-GAN
and Vocos) are done; no runs are pending. The three primary runs (400k generator steps, fp32, one GPU each) and 12
ablation runs (100k steps) are trained, and the Phase 3 evaluation on test.clean is complete (code `sparc.vocoders.eval`,
contract `docs/vocoders/EVALUATION.md`). The write-up is `docs/vocoders/RESULTS.md`; tables are in
`docs/vocoders/results/`. Large outputs (checkpoints, audio, per-utterance results, samples, failure cases) are under
`/data/user_data/xoy/sparc-vocoders/` (`runs/`, `eval/`).

Phase 6 (speaker-encoder pooling in the vocoder; pre-registered in `docs/speaker_embedding_plan.md` on the plan branch) is
prepared but not started: arms `p6_<e1|e2|e3>_<p5_r|p5_f|p5_fv>` (`speaker.layer` `l1l6` / `vus3_l6` / `l6_rep3`) wait for the
Phase 5 winner. Side vectors: `/data/user_data/xoy/spkemb/phase6/` (`features/spk_phase6.py`).

## Main findings (test.clean; robust = same sign and supported in both training seeds and larger than the seed spread)

- HiFi-GAN leads on quality and speaker similarity (robust, also on test.other): UTMOS 4.17 / 4.16 (two seeds) against
  3.90 / 3.94 (Vocos) and 3.83 / 3.82 (DDSP); ground truth 4.21. On test.other HiFi-GAN equals ground-truth UTMOS (4.03).
- WER does not separate the vocoders in either seed or on test.other (about 2.5 percent on test.clean, 5.0 percent on
  test.other; ground truth 1.9 and 3.4 percent).
- Prosody failures differ (robust): Vocos has the largest F0 error (about 35 cents RMSE); DDSP has the largest voicing and
  loudness errors. The HiFi-GAN-DDSP F0 ordering and the Vocos-DDSP UTMOS ordering are not robust.
- F0 and loudness edits are followed closely; single EMA channel edits are not (gain about 0.1). Pairing the X and Y
  edits of a sensor does not help; time-varying edits and the direction within a sensor do (best about 0.57).
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

- `load_model("en+", linear_model_path=...)` keeps the checkpoint's own head (#11). Load a different head through
  `linear_model_state_dict`, or through a yaml whose `linear_model_path` is set, and assert the loaded weights.
- With `normalize: true`, every waveform is z-scored per utterance before WavLM, CREPE and the loudness convolution
  (`src/sparc/speech.py:75-76`), so the cached loudness does not change with gain. Gain augmentation needs an
  un-normalized loudness, cached ungained and multiplied by the gain.
- torchcrepe adds random dither to pitch (#13); seed numpy per utterance, one utterance per `encode` call.
- Batched `encode()` corrupts short utterances, and utterances under 9,239 samples at 24 kHz cannot be encoded (#12).
  Cache with batch size 1.
- Feature length for 24 kHz input is `T = floor(L24/480) - 1 + [L24 mod 480 >= 119]`; frame j of loudness, pitch and
  the WavLM CNN is centred at 24 kHz samples 480j, 480j + 120 and 480j + 300.
- The `en+` speaker embedding pools WavLM `hidden_states[6]`; `training/prepare_spk_raw.py` pools layer 0 with
  different periodicity settings (#13).
- The existing training module is not preemption-safe, counts both optimizers in `global_step`, and fails under plain
  DDP (#15). Its learning-rate schedule (#10) and MSD pooling (#9) also have bugs.

## Periodicity as a vocoder input (2026-10-29, `docs/vocoder_periodicity_analysis.md`)

- Adding periodicity to the original SPARC vocoder inputs (EMA + pitch + loudness) gives a small, consistent gain in
  domain (HiFi-GAN, 100k steps, one seed, LibriTTS-R test-clean): UTMOS +0.05, mel L1 -1.7%, voiced/unvoiced decision
  error 7.4% -> 4.5%, HNR error -0.17 dB; WER and speaker similarity not clearly changed. The gain is largest for voiced
  fricatives, vowels and voicing onsets/offsets, smallest for voiceless fricatives and silence. Out of domain (VCTK) only
  the voicing measures keep the gain.
- The trained model uses it mostly as a frame-precise voicing gate: replacing it by a constant is catastrophic (UTMOS
  -0.43), by a voiced/unvoiced mask nearly harmless, by a 9-frame moving average costs about twice the mask.
- Ablation run `experiment=periodicity_none_100k` (`models/hifigan_inputs.py`, `FeatureFrontend(drop_periodicity=True)`).
