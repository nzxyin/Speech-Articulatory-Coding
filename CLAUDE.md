# CLAUDE.md

Living summary of the project's current direction and findings. Changes go in `CHANGELOG.md`.

## Current direction (2026-10)

Compare three vocoders that synthesize 24 kHz speech from SPARC articulatory features: HiFi-GAN (the fork's
generator, retrained), a DDSP harmonic-plus-noise vocoder (after DDSP-Articulatory-Vocoder and RT-VC) and Vocos
adapted to articulatory input. All three use the same 15 features per 50 Hz frame (12 EMA channels, pitch, loudness,
periodicity), the same speaker conditioning (FiLM from a SPARC speaker embedding), the same data (filtered LibriTTS-R)
and the same training recipe and budget. The EMA comes from the per-utterance MNGU0 refit of the linear AAI head
(`linear_aai_mngu0_refit_peralign.pkl`), not from the head embedded in the `en+` checkpoint.

Status (2026-10-12): all three phases are done. The three primary runs (400k generator steps, fp32, one GPU each) and 12
ablation runs (100k steps) are trained, and the Phase 3 evaluation on test.clean is complete (code `sparc.vocoders.eval`,
contract `docs/vocoders/EVALUATION.md`). The write-up is `docs/vocoders/RESULTS.md`; tables are in
`docs/vocoders/results/`. Large outputs (checkpoints, audio, per-utterance results, samples, failure cases) are under
`/data/user_data/xoy/sparc-vocoders/` (`runs/`, `eval/`).

## Main findings (test.clean, single training run per arm; see RESULTS.md for paired CIs)

- HiFi-GAN leads on quality and speaker similarity: UTMOS 4.17 against 3.90 (Vocos) and 3.83 (DDSP), ground truth 4.21.
- WER does not separate the vocoders (2.45, 2.52, 2.55 percent; ground truth 1.89 percent).
- Prosody failures differ: DDSP follows F0 best (18 cents RMSE) but has the worst voicing and loudness; Vocos has the
  largest F0 error (35 cents, broad deviations, not octave errors); HiFi-GAN is best on voicing and loudness.
- F0 and loudness edits are followed closely; single-channel EMA edits are not (gain about 0.1), although natural EMA
  variation is reproduced with slope 1.0. The cause is open.
- Vocos is the cheapest (RTF 0.0008 on an L40S, 0.027 on one CPU thread) and has the shortest lookahead (196 ms).
- Ablations at 100k: Vocos option B beats A; MSD instead of MRD does not change the ranking; the native SPARC recipe
  raises UTMOS but worsens mel L1, PESQ and WER; at about 6 M parameters the HiFi-GAN lead over DDSP is not supported.

## Measurements so far

- fp32 train step at the main shape (batch 16 × 64 frames, RTX 6000 Ada, `cudnn.benchmark` off): HiFi-GAN 304 ms,
  DDSP 236 ms, Vocos 214 ms (bf16: about 120-160 ms). A 400k-step run is about 24-34 GPU-hours in fp32.
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
