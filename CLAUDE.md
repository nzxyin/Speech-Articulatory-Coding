# CLAUDE.md

Living summary of the project's current direction and findings. Changes go in `CHANGELOG.md`.

## Current direction (2026-10)

Compare three vocoders that synthesize 24 kHz speech from SPARC articulatory features: HiFi-GAN (the fork's
generator, retrained), a DDSP harmonic-plus-noise vocoder (after DDSP-Articulatory-Vocoder and RT-VC) and Vocos
adapted to articulatory input. All three use the same 15 features per 50 Hz frame (12 EMA channels, pitch, loudness,
periodicity), the same speaker conditioning (FiLM from a SPARC speaker embedding), the same data (filtered LibriTTS-R)
and the same training recipe and budget. The EMA comes from the per-utterance MNGU0 refit of the linear AAI head
(`linear_aai_mngu0_refit_peralign.pkl`), not from the head embedded in the `en+` checkpoint.

Status (2026-10-05): Phase 1 approved in full (decisions D1-D10). Phase 2 code is complete on branch
`vocoder-comparison` (contract: `docs/vocoders/INTERFACES.md`; 304 tests). All three vocoders pass an end-to-end GPU
smoke run (train, validate, preempt/resume, predict) and an overfit check on 4 utterances (aligned within 0.6 ms by
envelope cross-correlation; Whisper transcripts match the reference audio). The feature cache is complete: 358,332
utterances in all 7 splits (171 too short to encode), training statistics over 339,007 utterances (97.2 M frames).
All training is fp32; the bf16 precision pilot was dropped. Running on `preempt` (tracking issue #19): the HiFi-GAN and
DDSP primary runs (400k generator steps) and the Vocos option A and B pilots (25k steps). The Vocos primary run uses
the pilot winner (lower dev mel L1 at 25k; option B wins within 2 %).

## Measurements so far

- fp32 train step at the main shape (batch 16 × 64 frames, RTX 6000 Ada, `cudnn.benchmark` off): HiFi-GAN 304 ms,
  DDSP 236 ms, Vocos 214 ms (bf16: about 120-160 ms). A 400k-step run is about 24-34 GPU-hours in fp32.
- Feature extraction: about 144 s per audio hour on an L40 (batch size 1), about 23 GPU-hours for all 572 h.
- Overfit regime (4 utterances, 4k steps): final mel L1 HiFi-GAN 0.18, DDSP 0.29 (plateaus from about 2k steps),
  Vocos 0.08; HiFi-GAN output about 3 dB quieter than the reference at that point.
- "fp32" runs use cuDNN TF32 convolutions (PyTorch default); the feature extractor disables TF32.

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
