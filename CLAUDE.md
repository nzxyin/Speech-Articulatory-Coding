# CLAUDE.md

Living summary of the project's current direction and findings. Changes go in `CHANGELOG.md`.

## Current direction (2026-10)

Compare three vocoders that synthesize 24 kHz speech from SPARC articulatory features: HiFi-GAN (the fork's
generator, retrained), a DDSP harmonic-plus-noise vocoder (after DDSP-Articulatory-Vocoder and RT-VC) and Vocos
adapted to articulatory input. All three use the same 15 features per 50 Hz frame (12 EMA channels, pitch, loudness,
periodicity), the same speaker conditioning (FiLM from a SPARC speaker embedding), the same data (filtered LibriTTS-R)
and the same training recipe and budget. The EMA comes from the per-utterance MNGU0 refit of the linear AAI head
(`linear_aai_mngu0_refit_peralign.pkl`), not from the head embedded in the `en+` checkpoint.

Status: the Phase 1 investigation is complete and awaiting review. No training code has been written yet.

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
