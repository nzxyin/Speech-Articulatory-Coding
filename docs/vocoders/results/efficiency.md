# Efficiency

Source: `eval/efficiency/<system>.json` (50 test.clean utterances, 290.2 s of audio, batch 1, fp32, 5 warm-up calls). GPU is an NVIDIA L40S for every system. RTF is compute time divided by audio duration, so values below 1 are faster than real time. Medians and 95th percentiles are over the 50 utterances; each vocoder is one trained model, and the RTF values come from one timing session per device, so differences of a few percent are not meaningful.

## Parameters and real-time factor of the vocoder (includes the frontend and the speaker FFN)

| system | generator (M) | speaker FFN (M) | total (M) | RTF L40S med (p95) | RTF CPU 1 thread med (p95) | RTF CPU 8 threads med (p95) |
|---|---|---|---|---|---|---|
| HiFi-GAN | 14.11 | 1.12 | 15.22 | 0.0061 (0.0120) | 0.183 (0.191) | 0.0671 (0.0702) |
| DDSP | 5.92 | 1.12 | 7.04 | 0.0050 (0.0088) | 0.0363 (0.0381) | 0.0188 (0.0210) |
| Vocos | 13.78 | 1.12 | 14.90 | 0.0008 (0.0013) | 0.0266 (0.0274) | 0.0070 (0.0076) |
| Vocos mel (reference) | 13.53 | n/a | 13.53 | 0.0006 (0.0011) | 0.0252 (0.0258) | 0.0047 (0.0053) |
| SPARC en+ 16 kHz (reference) | 15.38 | 1.12 | 16.50 | 0.0062 (0.0121) | 0.162 (0.167) | 0.0457 (0.0482) |

The vocoder rows for the two references are the decode-only timing: `vocos_mel` decodes a mel spectrogram (its copy-synthesis timing, which adds the mel computation, is within 1 percent of decode on CPU 1 thread in the JSON); `enplus16` decodes SPARC features with the shipped HiFi-GAN at 16 kHz, whose total includes a 12,300-parameter linear head. The `enplus16` generator is 15.38 M parameters and its RTF is dominated by the 16 kHz generator; it is not directly comparable with the 24 kHz arms.

## Feature extractor (needed at inference by the articulatory and en+ pipelines)

The extractor is SPARC en+ (WavLM-large, CREPE full, refit inversion head), batch 1, no dither.

| device | RTF median | RTF p95 |
|---|---|---|
| L40S | 0.0463 | 0.0490 |
| CPU 1 thr | 3.47 | 3.52 |
| CPU 8 thr | 0.738 | 0.751 |

## End-to-end speech-to-speech RTF (extractor plus vocoder, sum of medians)

| system | L40S | CPU 1 thread | CPU 8 threads |
|---|---|---|---|
| HiFi-GAN | 0.0525 | 3.66 | 0.805 |
| DDSP | 0.0513 | 3.51 | 0.757 |
| Vocos | 0.0471 | 3.50 | 0.745 |
| SPARC en+ 16 kHz (own encode_decode, measured) | 0.0472 | 3.65 | 0.536 |

The sums assume the same extractor (refit head) in front of each vocoder; the extractor dominates on every device (8 to 60 times the vocoder on the GPU). The en+ row is the separately timed full pipeline of the reference system (shipped head, own run): it is within 10 percent of the sum (en+ decode plus extractor) on the GPU and with one CPU thread, but 31 percent lower with 8 CPU threads (0.54 versus 0.78), which is a timing difference between runs that these files cannot explain.

## Measured lookahead and past context

Method (EVALUATION.md section 7): add 0.5 training std to every channel of frame 200 of a 400-frame input; the first and last output samples that change by more than 1e-4 of the peak give the lookahead (how long before the frame the output already differs) and the past context (how long after the frame it still differs, that is the context the output depends on in the past). One source utterance for all systems. Analytic values from the architecture are given for comparison.

| system | measured lookahead (ms) | measured past (ms) | influence reaches end of window | analytic lookahead (ms) | analytic past (ms) |
|---|---|---|---|---|---|
| HiFi-GAN | 205 | 217 | no | 262 | 261 |
| DDSP | 929 | 3980 | yes | 1357 | 1342 (unbounded) |
| Vocos | 196 | 250 | no | 295 | 295 |
| Vocos mel (reference) | 192 | 235 | no | 304 | 304 |
| SPARC en+ 16 kHz (reference) | 189 | 219 | no | 263 | 263 |

DDSP per input channel group (measured, ms): all lookahead 929, past 3980 (reaches end); ema lookahead 1055, past 809; f0 lookahead 889, past 3980 (reaches end); loudness lookahead 1063, past 1050; periodicity lookahead 717, past 830.

## Streaming feasibility

- Vocos has the shortest bounded context: about 196 ms lookahead and 250 ms past, so a chunked streaming decoder needs roughly 0.2 s of future features, in addition to the chunk length. Its CPU RTF with one thread is 0.027, so it runs in real time on a single core.
- HiFi-GAN has a similar lookahead (about 205 ms), a slightly longer past context, and is 7 times slower than Vocos on one CPU thread (0.18 versus 0.027) but still comfortably faster than real time.
- DDSP is bounded by its convolutional context for the EMA, loudness and periodicity channels (lookahead 0.7 to 1.1 s, which is long), and an F0 change moves every later harmonic sample because the fundamental phase is a cumulative sum of F0 (the measured influence reaches the end of the 4 s window). Streaming DDSP therefore requires carrying the oscillator phase across chunks, which is possible but needs state, and a lookahead of about 0.9 to 1.1 s. The measured value is below the analytic 1.36 s bound.
- The lookahead of the vocoder is not the limiting term for the articulatory pipeline: the en+ feature extractor (WavLM-large, CREPE) is the dominant cost (RTF 0.046 on the L40S, 0.74 with 8 CPU threads, 3.5 with one thread) and itself has a non-causal context. The offline pipeline is faster than real time on the GPU and, with 8 CPU threads, about 0.75 to 0.8 of real time, but not with one thread (3.5 to 3.7). These are offline whole-utterance timings, not streaming measurements.
- Lookahead measurements are for one source utterance and one perturbation size (0.5 std); a threshold at 1e-4 of the peak means that smaller tails are not counted. They are not latency measurements of an implemented streaming system.

