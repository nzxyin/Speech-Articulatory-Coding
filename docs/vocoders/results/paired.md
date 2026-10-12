# Paired comparisons of the three vocoders (test.clean)

Source: `tables/test.clean/paired.csv` (computed by `aggregate.py`, `paired_cell` and `bootstrap_paired`); reformatted by `scripts/paired_report.py` into `paired_vocoders.csv` (all rows used here, stratum `all`).

## How to read the tables

- **One training run per arm.** HiFi-GAN, DDSP and Vocos were each trained once (400k steps). Every difference below therefore mixes an architecture effect with run-to-run variance (seed, data order, preemption history), and this analysis cannot separate the two. The intervals cover the sampling of test speakers only, not the sampling of training runs. A supported difference means that the two trained models differ on this test set; it does not show that the architecture causes the difference.
- **Pairing.** Each row compares the two systems on the same utterances (all 4687 under T1, 4686 under T2 and T3, 39 respectively 38 speakers; the one utterance without a value is dropped for that metric). The difference is `a - b`. The interval is the speaker-cluster bootstrap CI95 (10000 resamples, seed 0): speakers are resampled with replacement and all their utterances kept. WER and CER are corpus-level ratios, so their differences are differences of corpus rates.
- **Speakers agreeing** is the fraction of speakers whose own mean difference has the sign of the overall difference.
- **Supported** (this analysis): the CI95 excludes 0 and at least 75 percent of the speakers agree. `CI only` marks rows where the CI excludes 0 but fewer than 75 percent of the speakers agree, so the difference is carried by a minority of speakers or by few large values. `no` means the CI includes 0. No correction for the 20 metrics, 3 pairs and 3 conditions was applied; roughly 180 comparisons are listed, and a few borderline results (CI edge within 0.001 of zero) should be read with that in mind. The criterion is a convention of this report, not a pre-registered test.
- **Size matters.** Many differences are supported but small, for example HiFi-GAN against Vocos on Sim spk (+0.008) or on EMA r (+0.002). Supported means that the sign is reliable across speakers, not that the difference is audible or useful. The tables give the raw difference; compare it with the scale of the metric (ground-truth UTMOS 4.21, WER 1.9 percent, ground-truth Sim spk 0.749).
- `better` names the system with the better value (higher is better for UTMOS, PESQ, Sim, F0 within 50 cents and EMA r; lower is better for the rest).
- Metrics that need a reference are computed against the ground truth of the same utterance; the prosody and EMA rows use the re-extracted features (CREPE dither disabled). The utterance 908_157963_000024_000000 has no re-extracted features and is missing from those rows.

## Summary

| metric group | what the paired evidence supports (identical in direction under T1, T2 and T3 unless stated) |
|---|---|
| Perceptual quality (UTMOS, PESQ) | HiFi-GAN is better than both DDSP (UTMOS +0.33, CI +0.28 to +0.40, all 39 speakers agree) and Vocos (+0.27, CI +0.24 to +0.30, 97 percent of speakers agree). DDSP against Vocos on UTMOS is not supported in any condition (T1 -0.065, CI -0.146 to +0.002, 64 percent agree). HiFi-GAN UTMOS (4.165) is 0.04 below the ground truth, Vocos 0.31 and DDSP 0.38. |
| Intelligibility (WER, CER) | No pair is supported. The largest corpus-WER difference is 0.14 percentage points (DDSP against Vocos, T2, CI only; 63 percent of speakers agree), and for no WER or CER pair do more than 65 percent of speakers agree in any condition. All three are about 0.6 to 0.7 percentage points above the ground truth (1.89 percent). |
| Speaker similarity (Sim same, Sim spk) | Supported ordering HiFi-GAN > Vocos > DDSP on both measures under T1. HiFi-GAN against Vocos is small (Sim same +0.013 under T1; under T2 and T3 +0.003 and +0.001, not supported, with the T2 interval only just excluding zero; Sim spk +0.008, +0.009, +0.004, supported by 76 to 81 percent of speakers). DDSP is clearly behind both (Sim same lower by 0.025 to 0.054 and Sim spk lower by 0.032 to 0.056, all supported, in every condition). |
| Prosody | F0: DDSP is best (F0 RMSE 18.2 cents), HiFi-GAN second (23.4), Vocos worst (35.0); all three pairs supported in all conditions (DDSP against Vocos -16.8 cents, CI -19.5 to -14.6, 100 percent agree). VDE and loudness reverse the order: HiFi-GAN is best, Vocos second, DDSP worst (loudness RMSE 1.49, 1.83 and 2.80 dB; VDE 0.039, 0.042 and 0.066), supported for all pairs under T1 (HiFi-GAN against Vocos on VDE: -0.003, supported under T1 and T2, only CI under T3). |
| Articulation consistency (EMA r, EMA RMSE) | HiFi-GAN > Vocos > DDSP, all supported, but the differences are small (EMA r 0.971, 0.969, 0.960; HiFi-GAN against Vocos +0.002). |
| Spectral (MCD, MR-STFT, mel L1, PESQ) | HiFi-GAN is better than DDSP on almost everything. Vocos has the lowest MCD (3.97 dB, 0.025 below HiFi-GAN, supported but tiny) and is better than HiFi-GAN in the 4-8 kHz and 8-12 kHz mel bands but worse in 0-4 kHz and in PESQ and MR-STFT. DDSP against Vocos on MR-STFT and PESQ is not supported under T1. |

## Reading across the three conditions

The conditions change the speaker vector only. The paired differences between vocoders hardly move: the HiFi-GAN advantage in UTMOS is +0.33, +0.33 and +0.32 against DDSP and +0.27 in all three conditions against Vocos; Vocos F0 RMSE is 11.3 to 11.6 cents worse than HiFi-GAN and 16.4 to 16.8 cents worse than DDSP in all three. Only borderline rows change status between conditions (Sim same HiFi-GAN against Vocos, VDE HiFi-GAN against Vocos), which is what the small effect sizes predict. T2 and T3 are therefore not a different test of the architectures; they mainly lower Sim same for all three vocoders (see `failures.md`, cases on T2 and T3).

## Tables

Rows are grouped by metric group and condition. `pair` is `a - b` as in `paired.csv`.
### Perception-relevant

**T1**

| metric | pair (a - b) | diff (CI95) | speakers agreeing | better | supported |
|---|---|---|---|---|---|
| UTMOS | hifigan - ddsp | +0.333 (+0.279, +0.398) | 1.00 | hifigan | yes |
| UTMOS | hifigan - vocos | +0.268 (+0.239, +0.296) | 0.97 | hifigan | yes |
| UTMOS | ddsp - vocos | -0.065 (-0.146, +0.002) | 0.64 | vocos | no |
| WER | hifigan - ddsp | -0.0007 (-0.0015, +0.0001) | 0.56 | hifigan | no |
| WER | hifigan - vocos | -0.0010 (-0.0016, -0.0003) | 0.56 | hifigan | CI only |
| WER | ddsp - vocos | -0.0003 (-0.0009, +0.0004) | 0.44 | ddsp | no |
| CER | hifigan - ddsp | -0.0004 (-0.0009, -0.0000) | 0.62 | hifigan | CI only |
| CER | hifigan - vocos | -0.0004 (-0.0008, -0.0001) | 0.59 | hifigan | CI only |
| CER | ddsp - vocos | -0.0000 (-0.0004, +0.0004) | 0.51 | ddsp | no |
| Sim same | hifigan - ddsp | +0.0544 (+0.0483, +0.0612) | 1.00 | hifigan | yes |
| Sim same | hifigan - vocos | +0.0130 (+0.0108, +0.0152) | 0.97 | hifigan | yes |
| Sim same | ddsp - vocos | -0.0414 (-0.0488, -0.0350) | 1.00 | vocos | yes |
| Sim spk | hifigan - ddsp | +0.0403 (+0.0326, +0.0486) | 1.00 | hifigan | yes |
| Sim spk | hifigan - vocos | +0.0083 (+0.0058, +0.0107) | 0.76 | hifigan | yes |
| Sim spk | ddsp - vocos | -0.0320 (-0.0403, -0.0246) | 1.00 | vocos | yes |

**T2**

| metric | pair (a - b) | diff (CI95) | speakers agreeing | better | supported |
|---|---|---|---|---|---|
| UTMOS | hifigan - ddsp | +0.328 (+0.272, +0.394) | 1.00 | hifigan | yes |
| UTMOS | hifigan - vocos | +0.271 (+0.242, +0.300) | 1.00 | hifigan | yes |
| UTMOS | ddsp - vocos | -0.057 (-0.136, +0.010) | 0.61 | vocos | no |
| WER | hifigan - ddsp | +0.0007 (+0.0001, +0.0014) | 0.66 | ddsp | CI only |
| WER | hifigan - vocos | -0.0007 (-0.0014, +0.0001) | 0.47 | hifigan | no |
| WER | ddsp - vocos | -0.0014 (-0.0022, -0.0006) | 0.63 | ddsp | CI only |
| CER | hifigan - ddsp | +0.0003 (-0.0001, +0.0007) | 0.66 | ddsp | no |
| CER | hifigan - vocos | -0.0002 (-0.0005, +0.0001) | 0.58 | hifigan | no |
| CER | ddsp - vocos | -0.0005 (-0.0010, -0.0001) | 0.61 | ddsp | CI only |
| Sim same | hifigan - ddsp | +0.0286 (+0.0223, +0.0356) | 0.97 | hifigan | yes |
| Sim same | hifigan - vocos | +0.0033 (+0.0012, +0.0055) | 0.66 | hifigan | CI only |
| Sim same | ddsp - vocos | -0.0252 (-0.0320, -0.0195) | 1.00 | vocos | yes |
| Sim spk | hifigan - ddsp | +0.0447 (+0.0368, +0.0532) | 1.00 | hifigan | yes |
| Sim spk | hifigan - vocos | +0.0093 (+0.0068, +0.0117) | 0.82 | hifigan | yes |
| Sim spk | ddsp - vocos | -0.0354 (-0.0438, -0.0281) | 0.97 | vocos | yes |

**T3**

| metric | pair (a - b) | diff (CI95) | speakers agreeing | better | supported |
|---|---|---|---|---|---|
| UTMOS | hifigan - ddsp | +0.323 (+0.265, +0.389) | 1.00 | hifigan | yes |
| UTMOS | hifigan - vocos | +0.272 (+0.243, +0.301) | 1.00 | hifigan | yes |
| UTMOS | ddsp - vocos | -0.051 (-0.128, +0.015) | 0.61 | vocos | no |
| WER | hifigan - ddsp | -0.0001 (-0.0009, +0.0007) | 0.45 | hifigan | no |
| WER | hifigan - vocos | -0.0006 (-0.0013, +0.0002) | 0.45 | hifigan | no |
| WER | ddsp - vocos | -0.0005 (-0.0012, +0.0003) | 0.63 | ddsp | no |
| CER | hifigan - ddsp | -0.0001 (-0.0005, +0.0003) | 0.45 | hifigan | no |
| CER | hifigan - vocos | -0.0003 (-0.0006, +0.0001) | 0.61 | hifigan | no |
| CER | ddsp - vocos | -0.0002 (-0.0006, +0.0003) | 0.58 | ddsp | no |
| Sim same | hifigan - ddsp | +0.0333 (+0.0267, +0.0408) | 0.97 | hifigan | yes |
| Sim same | hifigan - vocos | +0.0010 (-0.0012, +0.0030) | 0.53 | hifigan | no |
| Sim same | ddsp - vocos | -0.0323 (-0.0396, -0.0262) | 1.00 | vocos | yes |
| Sim spk | hifigan - ddsp | +0.0557 (+0.0471, +0.0648) | 1.00 | hifigan | yes |
| Sim spk | hifigan - vocos | +0.0041 (+0.0013, +0.0066) | 0.76 | hifigan | yes |
| Sim spk | ddsp - vocos | -0.0516 (-0.0605, -0.0430) | 1.00 | vocos | yes |

### Prosody

**T1**

| metric | pair (a - b) | diff (CI95) | speakers agreeing | better | supported |
|---|---|---|---|---|---|
| F0 RMSE (cents) | hifigan - ddsp | +5.18 (+4.15, +6.45) | 1.00 | ddsp | yes |
| F0 RMSE (cents) | hifigan - vocos | -11.61 (-13.22, -10.30) | 0.97 | hifigan | yes |
| F0 RMSE (cents) | ddsp - vocos | -16.79 (-19.50, -14.59) | 1.00 | ddsp | yes |
| F0 within 50c | hifigan - ddsp | -0.0330 (-0.0400, -0.0274) | 1.00 | ddsp | yes |
| F0 within 50c | hifigan - vocos | +0.0827 (+0.0713, +0.0957) | 0.97 | hifigan | yes |
| F0 within 50c | ddsp - vocos | +0.1157 (+0.0994, +0.1350) | 0.97 | ddsp | yes |
| VDE | hifigan - ddsp | -0.0274 (-0.0304, -0.0244) | 1.00 | hifigan | yes |
| VDE | hifigan - vocos | -0.0028 (-0.0036, -0.0020) | 0.87 | hifigan | yes |
| VDE | ddsp - vocos | +0.0246 (+0.0214, +0.0279) | 1.00 | vocos | yes |
| Loudness RMSE (dB) | hifigan - ddsp | -1.314 (-1.429, -1.214) | 1.00 | hifigan | yes |
| Loudness RMSE (dB) | hifigan - vocos | -0.342 (-0.364, -0.318) | 0.97 | hifigan | yes |
| Loudness RMSE (dB) | ddsp - vocos | +0.972 (+0.870, +1.088) | 1.00 | vocos | yes |

**T2**

| metric | pair (a - b) | diff (CI95) | speakers agreeing | better | supported |
|---|---|---|---|---|---|
| F0 RMSE (cents) | hifigan - ddsp | +5.21 (+4.22, +6.42) | 1.00 | ddsp | yes |
| F0 RMSE (cents) | hifigan - vocos | -11.43 (-12.90, -10.23) | 1.00 | hifigan | yes |
| F0 RMSE (cents) | ddsp - vocos | -16.40 (-18.88, -14.38) | 1.00 | ddsp | yes |
| F0 within 50c | hifigan - ddsp | -0.0336 (-0.0404, -0.0282) | 1.00 | ddsp | yes |
| F0 within 50c | hifigan - vocos | +0.0823 (+0.0709, +0.0959) | 1.00 | hifigan | yes |
| F0 within 50c | ddsp - vocos | +0.1158 (+0.0994, +0.1356) | 1.00 | ddsp | yes |
| VDE | hifigan - ddsp | -0.0261 (-0.0290, -0.0232) | 1.00 | hifigan | yes |
| VDE | hifigan - vocos | -0.0018 (-0.0026, -0.0011) | 0.79 | hifigan | yes |
| VDE | ddsp - vocos | +0.0243 (+0.0211, +0.0275) | 1.00 | vocos | yes |
| Loudness RMSE (dB) | hifigan - ddsp | -1.302 (-1.416, -1.200) | 1.00 | hifigan | yes |
| Loudness RMSE (dB) | hifigan - vocos | -0.326 (-0.348, -0.302) | 1.00 | hifigan | yes |
| Loudness RMSE (dB) | ddsp - vocos | +0.976 (+0.872, +1.093) | 1.00 | vocos | yes |

**T3**

| metric | pair (a - b) | diff (CI95) | speakers agreeing | better | supported |
|---|---|---|---|---|---|
| F0 RMSE (cents) | hifigan - ddsp | +5.40 (+4.38, +6.64) | 1.00 | ddsp | yes |
| F0 RMSE (cents) | hifigan - vocos | -11.29 (-12.77, -10.08) | 1.00 | hifigan | yes |
| F0 RMSE (cents) | ddsp - vocos | -16.44 (-18.98, -14.39) | 1.00 | ddsp | yes |
| F0 within 50c | hifigan - ddsp | -0.0346 (-0.0414, -0.0292) | 1.00 | ddsp | yes |
| F0 within 50c | hifigan - vocos | +0.0815 (+0.0701, +0.0949) | 1.00 | hifigan | yes |
| F0 within 50c | ddsp - vocos | +0.1160 (+0.0996, +0.1356) | 1.00 | ddsp | yes |
| VDE | hifigan - ddsp | -0.0259 (-0.0287, -0.0230) | 1.00 | hifigan | yes |
| VDE | hifigan - vocos | -0.0020 (-0.0029, -0.0011) | 0.68 | hifigan | CI only |
| VDE | ddsp - vocos | +0.0239 (+0.0207, +0.0271) | 1.00 | vocos | yes |
| Loudness RMSE (dB) | hifigan - ddsp | -1.285 (-1.408, -1.174) | 1.00 | hifigan | yes |
| Loudness RMSE (dB) | hifigan - vocos | -0.290 (-0.322, -0.258) | 1.00 | hifigan | yes |
| Loudness RMSE (dB) | ddsp - vocos | +0.995 (+0.890, +1.114) | 1.00 | vocos | yes |

### Articulation consistency

**T1**

| metric | pair (a - b) | diff (CI95) | speakers agreeing | better | supported |
|---|---|---|---|---|---|
| EMA r | hifigan - ddsp | +0.0108 (+0.0102, +0.0115) | 1.00 | hifigan | yes |
| EMA r | hifigan - vocos | +0.0024 (+0.0021, +0.0027) | 0.95 | hifigan | yes |
| EMA r | ddsp - vocos | -0.0084 (-0.0092, -0.0077) | 1.00 | vocos | yes |
| EMA RMSE (std) | hifigan - ddsp | -0.0342 (-0.0365, -0.0323) | 1.00 | hifigan | yes |
| EMA RMSE (std) | hifigan - vocos | -0.0096 (-0.0107, -0.0084) | 1.00 | hifigan | yes |
| EMA RMSE (std) | ddsp - vocos | +0.0246 (+0.0222, +0.0275) | 1.00 | vocos | yes |

**T2**

| metric | pair (a - b) | diff (CI95) | speakers agreeing | better | supported |
|---|---|---|---|---|---|
| EMA r | hifigan - ddsp | +0.0099 (+0.0093, +0.0106) | 1.00 | hifigan | yes |
| EMA r | hifigan - vocos | +0.0025 (+0.0022, +0.0029) | 0.97 | hifigan | yes |
| EMA r | ddsp - vocos | -0.0074 (-0.0083, -0.0067) | 1.00 | vocos | yes |
| EMA RMSE (std) | hifigan - ddsp | -0.0288 (-0.0314, -0.0266) | 1.00 | hifigan | yes |
| EMA RMSE (std) | hifigan - vocos | -0.0085 (-0.0095, -0.0074) | 0.95 | hifigan | yes |
| EMA RMSE (std) | ddsp - vocos | +0.0202 (+0.0176, +0.0234) | 0.97 | vocos | yes |

**T3**

| metric | pair (a - b) | diff (CI95) | speakers agreeing | better | supported |
|---|---|---|---|---|---|
| EMA r | hifigan - ddsp | +0.0101 (+0.0094, +0.0108) | 1.00 | hifigan | yes |
| EMA r | hifigan - vocos | +0.0025 (+0.0022, +0.0027) | 1.00 | hifigan | yes |
| EMA r | ddsp - vocos | -0.0076 (-0.0085, -0.0069) | 1.00 | vocos | yes |
| EMA RMSE (std) | hifigan - ddsp | -0.0295 (-0.0322, -0.0273) | 1.00 | hifigan | yes |
| EMA RMSE (std) | hifigan - vocos | -0.0084 (-0.0094, -0.0073) | 0.97 | hifigan | yes |
| EMA RMSE (std) | ddsp - vocos | +0.0211 (+0.0184, +0.0243) | 0.97 | vocos | yes |

### Spectral

**T1**

| metric | pair (a - b) | diff (CI95) | speakers agreeing | better | supported |
|---|---|---|---|---|---|
| PESQ | hifigan - ddsp | +0.177 (+0.163, +0.191) | 1.00 | hifigan | yes |
| PESQ | hifigan - vocos | +0.188 (+0.168, +0.206) | 1.00 | hifigan | yes |
| PESQ | ddsp - vocos | +0.011 (-0.011, +0.031) | 0.56 | ddsp | no |
| MCD (dB) | hifigan - ddsp | -0.200 (-0.211, -0.189) | 0.97 | hifigan | yes |
| MCD (dB) | hifigan - vocos | +0.025 (+0.017, +0.034) | 0.87 | vocos | yes |
| MCD (dB) | ddsp - vocos | +0.225 (+0.216, +0.235) | 0.97 | vocos | yes |
| MR-STFT | hifigan - ddsp | -0.055 (-0.058, -0.051) | 0.97 | hifigan | yes |
| MR-STFT | hifigan - vocos | -0.053 (-0.057, -0.049) | 0.97 | hifigan | yes |
| MR-STFT | ddsp - vocos | +0.001 (-0.003, +0.006) | 0.56 | vocos | no |
| Mel L1 | hifigan - ddsp | -0.027 (-0.028, -0.025) | 0.97 | hifigan | yes |
| Mel L1 | hifigan - vocos | -0.006 (-0.008, -0.005) | 0.92 | hifigan | yes |
| Mel L1 | ddsp - vocos | +0.021 (+0.019, +0.022) | 1.00 | vocos | yes |
| Mel L1 0-4k | hifigan - ddsp | -0.039 (-0.041, -0.036) | 1.00 | hifigan | yes |
| Mel L1 0-4k | hifigan - vocos | -0.014 (-0.016, -0.011) | 0.97 | hifigan | yes |
| Mel L1 0-4k | ddsp - vocos | +0.025 (+0.023, +0.027) | 1.00 | vocos | yes |
| Mel L1 4-8k | hifigan - ddsp | -0.009 (-0.010, -0.007) | 0.90 | hifigan | yes |
| Mel L1 4-8k | hifigan - vocos | +0.009 (+0.008, +0.010) | 1.00 | vocos | yes |
| Mel L1 4-8k | ddsp - vocos | +0.018 (+0.016, +0.019) | 0.95 | vocos | yes |
| Mel L1 8-12k | hifigan - ddsp | +0.004 (+0.003, +0.005) | 0.90 | ddsp | yes |
| Mel L1 8-12k | hifigan - vocos | +0.007 (+0.005, +0.008) | 0.97 | vocos | yes |
| Mel L1 8-12k | ddsp - vocos | +0.002 (+0.001, +0.003) | 0.79 | vocos | yes |

**T2**

| metric | pair (a - b) | diff (CI95) | speakers agreeing | better | supported |
|---|---|---|---|---|---|
| PESQ | hifigan - ddsp | +0.117 (+0.104, +0.132) | 0.97 | hifigan | yes |
| PESQ | hifigan - vocos | +0.151 (+0.135, +0.166) | 1.00 | hifigan | yes |
| PESQ | ddsp - vocos | +0.034 (+0.012, +0.055) | 0.66 | ddsp | CI only |
| MCD (dB) | hifigan - ddsp | -0.072 (-0.091, -0.054) | 0.84 | hifigan | yes |
| MCD (dB) | hifigan - vocos | +0.040 (+0.034, +0.047) | 0.97 | vocos | yes |
| MCD (dB) | ddsp - vocos | +0.112 (+0.094, +0.132) | 1.00 | vocos | yes |
| MR-STFT | hifigan - ddsp | -0.034 (-0.038, -0.030) | 1.00 | hifigan | yes |
| MR-STFT | hifigan - vocos | -0.045 (-0.049, -0.042) | 1.00 | hifigan | yes |
| MR-STFT | ddsp - vocos | -0.011 (-0.016, -0.006) | 0.76 | ddsp | yes |
| Mel L1 | hifigan - ddsp | -0.012 (-0.014, -0.010) | 0.95 | hifigan | yes |
| Mel L1 | hifigan - vocos | -0.003 (-0.005, -0.002) | 0.84 | hifigan | yes |
| Mel L1 | ddsp - vocos | +0.009 (+0.007, +0.011) | 0.89 | vocos | yes |
| Mel L1 0-4k | hifigan - ddsp | -0.023 (-0.026, -0.021) | 0.97 | hifigan | yes |
| Mel L1 0-4k | hifigan - vocos | -0.009 (-0.011, -0.007) | 0.95 | hifigan | yes |
| Mel L1 0-4k | ddsp - vocos | +0.014 (+0.011, +0.017) | 0.95 | vocos | yes |
| Mel L1 4-8k | hifigan - ddsp | +0.006 (+0.003, +0.009) | 0.82 | ddsp | yes |
| Mel L1 4-8k | hifigan - vocos | +0.009 (+0.008, +0.010) | 1.00 | vocos | yes |
| Mel L1 4-8k | ddsp - vocos | +0.003 (+0.001, +0.006) | 0.74 | vocos | CI only |
| Mel L1 8-12k | hifigan - ddsp | +0.014 (+0.012, +0.016) | 1.00 | ddsp | yes |
| Mel L1 8-12k | hifigan - vocos | +0.006 (+0.005, +0.008) | 0.97 | vocos | yes |
| Mel L1 8-12k | ddsp - vocos | -0.008 (-0.010, -0.006) | 0.92 | ddsp | yes |

**T3**

| metric | pair (a - b) | diff (CI95) | speakers agreeing | better | supported |
|---|---|---|---|---|---|
| PESQ | hifigan - ddsp | +0.122 (+0.105, +0.140) | 1.00 | hifigan | yes |
| PESQ | hifigan - vocos | +0.151 (+0.136, +0.165) | 1.00 | hifigan | yes |
| PESQ | ddsp - vocos | +0.029 (+0.006, +0.050) | 0.63 | ddsp | CI only |
| MCD (dB) | hifigan - ddsp | -0.107 (-0.128, -0.086) | 0.89 | hifigan | yes |
| MCD (dB) | hifigan - vocos | +0.020 (+0.007, +0.030) | 0.89 | vocos | yes |
| MCD (dB) | ddsp - vocos | +0.127 (+0.098, +0.154) | 0.95 | vocos | yes |
| MR-STFT | hifigan - ddsp | -0.038 (-0.043, -0.033) | 0.97 | hifigan | yes |
| MR-STFT | hifigan - vocos | -0.050 (-0.053, -0.046) | 1.00 | hifigan | yes |
| MR-STFT | ddsp - vocos | -0.011 (-0.016, -0.005) | 0.74 | ddsp | CI only |
| Mel L1 | hifigan - ddsp | -0.015 (-0.017, -0.012) | 0.97 | hifigan | yes |
| Mel L1 | hifigan - vocos | -0.004 (-0.006, -0.002) | 0.79 | hifigan | yes |
| Mel L1 | ddsp - vocos | +0.011 (+0.007, +0.013) | 0.95 | vocos | yes |
| Mel L1 0-4k | hifigan - ddsp | -0.026 (-0.028, -0.023) | 1.00 | hifigan | yes |
| Mel L1 0-4k | hifigan - vocos | -0.010 (-0.013, -0.008) | 0.95 | hifigan | yes |
| Mel L1 0-4k | ddsp - vocos | +0.015 (+0.012, +0.019) | 1.00 | vocos | yes |
| Mel L1 4-8k | hifigan - ddsp | +0.002 (-0.001, +0.005) | 0.58 | ddsp | no |
| Mel L1 4-8k | hifigan - vocos | +0.008 (+0.006, +0.010) | 0.97 | vocos | yes |
| Mel L1 4-8k | ddsp - vocos | +0.006 (+0.001, +0.010) | 0.79 | vocos | yes |
| Mel L1 8-12k | hifigan - ddsp | +0.013 (+0.011, +0.016) | 0.97 | ddsp | yes |
| Mel L1 8-12k | hifigan - vocos | +0.007 (+0.005, +0.009) | 0.89 | vocos | yes |
| Mel L1 8-12k | ddsp - vocos | -0.007 (-0.011, -0.003) | 0.68 | ddsp | CI only |

## Limits of this analysis

- Single training run per vocoder, as stated above; no seed variance is estimated.
- The bootstrap resamples speakers and treats the 39 test speakers as the population of interest; speaker 61 has one utterance, which enters only through its cluster weight.
- Sim same and Sim spk use the ECAPA model only; the WavLM-SV embeddings were not analysed here.
- The agreement fraction is computed over speakers with data; speakers with very few utterances (for example speakers 61, 1221, 672 with 1 to 9 utterances) count as much as speakers with more than 300, so the fraction is a robustness check on the sign, not a weighted share of utterances.
