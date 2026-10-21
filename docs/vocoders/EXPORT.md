# Exporting a vocoder for inference and to the Hugging Face Hub

`sparc-export` (module `sparc.cli.export_vocoder`, library `sparc.vocoders.export`) turns a training checkpoint into
a small inference export, checks it against the checkpoint, and uploads it only when asked.

## What an export contains

| file | content |
|---|---|
| `model.safetensors` | generator (`generator.*`) and speaker FFN (`speaker.*`) weights with their normalization buffers; no discriminators, optimizer states or counters (61 MB HiFi-GAN, 60 MB Vocos, 28 MB DDSP instead of about 690 MB) |
| `config.json` | format `sparc-articulatory-vocoder`, `format_version` 1; generator class and arguments; speaker FFN settings; input contract (feature channels, loudness and speaker-vector definitions, pinned SPARC en+ and WavLM-large snapshots, inversion-head sha256); training facts (experiment, seed, generator step); source checkpoint sha256 and fork commit; test-set metrics if given |
| `stats.json` | the training statistics the modules are built with (feature frontend and the speaker layer the model reads) |
| `README.md` | model card with Hub metadata (licence, tags, dataset), inputs, usage and metrics |
| `inversion/<name>.npz` | with `--include-inversion-head`: the refit linear inversion head of the training features |
| `verification.json` | exported against original synthesis on test utterances |

`load_exported(path or repo id)` rebuilds the generator and speaker FFN from `config.json` and loads the weights
strictly. Only `HiFiGANVocoder`, `DDSPVocoder` and `VocosVocoder` of this package are accepted as generator class, and
nested `_target_` entries are rejected, so a downloaded config cannot import other code. The buffers stored in the
weights must equal the values built from `stats.json`. `load_feature_extractor(head npz)` builds the training feature
extractor (it downloads the pinned snapshots and then runs offline), and `extraction_to_inputs` turns its output into
`features [1, 15, T]` (loudness channel = un-normalized loudness times a gain) and `spk_raw [1, 1024]`.

## Checks

1. The export is reloaded and every tensor is compared with the checkpoint (exact).
2. `--verify N` synthesizes N test.clean utterances (T1, evaluation gain) with the original training module and with the
   reloaded export under the same torch RNG seed; the largest absolute sample difference must not exceed `--tolerance`
   (default 1e-5). It was 0 for all three exports on the CPU (2026-10-21).
3. End to end (`phase3/export_e2e_check.py`, not in the repository): extracting two test.clean WAVs with
   `load_feature_extractor` and the cache's seed rule reproduces the cached inputs to within 8e-5 (cache on GPU,
   check on CPU), and the export gives the same audio as the training module on those inputs.

## Commands

Run on a compute node after `source ~/sparc-vocoders-work/env.sh` (checkpoints, statistics, cache and inversion head
are on /data). The Hub token comes from `HF_TOKEN` or the token saved by `hf auth login`.

```bash
T=$SV_ROOT/eval/tables/test.clean
# export and verify only (nothing leaves the machine)
.venv/bin/sparc-export --vocoder hifigan --experiment main_800k --out $SV_ROOT/exports/hifigan \
    --metrics-table $T/long/table.csv --metrics-system hifigan_800k --include-inversion-head --repo-id <user>/<name>
# also create the repository (private unless --public) and upload in one commit
.venv/bin/sparc-export ... --repo-id <user>/<name> --upload
```

The checkpoints recommended for export (2026-10-21, chosen on test.clean and confirmed by the dev-set mel L1):
HiFi-GAN `main_800k` step 800,000, Vocos `main_800k` step 800,000 and DDSP `main` (seed 0) step 400,000; their
metrics are in `$T/long/table.csv` (`hifigan_800k`, `vocos_800k`) and `$T/table.csv` (`ddsp`).
