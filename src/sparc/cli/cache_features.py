"""``sparc-cache``: builds the feature cache of the vocoder comparison, one stage per call.

``sparc-cache stage=manifest|extract|pack|stats|validate``; ``extract`` takes ``shard_index`` and ``num_shards``
(defaults from ``SLURM_ARRAY_TASK_ID`` and ``SLURM_ARRAY_TASK_COUNT``). Configuration: ``conf/cache_config.yaml``.
"""

import json
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

STAGES = ("manifest", "extract", "pack", "stats", "validate")


def run(cfg: DictConfig) -> int:
    """Runs ``cfg.stage`` and returns the process exit code."""
    stage = cfg.stage
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    if stage == "manifest":
        from sparc.vocoders.features.manifest import run_manifest

        run_manifest(cfg)
    elif stage == "extract":
        from sparc.vocoders.features.cache_module import run_extract

        return run_extract(cfg)
    elif stage == "pack":
        from sparc.vocoders.features.pack import run_pack

        run_pack(cfg)
    elif stage == "stats":
        from sparc.vocoders.features.stats import run_stats

        run_stats(cfg)
    else:
        from sparc.vocoders.features.cache import validate_rows
        from sparc.vocoders.features.manifest import load_manifest, select_rows

        report = validate_rows(select_rows(load_manifest(cfg), cfg), cfg.cache.utt_dir, int(cfg.pack.num_threads))
        print(json.dumps(report, indent=2))
        return int(report["missing"] > 0 or report["invalid"] > 0)
    return 0


@hydra.main(version_base=None, config_path="../conf", config_name="cache_config")
def main(cfg: DictConfig) -> None:
    code = run(cfg)
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
