"""Phase 3 evaluation: ``sparc-eval stage=<stage> system=<name|all> condition=<T1|T2|T3|all> eval.split=test.clean``.

Stages (docs/vocoders/EVALUATION.md, section 10): ``gt`` (materialize the reference audio), ``synth`` (vocoder audio
through ``sparc-predict``, skip-existing), ``refs`` (``vocos_mel`` and ``enplus16`` audio), ``signal``, ``utmos``,
``asr``, ``spk``, ``reextract``, ``prosody``, ``probes``, ``efficiency``, ``aggregate`` and ``samples``. Every stage is
resumable: finished chunks are skipped, and SIGUSR1 or SIGTERM stops the stage after the current chunk with exit code 75
(``scripts/slurm/eval.sh`` requeues on 75). ``eval.limit=N`` evaluates a fixed random subset of N utterances under
``eval_root/smoke_<N>/``, never mixed with the full results.
"""

import logging
import sys

import hydra
from omegaconf import DictConfig

from sparc.vocoders.eval.io import EXIT_STOPPED, StopFlag, StopRequested
from sparc.vocoders.eval.systems import STAGES

logger = logging.getLogger(__name__)


def run(cfg: DictConfig, stop: StopFlag | None = None) -> int:
    """Runs the configured stage; returns 0 when it completed and 75 when a stop signal interrupted it."""
    from sparc.vocoders.eval.stages import ALWAYS_RERUN, EvalContext, done_path, run_stage

    stage = str(cfg.stage)
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    flag = stop if stop is not None else StopFlag()
    with flag.installed():
        ctx = EvalContext(cfg, flag)
        marker = done_path(ctx.paths.root, ctx.split, stage, str(cfg.system), str(cfg.condition), ctx.device.type)
        if stage not in ALWAYS_RERUN and marker.is_file():
            logger.info("%s %s %s already done (%s)", stage, cfg.system, cfg.condition, marker)
            return 0
        try:
            run_stage(ctx, stage, str(cfg.system), str(cfg.condition))
        except StopRequested as stopped:
            logger.warning("stopped: %s", stopped)
            return EXIT_STOPPED
    return 0


@hydra.main(version_base=None, config_path="../conf", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    code = run(cfg)
    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
