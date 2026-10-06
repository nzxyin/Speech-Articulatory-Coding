"""Registry of the evaluated systems (``cfg.eval.systems``, contract section 1) and composition of vocoder configs.

A system is ``gt``, a ``reference`` (``vocos_mel`` and ``enplus16``, synthesized by the ``refs`` stage) or a ``vocoder``
(a trained run named by ``experiment`` and ``vocoder``, synthesized by ``synth`` with the existing predict path). Ablation
runs are added as further vocoder entries (``+eval.systems.hifigan_c256=...`` on the command line).
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from sparc.vocoders.data.dataset import CONDITIONS

KINDS = ("gt", "reference", "vocoder")
HEADS = ("refit", "shipped")
STAGES = (
    "gt",
    "synth",
    "refs",
    "signal",
    "utmos",
    "asr",
    "spk",
    "reextract",
    "prosody",
    "probes",
    "efficiency",
    "aggregate",
    "samples",
)
GLOBAL_STAGES = ("aggregate", "samples")  # not tied to one system or condition
# Kinds of system each per-system stage applies to; ``gt`` is its own reference, so the signal stage skips it.
STAGE_KINDS: dict[str, tuple[str, ...]] = {
    "gt": ("gt",),
    "synth": ("vocoder",),
    "refs": ("reference",),
    "signal": ("reference", "vocoder"),
    "utmos": KINDS,
    "asr": KINDS,
    "spk": KINDS,
    "reextract": KINDS,
    "prosody": KINDS,
    "probes": ("vocoder",),
    "efficiency": ("reference", "vocoder"),
}


@dataclass(frozen=True)
class SystemSpec:
    """One entry of ``cfg.eval.systems``.

    ``audio_from`` makes a system reuse another system's audio (``gt_shipped`` re-extracts the gt audio with the shipped
    head); ``stages`` restricts the stages that apply to it; ``ema_ref`` names the system whose re-extraction is the EMA
    reference of the prosody stage (the shipped-head gt re-extraction for ``enplus16``); ``ckpt`` and ``overrides`` are
    for vocoder systems (a fixed checkpoint, extra Hydra overrides such as a smaller generator in tests).
    """

    name: str
    kind: str
    sr: int
    conditions: tuple[str, ...]
    head: str = "refit"
    experiment: str | None = None
    vocoder: str | None = None
    ckpt: str | None = None
    overrides: tuple[str, ...] = ()
    audio_from: str | None = None
    stages: tuple[str, ...] | None = None
    ema_ref: str | None = None
    label: str = ""
    extra: dict = field(default_factory=dict, compare=False)

    def applies_to(self, stage: str) -> bool:
        if self.kind not in STAGE_KINDS.get(stage, ()):
            return False
        return self.stages is None or stage in self.stages


def _spec(name: str, node: DictConfig) -> SystemSpec:
    kind = str(node.kind)
    if kind not in KINDS:
        raise ValueError(f"system {name!r}: kind must be one of {KINDS}, got {kind!r}")
    conditions = tuple(str(c) for c in node.conditions)
    bad = [c for c in conditions if c not in CONDITIONS]
    if bad:
        raise ValueError(f"system {name!r}: unknown conditions {bad}")
    head = str(node.get("head", "refit"))
    if head not in HEADS:
        raise ValueError(f"system {name!r}: head must be one of {HEADS}, got {head!r}")
    if kind == "vocoder" and not (node.get("experiment") and node.get("vocoder")):
        raise ValueError(f"vocoder system {name!r} needs `experiment` and `vocoder`")
    stages = node.get("stages")
    if stages is not None:
        unknown = [s for s in stages if s not in STAGES]
        if unknown:
            raise ValueError(f"system {name!r}: unknown stages {unknown}")
    return SystemSpec(
        name=name,
        kind=kind,
        sr=int(node.sr),
        conditions=conditions,
        head=head,
        experiment=node.get("experiment"),
        vocoder=node.get("vocoder"),
        ckpt=node.get("ckpt"),
        overrides=tuple(str(o) for o in node.get("overrides", ()) or ()),
        audio_from=node.get("audio_from"),
        stages=None if stages is None else tuple(str(s) for s in stages),
        ema_ref=node.get("ema_ref"),
        label=str(node.get("label", name)),
    )


def load_systems(cfg: DictConfig) -> dict[str, SystemSpec]:
    """Registry of ``cfg.eval.systems`` in config order."""
    systems = {str(name): _spec(str(name), node) for name, node in cfg.eval.systems.items()}
    for spec in systems.values():
        for other in (spec.audio_from, spec.ema_ref):
            if other is not None and other not in systems:
                raise ValueError(f"system {spec.name!r} refers to unknown system {other!r}")
    return systems


def resolve_items(
    systems: dict[str, SystemSpec], stage: str, system: str = "all", condition: str = "all"
) -> list[tuple[SystemSpec, str]]:
    """``(system, condition)`` pairs a per-system stage runs for.

    ``system=all`` takes every system the stage applies to; an explicit name must apply. ``condition=all`` takes the
    system's own conditions; an explicit condition the system does not have is an error for an explicit system and
    skipped for ``system=all``.
    """
    if stage not in STAGE_KINDS:
        raise ValueError(f"{stage!r} is not a per-system stage; choose from {sorted(STAGE_KINDS)}")
    if condition != "all" and condition not in CONDITIONS:
        raise ValueError(f"condition must be all or one of {CONDITIONS}, got {condition!r}")
    if system == "all":
        chosen = [s for s in systems.values() if s.applies_to(stage)]
    else:
        if system not in systems:
            raise ValueError(f"unknown system {system!r}; configured: {list(systems)}")
        if not systems[system].applies_to(stage):
            raise ValueError(f"stage {stage!r} does not apply to system {system!r} ({systems[system].kind})")
        chosen = [systems[system]]
    items: list[tuple[SystemSpec, str]] = []
    for spec in chosen:
        if condition == "all":
            items.extend((spec, c) for c in spec.conditions)
        elif condition in spec.conditions:
            items.append((spec, condition))
        elif system != "all":
            raise ValueError(f"system {spec.name!r} has no condition {condition} (it has {list(spec.conditions)})")
    return items


# ----------------------------------------------------------------------------------------------- vocoder configs


@contextmanager
def hydra_cleared() -> Iterator[None]:
    """Clears Hydra's global instance for the block and puts the running application's instance back afterwards.

    ``hydra.compose`` and ``initialize_*`` refuse to run inside a running ``@hydra.main`` app (``GlobalHydra is already
    initialized``); code that composes configs itself (the vocoder loader of the probes and efficiency stages) runs
    inside this block.
    """
    from hydra.core.global_hydra import GlobalHydra

    outer = GlobalHydra.instance().hydra
    GlobalHydra.instance().clear()
    try:
        yield
    finally:
        GlobalHydra.instance().clear()
        GlobalHydra.instance().hydra = outer


def compose_config(config_name: str, overrides: Sequence[str]) -> DictConfig:
    """Composes a root config of the package (``sparc.conf``), also from inside a running ``@hydra.main`` app."""
    from hydra import compose, initialize_config_module

    with hydra_cleared():
        with initialize_config_module(config_module="sparc.conf", version_base=None):
            return compose(config_name=config_name, overrides=list(overrides))


def _quote(value: object) -> str:
    """Hydra override value as a quoted string (paths may contain characters of the override grammar)."""
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def vocoder_overrides(
    cfg: DictConfig,
    spec: SystemSpec,
    conditions: Sequence[str],
    output_dir: Path,
    accelerator: str,
    limit: int | None,
) -> list[str]:
    """Hydra overrides of ``vocoder_config`` for the synthesis of ``spec`` under ``conditions``."""
    split = str(cfg.eval.split)
    paths = cfg.paths
    overrides = [
        f"vocoder={spec.vocoder}",
        f"experiment={spec.experiment}",
        f"paths.cache_root={_quote(paths.cache_root)}",
        f"paths.runs_root={_quote(paths.runs_root)}",
        f"paths.libritts_r_raw={_quote(paths.libritts_r_raw)}",
        f"paths.refit_npz={_quote(paths.refit_npz)}",
        f"stats_path={_quote(cfg.stats_path)}",
        f"seed={int(cfg.eval.seed)}",
        f"data.predict_split={split}",
        f"data.predict_conditions=[{','.join(conditions)}]",
        f"data.predict_limit={int(limit) if limit else 'null'}",
        f"data.eval_num_workers={int(cfg.eval.predict.num_workers)}",
        f"data.eval_gain_db={float(cfg.data.eval_gain_db)}",
        "predict.skip_existing=true",
        f"predict.output_dir={_quote(output_dir)}",
        f"predict.precision={cfg.eval.predict.precision}",
        f"trainer.trainer.accelerator={accelerator}",
    ]
    if spec.ckpt:
        overrides.append(f"predict.ckpt={_quote(spec.ckpt)}")
    return overrides + list(spec.overrides)


def compose_vocoder_config(
    cfg: DictConfig, spec: SystemSpec, conditions: Sequence[str], output_dir: Path, accelerator: str, limit: int | None
) -> DictConfig:
    """``vocoder_config`` for ``spec`` with the predict settings of the evaluation (skip-existing on)."""
    return compose_config("vocoder_config", vocoder_overrides(cfg, spec, conditions, output_dir, accelerator, limit))


@dataclass(frozen=True)
class CheckpointInfo:
    """The checkpoint a vocoder system is evaluated at."""

    path: str
    g_step: int
    max_g_steps: int

    @property
    def final(self) -> bool:
        return self.g_step >= self.max_g_steps

    def as_dict(self) -> dict:
        return {"path": self.path, "g_step": self.g_step, "max_g_steps": self.max_g_steps, "final": self.final}


def resolve_checkpoint(vcfg: DictConfig) -> CheckpointInfo:
    """Checkpoint ``sparc-predict`` will load for ``vcfg``: ``predict.ckpt`` or the newest loadable ``step*.ckpt``."""
    import torch

    from sparc.vocoders.training.callbacks import CHECKPOINT_NAME, find_resume_checkpoint

    ckpt = vcfg.predict.ckpt
    if ckpt is None:
        found = find_resume_checkpoint(Path(vcfg.run_dir) / "ckpt")
        if found is None:
            raise FileNotFoundError(f"no loadable step*.ckpt in {Path(vcfg.run_dir) / 'ckpt'}")
        step, path = found
    else:
        path = Path(ckpt)
        counters = torch.load(path, map_location="cpu", weights_only=False)["vocoder_counters"]
        step = int(counters["g_step"])
        match = CHECKPOINT_NAME.fullmatch(path.name)
        if match and int(match.group(1)) != step:
            raise ValueError(f"{path} claims step {match.group(1)} but holds g_step {step}")
    return CheckpointInfo(str(path), int(step), int(vcfg.train.max_g_steps))


def synth_fingerprint(info: CheckpointInfo) -> dict:
    """What downstream result directories must agree on: which checkpoint produced the audio."""
    return {"ckpt": info.path, "g_step": info.g_step}


def system_config(cfg: DictConfig) -> dict:
    """Resolved ``eval.systems`` as plain containers (recorded in meta files)."""
    return OmegaConf.to_container(cfg.eval.systems, resolve=True)
