"""Loads a trained vocoder run for evaluation: composes the training config and restores the generator and speaker FFN.

``load_vocoder(experiment, vocoder, ckpt, device, overrides)`` composes ``vocoder_config`` with ``vocoder=<vocoder>``
and ``experiment=<experiment>`` (plus ``overrides``) from the package's conf directory, builds
:class:`~sparc.vocoders.training.module.VocoderGANModule` and loads the checkpoint into it. The checkpoint is ``ckpt``
when given (a ``.ckpt`` file, or a directory searched like a run's ``ckpt`` directory) and otherwise the newest
loadable ``step*.ckpt`` of ``${run_dir}/ckpt`` (``training.callbacks.find_resume_checkpoint``: newest by step number,
files that fail to load are skipped).

Provenance for the caller: the composed config gets a ``loaded`` block ``{ckpt, g_step, name}`` (``cfg.loaded``), and
the module gets the attributes ``checkpoint_path`` and ``checkpoint_step``. ``checkpoint_info(module)`` returns the
same facts as a plain dict. The module is returned in eval mode on ``device``; its discriminators are loaded as well
(they are part of the checkpoint) and are not used.
"""

from collections.abc import Sequence
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

import sparc
from sparc.vocoders.training.callbacks import find_resume_checkpoint
from sparc.vocoders.training.module import VocoderGANModule

CONF_DIR = str(Path(sparc.__file__).parent / "conf")
CONFIG_NAME = "vocoder_config"


def compose_vocoder_config(experiment: str | None, vocoder: str, overrides: Sequence[str] = ()) -> DictConfig:
    """``vocoder_config`` composed with ``vocoder=<vocoder>``, ``experiment=<experiment>`` and ``overrides``.

    ``experiment=None`` leaves the experiment group unset (the root config's own defaults apply). Safe to call inside
    a running Hydra application (``initialize_config_dir`` saves and restores the global state).
    """
    args = [f"vocoder={vocoder}", *([f"experiment={experiment}"] if experiment else []), *[str(o) for o in overrides]]
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return compose(config_name=CONFIG_NAME, overrides=args)


def resolve_checkpoint(cfg: DictConfig, ckpt: str | Path | None) -> tuple[int | None, Path]:
    """``(step or None, path)`` of the checkpoint to load; raises ``FileNotFoundError`` if there is none."""
    if ckpt is not None:
        path = Path(ckpt)
        if path.is_dir():
            found = find_resume_checkpoint(path)
            if found is None:
                raise FileNotFoundError(f"no loadable step*.ckpt in {path}")
            return found
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint {path} does not exist")
        return None, path
    ckpt_dir = Path(cfg.run_dir) / "ckpt"
    found = find_resume_checkpoint(ckpt_dir)
    if found is None:
        raise FileNotFoundError(f"no loadable step*.ckpt in {ckpt_dir}")
    return found


def load_vocoder(
    experiment: str,
    vocoder: str,
    ckpt: str | Path | None,
    device: torch.device | str,
    overrides: Sequence[str] = (),
) -> tuple[VocoderGANModule, DictConfig]:
    """Builds the module of ``experiment``/``vocoder`` and loads its checkpoint; returns ``(module, cfg)``.

    ``cfg.loaded.g_step`` is the generator step stored in the checkpoint (``vocoder_counters``) and
    ``cfg.loaded.ckpt`` its path; the module is in eval mode on ``device``.
    """
    cfg = compose_vocoder_config(experiment, vocoder, overrides)
    _, path = resolve_checkpoint(cfg, ckpt)
    state = torch.load(path, map_location="cpu", weights_only=False)
    module = VocoderGANModule(cfg)
    missing = module.load_state_dict(state["state_dict"], strict=True)
    if missing.missing_keys or missing.unexpected_keys:
        raise RuntimeError(f"{path}: state dict mismatch {missing}")
    g_step = int(state["vocoder_counters"]["g_step"])
    module.g_step = g_step
    module.d_step = int(state["vocoder_counters"].get("d_step", 0))
    module.checkpoint_path = str(path)
    module.checkpoint_step = g_step
    with open_dict(cfg):
        cfg.loaded = OmegaConf.create({"ckpt": str(path), "g_step": g_step, "name": f"{cfg.experiment_name}/{cfg.vocoder.name}"})
    return module.eval().to(torch.device(device)), cfg


def checkpoint_info(module: VocoderGANModule) -> dict:
    """``{"ckpt": path, "g_step": step}`` of a module returned by :func:`load_vocoder`."""
    return {"ckpt": getattr(module, "checkpoint_path", None), "g_step": getattr(module, "checkpoint_step", None)}
