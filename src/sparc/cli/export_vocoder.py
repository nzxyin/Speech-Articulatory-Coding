"""Export a trained vocoder for inference and optionally upload it to the Hugging Face Hub (``sparc-export``).

    sparc-export --vocoder hifigan --experiment main_800k --out $SV_ROOT/exports/hifigan \\
        --metrics-table $SV_ROOT/eval/tables/test.clean/long/table.csv --metrics-system hifigan_800k \\
        --include-inversion-head --repo-id <user>/<name>            # export and verify only
    sparc-export ... --repo-id <user>/<name> --upload [--public]    # also create the repository and upload

Steps:

1. compose the training config of the run (``vocoder_config`` with ``vocoder=`` and ``experiment=``);
2. take ``--ckpt`` or the newest loadable checkpoint of the run;
3. write the export (:func:`sparc.vocoders.export.export_vocoder`), which reloads it and compares every tensor with
   the checkpoint;
4. synthesize ``--verify`` test utterances (T1 speaker vectors, evaluation gain) with the original module and with the
   reloaded export under the same torch RNG seed and require the largest absolute sample difference to be at most
   ``--tolerance``; the result is written to ``verification.json`` in the export;
5. with ``--upload``: create the model repository if needed (private unless ``--public``) and upload the directory in
   one commit. Without ``--upload`` nothing leaves the machine.

Run on a compute node after ``source ~/sparc-vocoders-work/env.sh``: checkpoints, statistics, the feature cache and
the inversion head are on /data. The Hub token comes from ``HF_TOKEN`` or the token stored by ``hf auth login``.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from sparc.vocoders import export as ex

VOCODERS = ("hifigan", "ddsp", "vocos")
EXTRACTOR_KEYS_DROPPED = ("hf_hub_cache",)  # local paths, not part of the input contract


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="sparc-export", description=__doc__.split("\n\n")[0])
    p.add_argument("--vocoder", required=True, choices=VOCODERS)
    p.add_argument("--experiment", required=True, help="training experiment, for example main or main_800k")
    p.add_argument("--ckpt", default=None, help="checkpoint file or directory (default: newest loadable of the run)")
    p.add_argument("--out", required=True, help="export directory (must be empty unless --overwrite)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing export directory")
    p.add_argument("--metrics-table", default=None, help="evaluation table.csv whose row goes into the model card")
    p.add_argument("--metrics-system", default=None, help="system name of that row (default: the vocoder name)")
    p.add_argument("--include-inversion-head", action="store_true", help="copy the refit inversion head npz into the export")
    p.add_argument("--verify", type=int, default=8, help="number of test.clean utterances to compare (0: skip)")
    p.add_argument("--tolerance", type=float, default=1e-5, help="largest allowed absolute sample difference")
    p.add_argument("--device", default="cpu", help="device for the verification synthesis")
    p.add_argument("--repo-id", default=None, help="Hub model repository, user/name")
    p.add_argument("--license", default="mit", help="licence identifier for the model card metadata")
    p.add_argument("--upload", action="store_true", help="create the repository if needed and upload the export")
    p.add_argument("--public", action="store_true", help="create the repository public (default: private)")
    p.add_argument("--commit-message", default=None)
    p.add_argument("--override", action="append", default=[], help="extra Hydra override of the training config")
    args = p.parse_args(argv)
    if args.upload and not args.repo_id:
        p.error("--upload needs --repo-id")
    if args.metrics_system and not args.metrics_table:
        p.error("--metrics-system needs --metrics-table")
    return args


def cache_settings() -> tuple[dict, str, str]:
    """``(extractor block, inversion npz path, inversion sha256)`` of the feature-cache config."""
    from sparc.vocoders.eval.vocoder_loader import CONF_DIR

    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="cache_config", overrides=["stage=export"])
    extractor = {k: v for k, v in OmegaConf.to_container(cfg.extractor, resolve=True).items() if k not in EXTRACTOR_KEYS_DROPPED}
    return extractor, str(cfg.inversion.linear_npz), str(cfg.inversion.linear_sha256)


def verify(export_dir: Path, args: argparse.Namespace, cfg, ckpt: Path) -> dict:
    """Synthesis of ``args.verify`` test utterances by the original module and by the reloaded export."""
    from sparc.vocoders.data.dataset import FullUtteranceDataset
    from sparc.vocoders.eval.vocoder_loader import load_vocoder
    from sparc.vocoders.training.callbacks import fixed_torch_rng

    device = torch.device(args.device)
    original, _ = load_vocoder(args.experiment, args.vocoder, ckpt, device, args.override)
    exported = ex.load_exported(export_dir, device=device)
    packed = Path(cfg.paths.cache_root) / "packed" / cfg.data.predict_split
    ds = FullUtteranceDataset(packed, gain_db=float(cfg.data.eval_gain_db), condition="T1", speaker_layer=str(cfg.speaker.layer))
    picks = np.unique(np.linspace(0, len(ds) - 1, args.verify).round().astype(int)).tolist()
    rows, worst = [], 0.0
    with torch.no_grad():
        for i in picks:
            item = ds[i]
            features = item["features"][None].to(device)
            spk_raw = item["spk_raw"][None].to(device)
            with fixed_torch_rng(device):
                a = original.synthesize({"features": features, "spk_raw": spk_raw})
            with fixed_torch_rng(device):
                b = exported(features, spk_raw)
            diff = float((a - b).abs().max())
            worst = max(worst, diff)
            rows.append({"id": item["id"], "frames": int(features.shape[-1]), "max_abs_diff": diff,
                         "rms": float(a.pow(2).mean().sqrt())})
    return {"split": str(cfg.data.predict_split), "condition": "T1", "gain_db": float(cfg.data.eval_gain_db),
            "device": str(device), "tolerance": args.tolerance, "max_abs_diff": worst, "passed": worst <= args.tolerance,
            "utterances": rows}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from sparc.vocoders.eval.vocoder_loader import compose_vocoder_config, resolve_checkpoint
    from sparc.vocoders.features.extractor import fork_commit

    cfg = compose_vocoder_config(args.experiment, args.vocoder, args.override)
    step, ckpt = resolve_checkpoint(cfg, args.ckpt)
    print(f"checkpoint: {ckpt} (step {step})", flush=True)
    extractor, inv_npz, inv_sha = cache_settings()
    metrics = None
    if args.metrics_table:
        metrics = ex.metrics_from_table(args.metrics_table, args.metrics_system or args.vocoder)
    t0 = time.time()
    out = ex.export_vocoder(
        ckpt,
        cfg,
        args.out,
        extractor=extractor,
        inversion_npz=inv_npz if args.include_inversion_head else None,
        inversion_sha256=inv_sha,
        metrics=metrics,
        extra={"fork_commit": fork_commit(), "experiment_dir": f"{cfg.experiment_name}/{cfg.vocoder.name}"},
        overwrite=args.overwrite,
    )
    print(f"exported to {out} in {time.time() - t0:.1f} s; weights and buffers match the checkpoint", flush=True)
    if args.repo_id or args.license != "mit":
        ex.set_card_repo(out, args.repo_id, args.license)

    if args.verify > 0:
        report = verify(out, args, cfg, ckpt)
        (out / ex.VERIFICATION_FILE).write_text(json.dumps(report, indent=2) + "\n")
        print(f"verification on {len(report['utterances'])} utterances: max |difference| {report['max_abs_diff']:.3g} "
              f"(tolerance {args.tolerance:g})", flush=True)
        if not report["passed"]:
            print("verification failed; not uploading", file=sys.stderr)
            return 1
    for name, size in ex.list_files(out):
        print(f"  {name}  {size / 1e6:.2f} MB")

    if not args.upload:
        print(f"not uploaded (pass --upload{' --repo-id <user>/<name>' if not args.repo_id else ''} to upload)")
        return 0
    url = ex.upload_export(out, args.repo_id, private=not args.public, commit_message=args.commit_message)
    print(f"uploaded: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
