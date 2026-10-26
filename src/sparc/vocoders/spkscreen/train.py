"""Trainer of the speaker-encoder screen: AAM-softmax speaker loss + auxiliary M1-style head, then the metric suite.

Usage (see docs/spkemb_encoder_screen.md)::

    python -m sparc.vocoders.spkscreen.train --variant baseline --seed 0
    python -m sparc.vocoders.spkscreen.train --variant en_plus     # shipped embedding reference, no training

Writes ``<root>/runs/<variant>/seed<s>/{result.json, embeddings.npz, encoder.pt}``; finished runs are skipped.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from sparc.vocoders.spkscreen import cache as C
from sparc.vocoders.spkscreen import metrics as M
from sparc.vocoders.spkscreen.data import FrameSet, load_set
from sparc.vocoders.spkscreen.pooling import PoolingModule, build

CFG = dict(steps=6000, batch=64, lr=1e-3, wd=0.01, warmup=300, clip=5.0, margin=0.2, scale=30.0, aux_weight=1.0,
           crop_min=50, crop_max=300, aux_hidden=256)


class AAMSoftmax(nn.Module):
    """Additive angular margin softmax (cosine classifier, margin ``m``, scale ``s``)."""

    def __init__(self, dim: int, n_classes: int, margin: float, scale: float):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, dim) * 0.1)
        self.margin, self.scale = margin, scale

    def forward(self, emb: torch.Tensor, label: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = F.linear(F.normalize(emb), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.margin)
        onehot = F.one_hot(label, cos.shape[1]).bool()
        logits = self.scale * torch.where(onehot, target, cos)
        return F.cross_entropy(logits, label), (cos.argmax(1) == label).float().mean()


def lr_at(step: int, cfg: dict) -> float:
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    p = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
    return cfg["lr"] * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


class Sampler:
    """One utterance per distinct speaker, random crop, and a different same-speaker utterance as the M1 partner."""

    def __init__(self, train: FrameSet, seed: int, cfg: dict):
        self.rng = np.random.default_rng(seed)
        self.cfg = cfg
        names = sorted(set(train.speakers))
        self.names = names
        self.label = np.array([names.index(s) for s in train.speakers])
        self.by_spk = [np.where(self.label == k)[0] for k in range(len(names))]
        self.lens = train.lens
        if min(len(b) for b in self.by_spk) < 2:
            raise ValueError("every training speaker needs at least two utterances")

    def draw(self, batch: int):
        spk = self.rng.choice(len(self.names), min(batch, len(self.names)), replace=False)
        utt, partner = [], []
        for k in spk:
            a, b = self.rng.choice(self.by_spk[k], 2, replace=False)
            utt.append(a)
            partner.append(b)
        utt, partner = np.array(utt), np.array(partner)
        length = np.minimum(self.rng.integers(self.cfg["crop_min"], self.cfg["crop_max"] + 1, len(utt)), self.lens[utt])
        start = (self.rng.random(len(utt)) * (self.lens[utt] - length + 1)).astype(np.int64)
        return utt, start, length, partner, spk


@torch.no_grad()
def embed_all(enc: PoolingModule, fs: FrameSet, max_frames: int | None = None, chunk: int = 64) -> np.ndarray:
    """Embeddings ``[N, 64]`` of whole utterances (or their first ``max_frames`` frames), dropout off."""
    enc.eval()
    order = np.argsort(fs.lens)
    out = np.zeros((fs.n, enc.out_dim), np.float32)
    for i in range(0, fs.n, chunk):
        idx = order[i : i + chunk]
        out[idx] = enc(fs.full(idx, enc.layers, enc.needs_mel, max_frames)).float().cpu().numpy()
    return out


def run_metrics(heldout: FrameSet, X, X1=None, X2=None) -> dict:
    return M.evaluate(X, heldout.targets, heldout.logf0, heldout.speakers, heldout.is_test, X1, X2)


def train_variant(variant: str, seed: int, root: Path = C.ROOT, device: str = "auto", cfg: dict | None = None,
                  variant_kwargs: dict | None = None, train: FrameSet | None = None, heldout: FrameSet | None = None,
                  tag: str | None = None, quiet: bool = False) -> dict:
    cfg = {**CFG, **(cfg or {})}
    out = root / "runs" / (tag or variant) / f"seed{seed}"
    if (out / "result.json").exists():
        print(f"{out} is finished")
        return json.loads((out / "result.json").read_text())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc = build(variant, **(variant_kwargs or {}))
    t0 = time.time()
    if train is None:
        train = load_set("train", enc.layers, enc.needs_mel, dev, root)
    if heldout is None:
        heldout = load_set("heldout", enc.layers, enc.needs_mel, dev, root)
    t_load = time.time() - t0
    enc.setup(train)
    enc.to(dev)
    sampler = Sampler(train, seed, cfg)
    tmean, tstd = train.targets.mean(0), train.targets.std(0) + 1e-6
    tz = torch.as_tensor((train.targets - tmean) / tstd, device=dev)
    aam = AAMSoftmax(enc.out_dim, len(sampler.names), cfg["margin"], cfg["scale"]).to(dev)
    aux_head = nn.Sequential(nn.Linear(enc.out_dim, cfg["aux_hidden"]), nn.GELU(), nn.Linear(cfg["aux_hidden"], tz.shape[1])).to(dev)
    params = [*enc.parameters(), *aam.parameters(), *aux_head.parameters()]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.999))
    log = []
    t1 = time.time()
    for step in range(cfg["steps"]):
        enc.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        utt, start, length, partner, spk = sampler.draw(cfg["batch"])
        emb = enc(train.batch(utt, start, length, enc.layers, enc.needs_mel))
        label = torch.as_tensor(spk, device=dev)
        l_aam, acc = aam(emb, label)
        l_aux = F.mse_loss(aux_head(emb), tz[torch.as_tensor(partner, device=dev)])
        extra = enc.extra_loss() if hasattr(enc, "extra_loss") else 0.0
        loss = l_aam + cfg["aux_weight"] * l_aux + extra
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg["clip"])
        opt.step()
        if step % 500 == 0 or step == cfg["steps"] - 1:
            rec = {"step": step, "aam": l_aam.item(), "aux": l_aux.item(), "acc": acc.item()}
            log.append(rec)
            if not quiet:
                print(rec, flush=True)
    t_train = time.time() - t1
    t2 = time.time()
    X = embed_all(enc, heldout)
    t_embed = time.time() - t2
    X1, X2 = embed_all(enc, heldout, 50), embed_all(enc, heldout, 100)
    res = run_metrics(heldout, X, X1, X2)
    res.update({"variant": tag or variant, "seed": seed, "encoder_params": enc.n_params(), "train_seconds": t_train,
                "load_seconds": t_load, "embed_ms_per_utt": 1000 * t_embed / heldout.n, "cfg": cfg,
                "variant_kwargs": variant_kwargs or {}, "log": log, "device": str(dev)})
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "embeddings.npz", full=X, s1=X1, s2=X2)
    torch.save(enc.state_dict(), out / "encoder.pt")
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("log", "cfg")}, indent=1))
    return res


def reference_en_plus(root: Path = C.ROOT) -> dict:
    """The shipped en+ 64-d embedding stored in the Phase 1 files, scored with the same suite."""
    out = root / "runs" / "en_plus"
    heldout = load_set("heldout", (), False, "cpu", root)
    res = run_metrics(heldout, heldout.en_plus)
    res.update({"variant": "en_plus", "encoder_params": None})
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--variant", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", type=Path, default=C.ROOT)
    p.add_argument("--steps", type=int, default=None, help="override the registered 6000 steps (smoke tests only)")
    p.add_argument("--kwargs", default="{}", help="JSON of constructor arguments of the variant")
    p.add_argument("--tag", default=None, help="run directory name (default: the variant name)")
    a = p.parse_args()
    if a.variant == "en_plus":
        reference_en_plus(a.root)
        return
    cfg = {"steps": a.steps} if a.steps else None
    train_variant(a.variant, a.seed, a.root, cfg=cfg, variant_kwargs=json.loads(a.kwargs), tag=a.tag)


if __name__ == "__main__":
    main()
