"""H13, multi-reference aggregation at inference, on the trained baseline encoders of the speaker-encoder screen.

No training: for each held-out utterance A the reference set is A and the ``K - 1`` utterances of the same speaker that
precede it in Phase 1 row order (cyclic), never the utterance whose targets are predicted (the next one, M1's partner).
Four fixed fusion rules combine the references (fixed before the first run):

* ``emb_mean``: mean of the per-reference 64-d embeddings (what ``SpeakerFFN.embed`` does);
* ``pool_mean``: mean of the pooled 1024-d vectors, then the FFN;
* ``pool_mass``: pooled vectors weighted by their periodicity mass (equal to pooling the concatenated frames), then FFN;
* ``emb_attn``: attention over reference embeddings with weights softmax(cos(e_k, mean of the other references) / 0.1);
  with fewer than 3 references this is the plain mean.

Only M1 (RMSE, PCC, speaker-level SE) is reported: EER and identification would share references between utterances.
The ridge probe is fitted on the fused embeddings of the non-test speakers with the same rule and K, as in M1; the
1 s / 2 s columns use the first 50 / 100 frames of every reference with the probe fitted on full-length fused embeddings.

Usage::

    python -m sparc.vocoders.spkscreen.multiref run --seed 0
    python -m sparc.vocoders.spkscreen.multiref report
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sparc.vocoders.spkscreen import cache as C
from sparc.vocoders.spkscreen import metrics as M
from sparc.vocoders.spkscreen.data import FrameSet, load_set
from sparc.vocoders.spkscreen.pooling import build, pool_weights

KS = (1, 3, 5)
FUSIONS = ("emb_mean", "pool_mean", "pool_mass", "emb_attn")
TAU = 0.1


@torch.no_grad()
def pooled_and_mass(enc, fs: FrameSet, max_frames: int | None = None, chunk: int = 64):
    order = np.argsort(fs.lens)
    P = np.zeros((fs.n, 1024), np.float32)
    mass = np.zeros(fs.n, np.float32)
    for i in range(0, fs.n, chunk):
        idx = order[i : i + chunk]
        b = fs.full(idx, enc.layers, False, max_frames)
        P[idx] = enc.pooled(b).float().cpu().numpy()
        mass[idx] = pool_weights(b.periodicity, b.mask).sum(1).float().cpu().numpy()
    return P, mass


def reference_sets(spk: np.ndarray, k: int) -> list[list[int]]:
    """For every utterance: itself and the preceding same-speaker utterances, excluding the M1 partner."""
    partner, order = M.partners(spk)
    out = [None] * len(spk)
    for ids in order.values():
        n = len(ids)
        for j, i in enumerate(ids):
            refs = [i]
            for t in range(1, k):
                r = ids[(j - t) % n]
                if r != partner[i] and r not in refs:
                    refs.append(r)
            out[i] = refs
    return out


@torch.no_grad()
def fuse(enc, P, mass, refs: list[list[int]], rule: str) -> np.ndarray:
    dev = next(enc.parameters()).device
    Pt, mt = torch.as_tensor(P, device=dev), torch.as_tensor(mass, device=dev)
    E = enc.ffn(Pt)
    out = []
    for r in refs:
        r = torch.as_tensor(r, device=dev)
        if rule == "emb_mean":
            out.append(E[r].mean(0))
        elif rule == "pool_mean":
            out.append(enc.ffn(Pt[r].mean(0, keepdim=True))[0])
        elif rule == "pool_mass":
            w = mt[r].clamp_min(1e-6)
            out.append(enc.ffn(((Pt[r] * w[:, None]).sum(0) / w.sum())[None])[0])
        elif rule == "emb_attn":
            e = E[r]
            if len(r) < 3:
                out.append(e.mean(0))
                continue
            en = F.normalize(e, dim=-1)
            others = (en.sum(0, keepdim=True) - en) / (len(r) - 1)
            a = torch.softmax((en * F.normalize(others, dim=-1)).sum(-1) / TAU, 0)
            out.append((a[:, None] * e).sum(0))
        else:
            raise ValueError(rule)
    return torch.stack(out).float().cpu().numpy()


def m1_scores(Xfit, Xtest_sets: dict, Y, spk, is_test) -> dict:
    """M1 with the probe fitted on ``Xfit`` (non-test rows) and applied to each test embedding set."""
    partner, _ = M.partners(spk)
    te = np.where(is_test)[0]
    test_spk = sorted(set(spk[te]))
    ysd = Y[~is_test].std(0) + 1e-9
    f = M.ridge_fit(np.asarray(Xfit, np.float64)[~is_test], Y[partner[~is_test]])
    res = {}
    for name, X in Xtest_sets.items():
        P = f(np.asarray(X, np.float64)[is_test])
        rmse, pcc = M.score(P, Y[partner[is_test]], ysd)
        E = (P - Y[partner[te]]) / ysd
        per = [np.sqrt((E[spk[te] == s] ** 2).mean(0)).mean() for s in test_spk]
        res[name] = {"m1_rmse": rmse, "m1_pcc": pcc, "m1_rmse_se": float(np.std(per, ddof=1) / np.sqrt(len(per)))}
    return res


def run(seed: int, root: Path = C.ROOT, device: str = "auto") -> dict:
    out = root / "multiref" / f"seed{seed}.json"
    if out.exists():
        return json.loads(out.read_text())
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    enc = build("baseline")
    enc.load_state_dict(torch.load(root / "runs" / "baseline" / f"seed{seed}" / "encoder.pt", map_location="cpu"))
    enc.to(dev).eval()
    fs = load_set("heldout", enc.layers, False, dev, root)
    spk, is_test, Y = fs.speakers, fs.is_test, fs.targets.astype(np.float64)
    full = pooled_and_mass(enc, fs)
    s1, s2 = pooled_and_mass(enc, fs, 50), pooled_and_mass(enc, fs, 100)
    res = {}
    for k in KS:
        refs = reference_sets(spk, k)
        for rule in (FUSIONS if k > 1 else ("emb_mean",)):
            X = fuse(enc, *full, refs, rule)
            sets = {"full": X, "1s": fuse(enc, *s1, refs, rule), "2s": fuse(enc, *s2, refs, rule)}
            name = "single" if k == 1 else f"K{k}_{rule}"
            res[name] = m1_scores(X, sets, Y, spk, is_test)
            res[name]["mean_refs"] = float(np.mean([len(r) for r in refs]))
            print(name, {t: round(v["m1_rmse"], 4) for t, v in res[name].items() if isinstance(v, dict)}, flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return res


def report(root: Path = C.ROOT) -> str:
    runs = {int(p.stem[4:]): json.loads(p.read_text()) for p in sorted((root / "multiref").glob("seed*.json"))}
    names = list(next(iter(runs.values())))
    lines = ["| fusion | refs | M1 full | M1 1 s refs | M1 2 s refs | d M1 full vs single (SE) | d M1 1 s vs single (SE) |",
             "|---|---|---|---|---|---|---|"]
    for n in names:
        cells = []
        for t in ("full", "1s", "2s"):
            v = np.array([runs[s][n][t]["m1_rmse"] for s in runs])
            cells.append(f"{v.mean():.4f} ({v.std(ddof=1):.4f})" if len(v) > 1 else f"{v[0]:.4f}")
        diffs = []
        for t in ("full", "1s"):
            d = np.array([runs[s]["single"][t]["m1_rmse"] - runs[s][n][t]["m1_rmse"] for s in runs])
            diffs.append(f"{d.mean():+.4f} ({d.std(ddof=1) / np.sqrt(len(d)):.4f})" if len(d) > 1 else f"{d.mean():+.4f}")
        lines.append(f"| {n} | {np.mean([runs[s][n]['mean_refs'] for s in runs]):.2f} | " + " | ".join(cells + diffs) + " |")
    lines += ["", "Cells: mean over baseline seeds (SD); d = single-reference M1 minus fused M1, positive = fusion better, "
              "SE = SD over seeds / sqrt(n)."]
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=["run", "report"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", type=Path, default=C.ROOT)
    a = p.parse_args()
    if a.cmd == "run":
        run(a.seed, a.root)
    else:
        text = report(a.root)
        (a.root / "multiref" / "report.md").write_text(text)
        print(text)


if __name__ == "__main__":
    main()
