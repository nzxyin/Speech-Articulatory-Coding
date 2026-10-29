"""Phase 8.3 trial design: randomized, counterbalanced trial lists with attention checks (numpy only).

A trial = reference + two blinded candidates A and B; the listener rates both for similarity to the reference and for
naturalness (1-5) and chooses which is more similar to the reference (forced choice).

Test trials: one item = (reference, standard-code output, fitted-code output).
* Item balance: each listener gets ``n_vctk`` VCTK items (primary), ``n_globe`` GLOBE items (descriptive) and
  ``n_libritts`` LibriTTS-R control items; items are assigned greedily to the least-rated items (random ties), so every
  item is rated the same number of times (+-1) across listeners.
* Position balance: the fitted output is A about half the time for every item (across listeners) and for every
  listener (+-1 within each pool).
* Order: shuffled; attention checks are inserted at spread positions, never among the first two trials.
Attention checks (per listener, ``n_identical`` + ``n_mismatch``):
* identical: reference and the same standard output as A and B. Pass: both ratings differ by <= 1 point.
* mismatch: reference of a speaker, A/B = the standard output of the same speaker's item and the standard output of a
  speaker of the opposite gender. Pass: the same-speaker output is chosen and its similarity rating is higher.
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _balanced_pick(rng: np.random.Generator, ids: list[str], counts: dict[str, int], k: int) -> list[str]:
    """``k`` distinct ids with the lowest counts (random ties); increments the counts."""
    order = sorted(ids, key=lambda i: (counts[i], rng.random()))
    pick = order[:k]
    for i in pick:
        counts[i] += 1
    return pick


def _assign_positions(rng: np.random.Generator, picked: list[str], a_fit: dict[str, int], n_seen: dict[str, int]) -> dict[str, bool]:
    """Fit-is-A for each picked item: the item's less-used side (random ties), then flip the cheapest items until the
    listener's fit-as-A count is within one of half."""
    want: dict[str, bool] = {}
    for i in picked:
        d = 2 * a_fit[i] - n_seen[i]  # >0: fit has been A more often
        want[i] = bool(rng.random() < 0.5) if d == 0 else d < 0
    n_a = sum(want.values())
    half = len(picked) / 2
    while abs(n_a - half) > 0.5 + 1e-9:
        over = n_a > half
        cand = sorted((i for i in picked if want[i] == over), key=lambda i: (abs(2 * a_fit[i] - n_seen[i]), rng.random()))
        flip = cand[0]
        want[flip] = not over
        n_a += -1 if over else 1
    for i in picked:
        a_fit[i] += int(want[i])
        n_seen[i] += 1
    return want


def make_design(items: dict[str, dict[str, Any]], n_listeners: int, seed: int, n_vctk: int = 24, n_globe: int = 4, n_libritts: int = 2,
                n_identical: int = 2, n_mismatch: int = 2, prefix: str = "L") -> tuple[dict[str, list[dict]], list[dict]]:
    """``(public, rows)``: ``public[listener]`` = trial dicts for the page (``trial_id, ref, a, b``); ``rows`` = the
    private design (one dict per trial, with item, pool, speaker, arms and check type)."""
    rng = np.random.default_rng([seed, 8003])
    pools: dict[str, list[str]] = defaultdict(list)
    for i, r in sorted(items.items()):
        pools[r["pool"]].append(i)
    want = {"vctk": n_vctk, "globe": n_globe, "libritts": n_libritts}
    counts: dict[str, int] = defaultdict(int)
    a_fit: dict[str, int] = defaultdict(int)
    n_seen: dict[str, int] = defaultdict(int)
    chk_counts: dict[str, int] = defaultdict(int)
    public: dict[str, list[dict]] = {}
    rows: list[dict] = []
    check_pool = [i for i in pools["vctk"] + pools["libritts"]]
    for li in range(n_listeners):
        lid = f"{prefix}{li + 1:02d}"
        trials: list[dict] = []
        for pool, k in want.items():
            ids = pools.get(pool, [])
            if not ids or k == 0:
                continue
            picked = _balanced_pick(rng, ids, counts, min(k, len(ids)))
            pos = _assign_positions(rng, picked, a_fit, n_seen)
            for i in picked:
                f = items[i]["files"]
                a, b = ("fit", "std") if pos[i] else ("std", "fit")
                trials.append({"item": i, "a_arm": a, "b_arm": b, "ref": f["ref"], "a": f[a], "b": f[b], "check": ""})
        order = rng.permutation(len(trials))
        trials = [trials[k] for k in order]
        # attention checks
        checks: list[dict] = []
        for _ in range(n_identical):
            i = _balanced_pick(rng, check_pool, chk_counts, 1)[0]
            f = items[i]["files"]
            checks.append({"item": i, "a_arm": "std", "b_arm": "std", "ref": f["ref"], "a": f["std"], "b": f["std"], "check": "identical"})
        for _ in range(n_mismatch):
            i = _balanced_pick(rng, check_pool, chk_counts, 1)[0]
            other = [j for j in check_pool if items[j]["gender"] != items[i]["gender"] and items[j]["gender"] and items[i]["gender"]]
            if not other:
                other = [j for j in check_pool if items[j]["speaker"] != items[i]["speaker"]]
            j = other[int(rng.integers(len(other)))]
            f, g = items[i]["files"], items[j]["files"]
            if rng.random() < 0.5:
                checks.append({"item": i, "a_arm": "std", "b_arm": "std_other", "ref": f["ref"], "a": f["std"], "b": g["std"], "check": "mismatch", "other_item": j})
            else:
                checks.append({"item": i, "a_arm": "std_other", "b_arm": "std", "ref": f["ref"], "a": g["std"], "b": f["std"], "check": "mismatch", "other_item": j})
        n_tot = len(trials) + len(checks)
        slots = np.linspace(2, n_tot - 1, len(checks) + 2)[1:-1] if checks else []
        slots = sorted(int(round(s)) for s in slots)
        for s, c in zip(slots, rng.permutation(len(checks))):
            trials.insert(min(s, len(trials)), checks[c])
        public[lid] = []
        for p, t in enumerate(trials):
            tid = f"{lid}-{p + 1:02d}"
            public[lid].append({"trial_id": tid, "ref": t["ref"], "a": t["a"], "b": t["b"]})
            r = items[t["item"]]
            rows.append({"listener": lid, "trial_id": tid, "pos": p + 1, "item": t["item"], "pool": r["pool"], "speaker": r["speaker"], "a_arm": t["a_arm"],
                         "b_arm": t["b_arm"], "check": t["check"], "other_item": t.get("other_item", "")})
    return public, rows


def write_design(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
