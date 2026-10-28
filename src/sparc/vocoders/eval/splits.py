"""Validation / test speaker splits of the speaker-embedding evaluations (``docs/spkemb_validation_protocol.md``).

Rule: every choice among arms, variants or hyperparameters is made on the **validation** speakers; the **test** speakers
only report the selected configuration(s) and the baseline (a single pre-registered confirmatory comparison may use them).

Splits (speakers never move between splits; every pool or corpus below is assigned whole or by speaker):

* LibriTTS-R: dev-clean = validation, test-clean = test (cloning pools ``libritts_dev`` / ``libritts``; in the encoder screen
  the held-out speakers named ``libri_dev:*`` / ``libri_test:*``).
* GLOBE: dev split = validation, test split = test. GLOBE speakers are disjoint across splits (checked in ``build``); the
  split of a pool speaker is that of its utterances (``dev-*`` / ``test-*`` stems). Applies to the real-speaker pool
  ``globe_spk`` and to the pseudo-speaker pool ``globe``.
* VCTK: a seeded, gender-balanced half split. The 40 cloning-evaluation speakers are split first (10 + 10 per gender); every
  other VCTK speaker (screen held-out speakers outside the cloning subset) is split by the same rule among themselves
  (per gender: sorted ids, permuted with ``default_rng([seed, 31])``, first ``n // 2`` validation, the rest test).

``build`` writes the split file (default ``docs/spkemb_validation_splits.json``); ``restrict_items`` filters a per-output
cloning table (``items.parquet``) so that stored outputs are re-scored without re-synthesis; the screen uses
``screen_masks`` (see ``spkscreen/ood_report.py --split``).

    python -m sparc.vocoders.eval.splits build [--out F]
    python -m sparc.vocoders.eval.splits show [--splits-file F]
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS_STREAM = 31
SPLIT_NAMES = ("val", "test")
DEFAULT_FILE = Path(__file__).resolve().parents[4] / "docs" / "spkemb_validation_splits.json"
PHASE4 = Path("/data/user_data/xoy/spkemb/phase4")
VCTK_INFO = "/data/group_data/UTD-NAS/Databases/VCTK/VCTK-Corpus/speaker-info.txt"
# cloning pool -> (validation pool name, test pool name) is expressed by ``cloning[split]`` keys; the LibriTTS pools differ
POOL_OF_SPLIT = {"libritts": "test", "libritts_dev": "val"}


def halve(ids_by_gender: dict[str, Sequence[str]], seed: int = 0) -> dict[str, str]:
    """Seeded gender-balanced half split: per gender, sorted ids permuted, first ``n // 2`` -> ``val``, the rest ``test``."""
    out: dict[str, str] = {}
    for gender in sorted(ids_by_gender):
        ids = sorted(ids_by_gender[gender])
        order = np.random.default_rng([seed, SEEDS_STREAM]).permutation(len(ids))
        for rank, k in enumerate(order):
            out[ids[k]] = "val" if rank < len(ids) // 2 else "test"
    return out


def globe_speaker_splits(table: pd.DataFrame) -> dict[str, str]:
    """Pool speaker -> ``val`` / ``test`` from the ``dev-`` / ``test-`` stem prefix of its utterance ids. Raises if a speaker
    has utterances of both splits."""
    sp = table.assign(split=np.where(table["id"].str.split(":").str[-1].str.startswith("dev"), "val", "test"))
    n = sp.groupby("speaker")["split"].nunique()
    if (n > 1).any():
        raise ValueError(f"speakers spanning dev and test: {list(n[n > 1].index)[:5]}")
    return sp.groupby("speaker")["split"].first().to_dict()


def build(phase4: str | Path = PHASE4, screen_root: str | Path | None = None, seed: int = 0, vctk_info: str = VCTK_INFO) -> dict:
    """Split definitions from the stored plans / pool tables / encoder-screen manifests (nothing is modified)."""
    from sparc.vocoders.eval.clone_pools import parse_vctk_speaker_info
    from sparc.vocoders.spkscreen import cache as C

    phase4, screen_root = Path(phase4), Path(screen_root or C.ROOT)
    plans = {p: json.loads((phase4 / "plans" / f"{p}.json").read_text()) for p in ("libritts", "libritts_dev", "vctk", "globe", "globe_spk", "globe_spk_test")}
    info = parse_vctk_speaker_info(Path(vctk_info).read_text())
    clone_vctk = sorted(plans["vctk"]["speakers"])
    vctk_assign = halve({g: [s for s in clone_vctk if info[s][0] == g] for g in ("F", "M")}, seed)
    # screen held-out VCTK speakers outside the cloning subset: same rule among themselves
    man = pd.read_parquet(screen_root / "heldout" / "manifest.parquet")
    names = sorted(man["speaker"].unique())
    vctk_screen = sorted(n.split(":", 1)[1] for n in names if n.startswith("vctk:"))
    unassigned = [s for s in vctk_screen if s not in info]  # no gender in speaker-info.txt (p280): in neither split
    rest = [s for s in vctk_screen if s not in vctk_assign and s in info]
    vctk_assign.update(halve({g: [s for s in rest if info[s][0] == g] for g in ("F", "M")}, seed))

    globe_assign = {}
    for pool in ("globe", "globe_spk", "globe_spk_test"):
        tab = pd.read_parquet(phase4 / "pools" / pool / "table.parquet")
        globe_assign[pool] = globe_speaker_splits(tab)
    cloning: dict[str, dict] = {s: {} for s in SPLIT_NAMES}
    cloning["val"]["libritts_dev"] = sorted(plans["libritts_dev"]["speakers"])
    cloning["test"]["libritts"] = sorted(plans["libritts"]["speakers"])
    for s in SPLIT_NAMES:
        cloning[s]["vctk"] = sorted(x for x in clone_vctk if vctk_assign[x] == s)
        for pool in ("globe", "globe_spk", "globe_spk_test"):
            cloning[s][pool] = sorted(x for x in plans[pool]["speakers"] if globe_assign[pool][x] == s)
    screen: dict[str, dict] = {s: {"heldout": [], "globe": [], "globe_spk": []} for s in SPLIT_NAMES}
    te = set(man.loc[man["is_test"].astype(bool), "speaker"])
    for n in sorted(te):
        corp, sid = n.split(":", 1)
        if corp == "vctk" and sid not in vctk_assign:
            continue
        s = "val" if corp == "libri_dev" else "test" if corp == "libri_test" else vctk_assign[sid]
        screen[s]["heldout"].append(n)
    for pool in ("globe", "globe_spk"):
        gm = pd.read_parquet(screen_root / pool / "manifest.parquet")
        for n in sorted(gm["speaker"].unique()):
            screen[globe_assign[pool][n.split(":", 1)[1]]][pool].append(n)
    return {
        "version": 1, "seed": seed, "rule": __doc__.split("\n\n")[1].strip(),
        "primary_pool": {"val": "libritts_dev", "test": "libritts"},
        "cloning": cloning, "screen": screen,
        "notes": {"globe_spk_test": "real GLOBE test contributors with exactly 2 utterances of >= 3 s each (no test contributor has 3): relaxed "
                  "rule, 1 reference + 1 content utterance per speaker, swap items only (no own-speaker items), 400 speakers stratified by "
                  "gender and accent; the real-speaker GLOBE test pool of the cloning protocol (pool `globe_spk_test`, not part of the screen)",
                  "globe_test_fallback": "the pseudo-speaker pool `globe` has 30 test speakers (ECAPA-clustered, descriptive only)"},
        "vctk_all": {k: v for k, v in sorted(vctk_assign.items())}, "vctk_unassigned": unassigned,
    }


def load(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or DEFAULT_FILE).read_text())


def restrict_items(items: pd.DataFrame, splits: dict, split: str, require_source: bool = False) -> pd.DataFrame:
    """Rows of a per-output cloning table whose **target** speaker belongs to ``split`` in the row's pool; pools the split does
    not define (e.g. ``libritts`` for ``val``) are dropped. ``split="all"`` returns the table unchanged. ``require_source``
    also requires the content (source) speaker to be in the split (fewer swaps, no cross-split content).

    Per-output columns are unchanged (top-1 identification ranks over the full pool's speakers; ``accept`` uses the full pool's
    EER threshold), so a restricted summary is exactly the mean over the full-pool outputs of the selected target speakers."""
    if split == "all":
        return items
    members = {p: set(v) for p, v in splits["cloning"][split].items()}
    keep = np.zeros(len(items), dtype=bool)
    for pool, spk in members.items():
        m = (items["pool"] == pool).to_numpy() & items["target"].isin(spk).to_numpy()
        if require_source:
            m &= items["source"].isin(spk).to_numpy()
        keep |= m
    return items[keep].reset_index(drop=True)


def screen_masks(speakers: np.ndarray, splits: dict, split: str, key: str) -> np.ndarray:
    """Boolean mask over an array of screen speaker names (``key``: ``heldout`` / ``globe`` / ``globe_spk``)."""
    return np.isin(np.asarray(speakers).astype(str), splits["screen"][split][key])


def screen_scored(speakers: np.ndarray, splits: dict) -> dict[str, np.ndarray]:
    """Scored held-out speakers of the encoder screen per split (OOD screen, Amendment 2).

    LibriTTS-R: the screen lists (``libri_dev:*`` validation, ``libri_test:*`` test). VCTK: the 40 cloning-evaluation speakers
    (``cloning.vctk``, 20 per split), not the screen's own VCTK lists (half of those are in the vocoder's training data).
    Returns boolean masks over ``speakers`` for ``val`` and ``test``; the probe-fit rows are the complement of both."""
    sp = np.asarray(speakers).astype(str)
    out = {}
    for s in SPLIT_NAMES:
        names = {n for n in splits["screen"][s]["heldout"] if n.startswith("libri")}
        names |= {f"vctk:{x}" for x in splits["cloning"][s]["vctk"]}
        out[s] = np.isin(sp, sorted(names))
    return out


def probe_fit_mask(speakers: np.ndarray, splits: dict) -> np.ndarray:
    """Held-out rows whose speaker is in neither the validation nor the test split (the only rows probes are fitted on)."""
    m = screen_scored(speakers, splits)
    return ~(m["val"] | m["test"])


def counts(splits: dict) -> dict:
    return {
        "cloning": {s: {p: len(v) for p, v in splits["cloning"][s].items()} for s in SPLIT_NAMES},
        "screen": {s: {p: len(v) for p, v in splits["screen"][s].items()} for s in SPLIT_NAMES},
    }


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=str(DEFAULT_FILE))
    b.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("show")
    s.add_argument("--splits-file", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        sp = build(seed=a.seed)
        Path(a.out).write_text(json.dumps(sp, indent=1) + "\n")
        print(json.dumps(counts(sp), indent=1))
    else:
        print(json.dumps(counts(load(a.splits_file)), indent=1))


if __name__ == "__main__":
    main()
