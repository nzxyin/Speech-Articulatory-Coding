"""Write trial lists for the pilot (5 listeners) and the main test (``--n-listeners``).

    python scripts/spkemb/listening/make_trials.py --dir /data/user_data/xoy/spkemb/phase8/listening --name pilot --n-listeners 5 --seed 1
    python scripts/spkemb/listening/make_trials.py --dir ... --name main --n-listeners 24 --seed 2

Outputs in ``<dir>/trials/``: ``<name>.json`` (public: what the page loads; no arm labels), ``<name>_design.csv`` (private:
arms, checks, items; read by the analysis). See ``design.py`` for the design.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design import make_design, write_design  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-listeners", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-vctk", type=int, default=24)
    ap.add_argument("--n-globe", type=int, default=4)
    ap.add_argument("--n-libritts", type=int, default=2)
    ap.add_argument("--n-identical", type=int, default=2)
    ap.add_argument("--n-mismatch", type=int, default=2)
    args = ap.parse_args()
    d = Path(args.dir)
    items = json.loads((d / "items.json").read_text())["items"]
    public, rows = make_design(items, args.n_listeners, args.seed, args.n_vctk, args.n_globe, args.n_libritts, args.n_identical, args.n_mismatch)
    out = d / "trials"
    out.mkdir(exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps({"set": args.name, "seed": args.seed, "listeners": public}))
    write_design(rows, out / f"{args.name}_design.csv")
    n = sum(len(v) for v in public.values())
    print(f"{args.name}: {len(public)} listeners, {n} trials ({n // len(public)} each, {args.n_identical + args.n_mismatch} checks)")


if __name__ == "__main__":
    main()
