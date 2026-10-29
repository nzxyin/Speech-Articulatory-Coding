"""Phase 8.3: fix the listening-test items and write the loudness-normalized, blind-named stimuli.

Items (fixed here, before any listening):

* ``vctk``     all own-speaker items of the VCTK test half from Phase 8.2 (the primary pool);
* ``globe``    all GLOBE pseudo-speaker items of Phase 8.2 (descriptive);
* ``libritts`` a few LibriTTS-R control items (one per speaker, seeded).

Each item has three stimuli: ``ref`` (a real recording of a DIFFERENT utterance of the speaker), ``std`` (output with
the standard speaker code) and ``fit`` (output with the code chosen by 8.2 on validation, the ``nat`` arm). ``std`` and
``fit`` are the saved Phase 8.2 outputs (same content utterance, same vocoder run, same renderer seed).

The reference is a recording that the 8.2 code fit did not use (a held-out reference of the speaker's plan list, not
in the first five references the code was fitted on, and not the content utterance). GLOBE pseudo-speakers have only
two references and the code used both, so their reference is one of the fitting references (flagged
``ref_in_fit``); GLOBE is descriptive.

Loudness: every stimulus (reference and outputs) is scaled by a single linear gain to the same integrated loudness,
ITU-R BS.1770-4 K-weighted, gated (``torchaudio.functional.loudness``), target -26 LUFS at 24 kHz, 16-bit PCM wav.
Gains are logged in ``loudness.csv``; the script fails if a normalized stimulus would clip (peak > 0.97).

Blinding: stimulus files are named by a salted hash, ``<audio>/<hash>.wav``. The mapping from hash to
(item, role) is in ``items.json`` (private). ``manifest.json`` (public) lists only the hashes and durations.

Run through ``p5_job.sh`` (CPU)::

    scripts/spkemb/p5_job.sh scripts/spkemb/listening/build_items.py --out /data/user_data/xoy/spkemb/phase8/listening
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

SALT = "p8-listening-v1"
TARGET_LUFS = -26.0
PEAK_LIMIT = 0.97
SR = 24000
OWN = "/data/user_data/xoy/spkemb/phase8/svfit/audio/own"
ROOT = "/data/user_data/xoy/spkemb/phase4"
POOLS = {"vctk": "vctk", "globe": "globe", "libritts": "libritts"}  # listening pool -> plan name
N_LIBRITTS_CONTROL = 6
FIT_ARM = "nat"


def blind_name(item_id: str, role: str) -> str:
    return hashlib.sha1(f"{SALT}|{item_id}|{role}".encode()).hexdigest()[:12]


def loudness_lufs(w: np.ndarray) -> float:
    import torch
    import torchaudio

    return float(torchaudio.functional.loudness(torch.from_numpy(w.astype(np.float32))[None], SR))


def normalize(w: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    before = loudness_lufs(w)
    gain_db = TARGET_LUFS - before
    for _ in range(8):  # the gating is level dependent near the absolute gate: refine until within 0.01 LU
        out = (w.astype(np.float64) * 10 ** (gain_db / 20)).astype(np.float32)
        err = loudness_lufs(out) - TARGET_LUFS
        if abs(err) < 0.01:
            break
        gain_db -= err
    peak = float(np.abs(out).max())
    if peak > PEAK_LIMIT:
        raise RuntimeError(f"normalized stimulus would clip (peak {peak:.3f}, gain {gain_db:.1f} dB)")
    return out, {"lufs_before": before, "gain_db": gain_db, "peak_after": peak, "lufs_after": loudness_lufs(out)}


def choose_reference(plan: dict, own: dict, store, code_ids: list[str], rng: np.random.Generator) -> tuple[str, bool]:
    """A recording of another utterance of the speaker: prefer references not used by the code fit, 2.5-10 s long."""
    refs = list(plan["refs"][own["speaker"]])
    held = [u for u in refs if u not in set(code_ids) and u != own["content_id"]]
    in_fit = not held
    cand = held or [u for u in refs if u != own["content_id"]]
    dur = {u: int(store.T[store.index_of(u)]) * 480 / SR for u in cand}
    good = [u for u in cand if 2.5 <= dur[u] <= 10.0] or sorted(cand, key=lambda u: abs(dur[u] - 5.0))[:1]
    return str(good[int(rng.integers(len(good)))]), in_fit


def main() -> None:
    import soundfile as sf

    from sparc.vocoders.eval.clone import build_items, own_variants
    from sparc.vocoders.eval.clone_pools import Pool
    from sparc.vocoders.eval.io import gt_audio

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--audio", default=OWN)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--gain-db", type=float, default=-3.0, help="evaluation gain of the ground-truth recordings (as the scoring)")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "stimuli").mkdir(parents=True, exist_ok=True)

    items: dict[str, dict] = {}
    files: dict[str, dict] = {}
    log_rows = []
    for pool, plan_name in POOLS.items():
        plan = json.loads((Path(args.root) / "plans" / f"{plan_name}.json").read_text())
        store = Pool(plan_name, args.root).store("l6")
        std_dir, fit_dir = Path(args.audio) / pool / "std", Path(args.audio) / pool / FIT_ARM
        ids = sorted(p.stem for p in std_dir.glob("*.wav"))
        owns = {o["own_id"]: o for o in plan["own"]}
        if pool == "libritts":  # control: one item per speaker, seeded
            by: dict[str, list[str]] = {}
            for i in ids:
                by.setdefault(owns[i]["speaker"], []).append(i)
            rng = np.random.default_rng([83, 1])
            spk = sorted(by)
            chosen = [spk[k] for k in rng.permutation(len(spk))[:N_LIBRITTS_CONTROL]]
            ids = sorted(by[s][0] for s in chosen)
        for own_id in ids:
            o = owns[own_id]
            item_id = f"{pool}_{own_id}"
            code_ids = build_items({**plan, "own": [o], "swaps": []}, own_variants(5), ["own_t3"])[0].code_ids
            rng = np.random.default_rng([83, 2, int(hashlib.sha1(item_id.encode()).hexdigest()[:8], 16)])
            ref_utt, in_fit = choose_reference(plan, o, store, list(code_ids), rng)
            stim = {
                "ref": gt_audio(store, store.index_of(ref_utt), args.gain_db),
                "std": sf.read(std_dir / f"{own_id}.wav", dtype="float32")[0],
                "fit": sf.read(fit_dir / f"{own_id}.wav", dtype="float32")[0],
            }
            gender = plan["gender"].get(o["speaker"]) if isinstance(plan.get("gender"), dict) else None
            rec = {"pool": pool, "own_id": own_id, "speaker": o["speaker"], "gender": gender, "content_id": o["content_id"],
                   "ref_utt": ref_utt, "ref_in_fit": in_fit, "fit_arm": FIT_ARM, "files": {}}
            for role, w in stim.items():
                w = np.asarray(w, dtype=np.float32).reshape(-1)
                wn, info = normalize(w)
                h = blind_name(item_id, role)
                sf.write(out / "stimuli" / f"{h}.wav", wn, SR, subtype="PCM_16")
                rec["files"][role] = h
                files[h] = {"dur_s": round(len(wn) / SR, 3)}
                log_rows.append({"item": item_id, "role": role, "file": h, "dur_s": round(len(wn) / SR, 3), **{k: round(v, 3) for k, v in info.items()}})
            items[item_id] = rec
            print(item_id, rec["speaker"], gender, ref_utt, "in_fit" if in_fit else "held", flush=True)

    key = {"salt": SALT, "target_lufs": TARGET_LUFS, "fit_arm": FIT_ARM, "items": items}
    (out / "items.json").write_text(json.dumps(key, indent=1))
    (out / "manifest.json").write_text(json.dumps({"audio_base": "stimuli/", "ext": ".wav", "target_lufs": TARGET_LUFS, "files": files}, indent=1))
    with open(out / "loudness.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(log_rows[0]))
        wr.writeheader()
        wr.writerows(log_rows)
    counts = {p: sum(1 for r in items.values() if r["pool"] == p) for p in POOLS}
    lu = [r["lufs_after"] for r in log_rows]
    print("items", counts, "speakers", {p: len({r["speaker"] for r in items.values() if r["pool"] == p}) for p in POOLS})
    print(f"lufs after: min {min(lu):.2f} max {max(lu):.2f}; ref_in_fit: {sum(r['ref_in_fit'] for r in items.values())}")


if __name__ == "__main__":
    main()
