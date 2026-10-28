"""Phase 7 config check (no training): composes experiment p7_gspk, checks that it differs from p5_fv only in the
experiment name and the GLOBE split path, that the GLOBE rows equal the Phase 5 split's, runs the Phase 5 batch checks
and counts, for GLOBE targets, how often the reference is another utterance of the same real speaker.

    scripts/spkemb/p5_job.sh scripts/spkemb/p7_validate_configs.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import p5_validate_configs as p5
from sparc.vocoders.data.datamodule import VocoderDataModule
from sparc.vocoders.data.dataset import PackedStore

OLD = "/data/user_data/xoy/spkemb/phase5/packed/globe_f/globe.train"
NEW = "/data/user_data/xoy/spkemb/phase7/packed/globe_f_spk/globe.train"


def main() -> dict:
    cfg = p5.compose_cfg("p7_gspk")
    diff = p5.composed_diff("p5_fv", "p7_gspk")
    assert set(diff) == {"experiment_name", "data.train_splits"}, diff
    a, b = list(diff["data.train_splits"][0]), list(diff["data.train_splits"][1])
    assert [x for x in a if x != OLD] == [x for x in b if x != NEW] and OLD in a and NEW in b
    assert a.index(OLD) == b.index(NEW)
    old, new = pd.read_parquet(f"{OLD}/index.parquet"), pd.read_parquet(f"{NEW}/index.parquet")
    assert len(old) == len(new) and all(old[c].equals(new[c]) for c in old.columns if c != "speaker")
    out = p5.check("p7_gspk")
    # GLOBE targets: reference = another utterance of the same real speaker (when the speaker has another utterance)
    dm = VocoderDataModule(cfg)
    dirs = [dm._packed(s) for s in cfg.data.train_splits]
    store = PackedStore(dirs, cfg.speaker.layer)
    ds = dm.train_dataloader().dataset
    globe = [i for i in range(len(ds)) if str(store.ids[int(ds.items[i])]).startswith("train-")]
    rng = np.random.default_rng(1)
    n_cross = n_same_spk = n_self = n_single = 0
    per = pd.Series(store.speaker_codes).value_counts()
    for pos, i in enumerate(rng.choice(globe, 3000, replace=False)):
        s = ds[(0, pos, int(i))]
        u = int(s["utt_index"])
        if per[store.speaker_codes[u]] == 1:
            n_single += 1
            continue
        r = int(s["ref_index"])
        if r == u:
            n_self += 1  # p_cross draw kept the utterance itself
        else:
            n_cross += 1
            n_same_spk += int(store.speaker_codes[u] == store.speaker_codes[r])
            assert store.ids[r] != store.ids[u]
    assert n_cross == n_same_spk and n_cross > 0
    out["globe_reference_check"] = {"sampled_globe_targets": 3000, "singleton_speaker_targets": n_single, "self_reference_draws": n_self,
                                    "other_utterance_references": n_cross, "of_them_same_real_speaker": n_same_spk}
    out["composed_diff_vs_p5_fv"] = diff
    out["globe_speakers_old_new"] = [int(old["speaker"].nunique()), int(new["speaker"].nunique())]
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
