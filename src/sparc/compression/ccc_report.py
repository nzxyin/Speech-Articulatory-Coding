"""Markdown rendering of ccc_analysis results.json (tables per arm and contrast, rankings, gain sweep, re-read, downstream)."""

import numpy as np

NM = {"rmse_z": "rmse_z", "rmse_centered": "centered rmse_z", "pcc": "PCC", "pcc_concat": "PCC concat",
      "ccc_utt": "CCC utt", "ccc_spk": "CCC spk", "ccc_spk_centered": "CCC spk centered", "ccc_utt_1s": "CCC utt >=1 s", "ccc_utt_3s": "CCC utt >=3 s (exploratory)"}


def f(x, d=4):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def ci(c):
    return f"[{c[0]:+.4f}, {c[1]:+.4f}]"


def reading(t):
    return "same ranking" if t >= 0.8 else ("partly" if t >= 0.5 else "different ranking")


def tau_table(q, groups):
    L = ["| group | n arms | CCC utt vs rmse_z | vs centered rmse_z | vs PCC | vs PCC concat | CCC utt vs CCC spk | CCC spk vs rmse_z | CCC spk vs centered rmse_z | CCC spk vs PCC concat |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for g in groups:
        def cell(m1, m2):
            v = q["tau"].get(f"{g}|{m1}|{m2}")
            return "-" if v is None else f"{v['tau']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}] {reading(v['tau'])}"
        n = next((v["n"] for k, v in q["tau"].items() if k.startswith(g + "|")), "-")
        L.append(f"| {g} | {n} | " + " | ".join(cell(a, b) for a, b in (
            ("ccc_utt", "rmse_z"), ("ccc_utt", "rmse_centered"), ("ccc_utt", "pcc"), ("ccc_utt", "pcc_concat"), ("ccc_utt", "ccc_spk"),
            ("ccc_spk", "rmse_z"), ("ccc_spk", "rmse_centered"), ("ccc_spk", "pcc_concat"))) + " |")
    return L


def rules(res):
    q2, q3, r1 = res["q2"], res["q3"], res["rule1"]
    out = {}
    out["1"] = (r1["higher"] >= r1["lower"]) and (r1["mde_spread_ccc_utt"] <= r1["mde_spread_pcc"])
    out["1_text"] = (f"standardized effect higher / lower / tie (within 10%) than PCC on {r1['higher']} / {r1['lower']} / {r1['tie']} contrasts ({r1['skipped_pcc_cannot_differ']} gain-only contrasts set aside: PCC cannot differ there); "
                     f"median detectable effect in units of the across-arm spread: CCC utt {r1['mde_spread_ccc_utt']:.2f}, PCC {r1['mde_spread_pcc']:.2f}, "
                     f"rmse_z {r1['mde_spread_rmse_z']:.2f}")
    out["2"] = (q3["cb_across_arm_sd"] > q3["cb_median_paired_se_q2"]) and q3["cb_share_q2_ci_excludes_0"] >= 0.5
    out["2_text"] = (f"C_b = CCC / PCC across arms: sd {q3['cb_across_arm_sd']:.4f} (range {q3['cb_range'][0]:.3f}-{q3['cb_range'][1]:.3f}); "
                     f"without gain-derived arms sd {q3['cb_across_arm_sd_plain']:.4f} (range {q3['cb_range_plain'][0]:.3f}-{q3['cb_range_plain'][1]:.3f}); "
                     f"median paired fold SE of C_b differences over the Q2 contrasts {q3['cb_median_paired_se_q2']:.4f}; "
                     f"interval of the C_b difference excludes 0 on {100 * q3['cb_share_q2_ci_excludes_0']:.0f}% of the contrasts")
    # rule 3
    q6 = res.get("q6")
    if q6:
        sp = q6["spearman"]
        rho = lambda m, k: sp.get(f"{m}|{k}", {}).get("rho", np.nan)  # noqa: E731
        ok_wer = all(rho("ccc_utt", k) >= rho("rmse_z", k) - 1e-9 for k in ("wer_usc", "wer_libri"))
        rev = [c for c, v in res["q5"].items() if v["ccc_utt"].startswith("reversed")]
        p1 = res["q5"]["P1"]["ccc_utt"]
        p1_changed = p1.startswith("now significant") or p1.startswith("reversed")
        bo = res["q4"]["blend"].get("W12 BiGRU->conv", {})
        o = bo.get("oracle_single_gain", {}).get("ccc_utt")
        out["3"] = bool(ok_wer and not rev and not p1_changed)
        out["3_text"] = (f"Spearman with USC-TIMIT WER over {sp['ccc_utt|wer_usc']['n']} systems: CCC utt {rho('ccc_utt', 'wer_usc'):+.3f}, rmse_z {rho('rmse_z', 'wer_usc'):+.3f}, "
                         f"PCC {rho('pcc', 'wer_usc'):+.3f}; LibriTTS-R WER ({sp['ccc_utt|wer_libri']['n']} systems): CCC utt {rho('ccc_utt', 'wer_libri'):+.3f}, "
                         f"rmse_z {rho('rmse_z', 'wer_libri'):+.3f}, PCC {rho('pcc', 'wer_libri'):+.3f} (all five metrics order the head systems identically, so this does not separate them); "
                         f"contrasts reversed under CCC utt: {rev if rev else 'none'}; P1 (blend vs shrunk LoRA) under CCC utt: {p1}"
                         + (f" - the plan's required statement applies: the conclusion depends on whether amplitude matters downstream, not that the blend is better "
                            f"(Q4: blend vs LoRA at its best single gain, CCC utt {o['mean']:+.4f}, {o['won']}/7 folds)" if p1_changed and o else "")
                         + ". The test that could show the CCC-optimal gain to be the better calibrated output (calibrated vs uncalibrated outputs of one head, vocoder retrained) does not exist")
    else:
        out["3"] = None
        out["3_text"] = "downstream tables missing"
    t = res["q1"]["all"]["tau"]["all|ccc_utt|ccc_spk"]["tau"]
    tp = res["q1"]["plain"]["tau"]["all|ccc_utt|ccc_spk"]["tau"]
    out["4"] = bool(t >= 0.8 and res["rule4_sensitivity_1s_all_same"])
    out["4_text"] = (f"Kendall tau CCC utt vs CCC spk {t:.2f} over all arms, {tp:.2f} over the arms without gain-derived arms; the 1 s cut removes no "
                     f"utterance (shortest {res['frames_per_utterance']['min']:.0f} frames = {res['frames_per_utterance']['min'] / 50:.2f} s), so that "
                     f"sensitivity run is identical by construction; exploratory 3 s cut changes the conclusion of: "
                     f"{res['sensitivity_3s_exploratory_changed'] if res['sensitivity_3s_exploratory_changed'] else 'no contrast'}")
    allpass = all(out[k] is True for k in "1234")
    out["verdict"] = "primary" if allpass else "companion"
    return out


def render(res):
    L = []
    A = res["arms"]
    plain = [n for n in A if "gain" not in n and "ORACLE" not in n]
    L += ["# Concordance correlation (CCC) as the AAI metric: results (plan: docs/ccc_metric_plan.md)\n",
          "All numbers: 7 leave-one-speaker-out folds (4 USC-TIMIT, 3 EMA_5EMO), `test_unseen` sentences, z units of the held-out speaker's training "
          "statistics, mean over seeds within a fold (3 seeds for trained arms, 1 for probes), then mean over folds. **Every arm was already seen on "
          "these folds (selected on the test folds); no untouched data exist.** Seed sd = sd over seeds of the fold-mean difference; fold sd = sd over the 7 "
          "folds of the paired difference. Plain names: concordance correlation (per utterance) = CCC utt, (per speaker) = CCC spk; the "
          "decomposition is correlation factor x scale factor x location factor (arm level, exact product, see plan section 6).\n"]
    av = res["availability"]
    L += ["## What is cached, and checks\n",
          f"- Re-scored arms: {len(A)}; max |difference| to `analysis_rigor/results.json` in rmse_z and PCC over the {len(A) - len(res['check_vs_results_json']['missing_in_results_json'])} shared arms: "
          f"{res['check_vs_results_json']['max_abs_diff']:.2e}; arm not in results.json (from tier2): {res['check_vs_results_json']['missing_in_results_json']}.",
          f"- Frames per test utterance (50 Hz): {res['frames_per_utterance']}.",
          f"- Per-depth probe predictions cached for depths {av['probe_depths_with_predictions']} only: **N1 (nested depth selection) is not assessable under CCC** "
          f"(no per-depth predictions; not recomputed).",
          f"- Gender conditioning (A0-A4): {av['gender_npz_files']} prediction files under `analysis_gender` (fold summaries only): **not assessable under CCC**; not rerun.",
          f"- `analysis_rigor/extra` (speaker adaptation, data curves, alignment): {av['extra_npz_files']} prediction files: **not assessable**, not approximated.",
          ""]
    # arm table
    L += ["## Every arm\n",
          "rmse_z, centered rmse_z, PCC (per utterance), PCC (concatenated), CCC utt, CCC utt restricted to utterances >= 1 s (identical here), CCC spk, CCC spk centered, "
          "prediction/target std ratio, mean |utterance mean offset| in target sd, and the utterance-level decomposition (correlation, scale, location factors; "
          "product = CCC utt) and plain mean scale (C_s) and location (C_l) factors.\n",
          "| arm | seeds | rmse_z | centered rmse_z | PCC | PCC concat | CCC utt | CCC utt >=1 s | CCC spk | CCC spk centered | std ratio | offset (sd) | correlation | scale | location | mean C_s | mean C_l |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in list(A):
        a, d = A[n], A[n]["decomp_utt"]
        L.append(f"| {n} | {a['seeds']} | {f(a['rmse_z'])} | {f(a['rmse_centered'])} | {f(a['pcc'])} | {f(a['pcc_concat'])} | {f(a['ccc_utt'])} | {f(a['ccc_utt_1s'])} | "
                 f"{f(a['ccc_spk'])} | {f(a['ccc_spk_centered'])} | {f(a['std_ratio'], 3)} | {f(a['offset_sd'], 3)} | {f(d['corr'])} | {f(d['scale_w'])} | "
                 f"{f(d['location_w'])} | {f(a['utt_cs'])} | {f(a['utt_cl'])} |")
    L += ["\nSpeaker-level decomposition (frames of a speaker concatenated):\n", "| arm | CCC spk | correlation | scale | location |", "|---|---|---|---|---|"]
    for n in A:
        d = A[n]["decomp_spk"]
        L.append(f"| {n} | {f(d['ccc'])} | {f(d['corr'])} | {f(d['scale_w'])} | {f(d['location_w'])} |")
    L += ["\nFold sd / seed sd of the arm means (rmse_z, PCC, CCC utt, CCC spk):\n", "| arm | fold sd rmse_z | PCC | CCC utt | CCC spk | seed sd rmse_z | PCC | CCC utt | CCC spk |", "|---|---|---|---|---|---|---|---|---|"]
    for n in A:
        a = A[n]
        L.append(f"| {n} | " + " | ".join(f(a[f"fold_sd_{m}"]) for m in ("rmse_z", "pcc", "ccc_utt", "ccc_spk")) + " | " +
                 " | ".join(f(np.sqrt(a[f"seed_var_{m}"] / max(a["folds"], 1)) if np.isfinite(a[f"seed_var_{m}"]) else np.nan) for m in ("rmse_z", "pcc", "ccc_utt", "ccc_spk")) + " |")
    L.append("\n(seed sd column: sd of a fold's seed values averaged over folds, divided by sqrt(7) = the seed contribution to the sd of the 7-fold mean; blank for 1-seed arms.)")
    vc = res["variance_components"]
    L += ["\nSeed vs fold variance (median over arms): " + "; ".join(
        f"{NM[m]}: seed sd {vc[m]['median_seed_sd']:.4f}, fold sd {vc[m]['median_fold_sd']:.4f}, seed/fold variance {vc[m]['median_seed_over_fold_var']:.3f}" for m in vc) + "\n"]
    # Q1
    L += ["## Q1. Ranking agreement (Kendall tau-b over arm means; fold-bootstrap 95% interval; reading rule: >= 0.8 same, 0.5-0.8 partly, < 0.5 different)\n"]
    for key in ("all", "plain"):
        q = res["q1"][key]
        L.append(f"### {q['label']} (n = {q['n_arms']})\n")
        L += tau_table(q, ["all", "probes", "frozen heads", "LoRA", "WavLM k12 (W12)"])
        L.append("")
        for m2 in ("pcc", "rmse_z"):
            fl = q["flips"][m2]
            L.append(f"Order flips between CCC utt and {NM[m2]}: {fl['n_flips']} of {fl['n_pairs']} arm pairs. Largest by CCC difference:\n")
            L += ["| arm a | arm b | d CCC utt (a-b) | d " + NM[m2] + " (a-b, sign so + = better) |", "|---|---|---|---|"]
            for r in fl["list"][:15]:
                L.append(f"| {r['a']} | {r['b']} | {r['d_ccc_utt']:+.4f} | {r['d_' + m2]:+.4f} |")
            L.append("")
    # Q2
    q2 = res["q2"]
    L += ["## Q2. Discriminability on the contrasts (+ = first arm better; folds won of 7; exact sign-flip p; bootstrap = hierarchical over folds, seeds, utterances, 1000 resamples)\n",
          "Standardized effect = |mean diff| / fold sd; MDE80 = detectable effect at 7 folds with 80% power. Pre-registered contrasts first.\n",
          "| id | first arm | second arm | metric | diff | bootstrap 95% | t 95% | won | p | fold sd | seed sd | MDE80 | std. effect |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = [c for c in ("P1", "P2", "P3", "T1", "T2") if c in q2] + [c for c in q2 if c not in ("P1", "P2", "P3", "T1", "T2")]
    for c in order:
        r = q2[c]
        for m in ("rmse_z", "pcc", "ccc_utt", "ccc_spk"):
            x = r[m]
            L.append(f"| {c if m == 'rmse_z' else ''} | {r['a'] if m == 'rmse_z' else ''} | {r['b'] if m == 'rmse_z' else ''} | {NM[m]} | {x['mean']:+.4f} | "
                     f"{ci(x['boot_ci']) if 'boot_ci' in x else '-'} | {ci(x['t_ci'])} | {x['won']}/{x['n_folds']} | {x['sign_flip_p']:.3f} | {f(x['fold_sd'])} | "
                     f"{f(x['seed_sd'])} | {f(x['mde80'])} | {f(x['std_effect'], 2)} |")
    L += ["\nSummary over the contrasts above (counts at p < 0.05 unadjusted; 'and interval' = also bootstrap interval excluding 0). Detectable effect in units of the across-arm sd of arm means (Q1, all arms):\n",
          "| metric | contrasts | p < 0.05 | p < 0.05 and interval clear | median std. effect | median MDE80 / arm spread | median fold sd | median seed sd |", "|---|---|---|---|---|---|---|---|"]
    for m, s in res["q2_summary"].items():
        L.append(f"| {NM[m]} | {s['n_contrasts']} | {s['n_sig_p05']} | {s['n_sig_p05_and_ci']} | {s['median_std_effect']:.2f} | {s['median_mde_in_arm_spread']:.2f} | {s['median_fold_sd']:.4f} | {f(s['median_seed_sd'])} |")
    L.append("\nExploratory paired fold tests (no bootstrap) for the companion variants on the pre-registered contrasts:\n")
    L += ["| id | metric | diff | t 95% | won | p |", "|---|---|---|---|---|---|"]
    for c in ("P1", "P2", "P3", "T1", "T2"):
        if c in q2:
            for m in ("rmse_centered", "ccc_spk_centered", "ccc_utt_1s", "ccc_utt_3s"):
                x = q2[c][m]
                L.append(f"| {c} | {NM.get(m, m)} | {x['mean']:+.4f} | {ci(x['t_ci'])} | {x['won']}/{x['n_folds']} | {x['sign_flip_p']:.3f} |")
    # Q3
    q3 = res["q3"]
    L += ["\n## Q3. What drives CCC differences\n",
          f"Across the {len(A)} arms (arm means): R^2 of CCC utt on PCC alone {q3['r2_ccc_on_pcc']:.3f}; on PCC + std ratio {q3['r2_ccc_on_pcc_stdratio']:.3f}; on PCC + std ratio + offset "
          f"{q3['r2_ccc_on_pcc_stdratio_offset']:.3f}. Without the gain-derived arms (n = {len([n for n in A if 'gain' not in n and 'ORACLE' not in n])}): PCC alone {q3['r2_ccc_on_pcc_plain_arms']:.3f}, "
          f"PCC + std ratio {q3['r2_ccc_on_pcc_stdratio_plain_arms']:.3f}.\n",
          "Share of the log CCC utt difference between the two arms of each contrast (log CCC = log correlation + log scale + log location; shares add to 1 when the total is not about 0):\n",
          "| id | log CCC diff | correlation | scale | location | PCC diff | CCC utt diff | C_b diff [t 95%] |", "|---|---|---|---|---|---|---|---|"]
    for c in order:
        r, s = q2[c], q2[c]["shares_utt"]
        sh = q3["share_of_ccc_diff_q2"][c]
        L.append(f"| {c} | {s['total']:+.4f} | {s['correlation']:+.4f} ({f(sh['correlation'], 2)}) | {s['scale']:+.4f} ({f(sh['scale'], 2)}) | {s['location']:+.4f} ({f(sh['location'], 2)}) | "
                 f"{r['pcc']['mean']:+.4f} | {r['ccc_utt']['mean']:+.4f} | {r['cb']['mean']:+.4f} {ci(r['cb']['t_ci'])} |")
    L.append("\n(In rows where the log CCC difference is near 0 the shares are unstable; read the three log parts.)\n")
    # Q4
    q4 = res["q4"]
    L += ["## Q4. Shrinkage sweep (prediction x gain around the speaker's zero; grid 0.5-1.5 step 0.02)\n",
          "Pooled optimum = grid optimum of the fold-and-seed mean curve (oracle, selected on the test folds); leave-fold-out gain = fitted on the other six folds (same seed); "
          "rmse_z closed-form gain = sum(p y) / sum(p p). A gain leaves PCC unchanged (range of the PCC curve checked in the last column).\n",
          "| arm | rmse_z gain (closed form) | rmse_z gain (grid) | CCC utt gain | CCC spk gain | LFO gain rmse_z / CCC utt / CCC spk (mean) | CCC utt loss at rmse_z gain | CCC spk loss at rmse_z gain | rmse_z loss at CCC utt gain | rmse_z loss at CCC spk gain | PCC range |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, r in q4["arms"].items():
        o, ls, g = r["optimal_gain_pooled"], r["losses"], r["lfo"]["gains"]
        L.append(f"| {a} | {o['rmse_z_closed_form_pooled']:.3f} | {o['rmse_z_grid']:.2f} | {o['ccc_utt']:.2f} | {o['ccc_spk']:.2f} | {g['rmse_z'][0]:.3f} / {g['ccc_utt'][0]:.3f} / {g['ccc_spk'][0]:.3f} | "
                 f"{ls['ccc_utt_loss_at_rmse_gain']:.4f} | {ls['ccc_spk_loss_at_rmse_gain']:.4f} | {ls['rmse_loss_at_ccc_utt_gain']:.4f} | {ls['rmse_loss_at_ccc_spk_gain']:.4f} | {o['pcc_range']:.1e} |")
    L += ["\nHeld-out score at each gain choice (mean over folds and seeds): gain 1 / leave-fold-out gain fitted for rmse_z / for CCC utt / for CCC spk, each reported as rmse_z, CCC utt, CCC spk:\n",
          "| arm | metric | gain 1 | LFO rmse_z gain | LFO CCC utt gain | LFO CCC spk gain | oracle best per fold |", "|---|---|---|---|---|---|---|"]
    for a, r in q4["arms"].items():
        for m in ("rmse_z", "ccc_utt", "ccc_spk"):
            L.append(f"| {a if m == 'rmse_z' else ''} | {NM[m]} | {r['at_gain_1'][m]:.4f} | {r['lfo']['scored_at_lfo_gain_rmse_z'][m]:.4f} | "
                     f"{r['lfo']['scored_at_lfo_gain_ccc_utt'][m]:.4f} | {r['lfo']['scored_at_lfo_gain_ccc_spk'][m]:.4f} | {r['oracle_best_value'][m]:.4f} |")
    L += ["\n### Blend against the calibrated LoRA (+ = blend better; paired over folds; t 95%, folds won, p)\n",
          "Rows: blend (gain 1) vs LoRA with the gain leave-fold-out-fitted for the metric in the column header (each metric's own gain); the common-gain rows use the rmse_z gain "
          "for every metric; oracle rows compare the blend at gain 1 / at its own best gain with LoRA at its best single gain (c: is the blend gain beyond any single gain?).\n",
          "| system | comparison | metric | diff | t 95% | won | p |", "|---|---|---|---|---|---|---|"]
    for sysn, o in q4["blend"].items():
        for lab, key, ms in (("blend vs LoRA, no gain", "raw", ("rmse_z", "pcc", "ccc_utt", "ccc_spk")),
                             ("blend vs LoRA x own LFO gain", "own_gain", ("rmse_z", "ccc_utt", "ccc_spk")),
                             ("blend vs LoRA x rmse_z LFO gain (common)", "common_gain_rmse_z", ("rmse_z", "pcc", "ccc_utt", "ccc_spk")),
                             ("blend vs LoRA x best single gain (oracle)", "oracle_single_gain", ("rmse_z", "ccc_utt", "ccc_spk")),
                             ("blend x best gain vs LoRA x best gain (oracle)", "blend_at_own_oracle_gain", ("rmse_z", "ccc_utt", "ccc_spk"))):
            for m in ms:
                x = o[key][m]
                L.append(f"| {sysn if (key == 'raw' and m == ms[0]) else ''} | {lab if m == ms[0] else ''} | {NM[m]} | {x['mean']:+.4f} | {ci(x['t_ci'])} | {x['won']}/{x['n_folds']} | {x['sign_flip_p']:.3f} |")
    # Q5
    q5 = res["q5"]
    L += ["\n## Q5. Key conclusions re-read under CCC (label relative to the contrast's pre-registered metric, or to rmse_z and PCC when none)\n",
          "| id | contrast | rmse_z diff (won) | PCC diff (won) | CCC utt diff (won) | CCC spk diff (won) | reference metric | CCC utt | CCC spk | 1 s cut same conclusion |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in order:
        r, v = q2[c], q5[c]
        cell = lambda m: f"{r[m]['mean']:+.4f} ({r[m]['won']}/7, p {r[m]['sign_flip_p']:.3f})"  # noqa: E731
        L.append(f"| {c} | {r['a']} vs {r['b']} | {cell('rmse_z')} | {cell('pcc')} | {cell('ccc_utt')} | {cell('ccc_spk')} | {v['reference']} | {v['ccc_utt']} | {v['ccc_spk']} | {v['ccc_utt_1s_same_conclusion']} |")
    L += ["\nN1 (nested depth selection), gender conditioning (A0-A4) and the `extra/` arms: not assessable (no cached per-utterance predictions).\n"]
    # Q6
    if "q6" in res:
        q6 = res["q6"]
        L += ["## Q6. Downstream validity (existing evidence only)\n",
              f"Per-system articulatory agreement of each system's EMA output with the shipped inverter's own output on {q6['n_items']} USC-TIMIT test utterances (z units of the shipped output's per-channel "
              "std; the shipped head is read from the cached pickle, its output scale reproduced by the cached head maps to within "
              f"{min(q6['head_check_std_ratio']):.3f}-{max(q6['head_check_std_ratio']):.3f}), against corpus-level USC-TIMIT / LibriTTS-R WER (Whisper large-v3), ECAPA similarity and UTMOS "
              "(UTMOS exists for the head systems only). LibriTTS-R states are not cached, so LibriTTS-R measures are related to the agreement measured on USC-TIMIT items. "
              "MNGU0 oracle system (`mngu0_ema`): not computable from cached arrays (no cached states for MNGU0 audio), so 8 of the 9 systems are scored.\n",
              "| system | rmse_z | centered rmse_z | PCC | CCC utt | CCC spk | std ratio | scale factor | location factor | offset (sd) | WER % USC | WER % Libri | ECAPA sim USC | ECAPA sim Libri | UTMOS USC | UTMOS Libri |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for s, a in q6["agreement"].items():
            d = q6["downstream"][s]
            L.append(f"| {s} | {f(a['rmse_z'], 3)} | {f(a['rmse_centered'], 3)} | {f(a['pcc'], 3)} | {f(a['ccc_utt'], 3)} | {f(a['ccc_spk'], 3)} | {f(a['std_ratio'], 3)} | {f(a['utt_scale_w'], 3)} | "
                     f"{f(a['utt_location_w'], 3)} | {f(a['offset_sd'], 3)} | {f(d['wer_usc'], 1)} | {f(d['wer_libri'], 1)} | {f(d['sim_usc'], 3)} | {f(d['sim_libri'], 3)} | {f(d['utmos_usc'], 2)} | {f(d['utmos_libri'], 2)} |")
        L += ["\nSpearman rank correlation (metric goodness vs downstream goodness; descriptive, n = 5-8 systems):\n",
              "| metric | WER USC (n) | WER Libri (n) | ECAPA USC | ECAPA Libri | UTMOS USC |", "|---|---|---|---|---|---|"]
        for m in ("rmse_z", "rmse_centered", "pcc", "ccc_utt", "ccc_spk"):
            def g(k):
                v = q6["spearman"].get(f"{m}|{k}")
                return "n/a" if v is None else f"{v['rho']:+.2f} ({v['n']})"
            L.append(f"| {NM[m]} | {g('wer_usc')} | {g('wer_libri')} | {g('sim_usc')} | {g('sim_libri')} | {g('utmos_usc')} |")
        L.append(f"\nHeads only (shipped + the five head systems, WER USC): " + ", ".join(f"{NM[k]} {v:+.2f}" for k, v in q6["spearman_heads_only_wer_usc"].items()) + ".\n")
        ag, dn = q6["agreement"], q6["downstream"]
        L += ["Vocoder controls (shipped output, amplitude-only shrink, circularly shifted residual): the shrunk output keeps WER (" +
              f"{dn['shipped']['wer_usc']:.1f} -> {dn['ship_shrink']['wer_usc']:.1f} % USC; {dn['shipped']['wer_libri']:.1f} -> {dn['ship_shrink']['wer_libri']:.1f} % Libri) while CCC utt falls from 1 to {ag['ship_shrink']['ccc_utt']:.3f} "
              f"(PCC stays {ag['ship_shrink']['pcc']:.3f}, rmse_z {ag['ship_shrink']['rmse_z']:.3f}); the shifted residual with the same error size as `all_lin` ({ag['ship_err']['rmse_z']:.3f} vs {ag['all_lin']['rmse_z']:.3f} rmse_z) has CCC utt {ag['ship_err']['ccc_utt']:.3f} "
              f"and WER {dn['ship_err']['wer_usc']:.1f} % against `all_lin` {ag['all_lin']['ccc_utt']:.3f} and {dn['all_lin']['wer_usc']:.1f} %. "
              "Missing (named, not run): retraining a vocoder on a multi-speaker inverter's outputs with arms that differ in scale (shrunk vs unshrunk) and in offset; a WER/perceptual test of gain-calibrated vs uncalibrated outputs of one head; more than nine systems.\n"]
    elif "q6_error" in res:
        L += ["## Q6\n", "Failed: see results.json (q6_error).\n"]
    # recommendation
    rl = rules(res)
    L += ["## Recommendation by the pre-registered rule (plan section 4)\n",
          f"1. Discriminability: **{'pass' if rl['1'] else 'fail'}** - {rl['1_text']}.",
          f"2. Decomposition informative: **{'pass' if rl['2'] else 'fail'}** - {rl['2_text']}.",
          f"3. No reversal of a downstream-supported conclusion, not worse than rmse_z against WER: **{('pass' if rl['3'] else 'fail') if rl['3'] is not None else 'n/a'}** - {rl['3_text']}.",
          f"4. Stability: **{'pass' if rl['4'] else 'fail'}** - {rl['4_text']}.",
          f"\nOutcome by the rule: **{rl['verdict']}**" + ("" if rl["verdict"] == "primary" else " (CCC as a companion with the decomposition; rmse_z and PCC stay primary). The downstream experiment of Q6(c) is what would allow promotion.") +
          " The user decides whether the metric convention changes; seven folds, one corpus family, selected on the test folds.\n"]
    return "\n".join(L) + "\n"
