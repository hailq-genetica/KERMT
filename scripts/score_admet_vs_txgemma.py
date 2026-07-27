#!/usr/bin/env python3
"""
Score a KERMT ADMET panel in the CORRECT TDC metric per task and lay it next to
TxGemma-27B-Predict (arXiv:2504.06196, Table 1).

Why this script is necessary
----------------------------
1. TDC uses a DIFFERENT metric per endpoint. Reporting one metric across the
   panel is wrong three ways:
     - CYP{2D6,3A4,2C9}_Veith are AUPRC, not AUROC (KERMT prints AUROC).
     - VDss/t_half/CL_hep are Spearman rho, not RMSE (KERMT prints RMSE in raw
       units -- 4.35, 10.4, 35.4 are uninterpretable and uncomparable).
   Half of a naively-reported table is in the wrong metric.

2. KERMT's predictions CSV (task/predict.py:write_prediction) is indexed by
   SMILES with one column per task_name and NO ground-truth columns. So we
   re-join predictions to the ground-truth test CSV on canonical SMILES. The
   join key must be canonicalized the SAME way on both sides or it silently
   drops rows.

3. The KERMT run used a UNION scaffold split, NOT TDC's official per-dataset
   splits. So these numbers are DIRECTIONAL vs TxGemma, not leaderboard-grade.
   That caveat is printed in every table. The only way to get comparable
   numbers is to rerun through tdc.benchmark_group.admet_group (single-task,
   5 seeds) -- see the note at the end.

4. AUC/AUPRC/Spearman on a 50-200 molecule test set carry big CIs. We
   bootstrap a 95% CI per task so the point estimate is never read naked, and
   flag any task whose test set (or minority count) is too small to trust.

Usage
-----
    pip install pandas numpy scipy scikit-learn rdkit
    python score_admet_vs_txgemma.py \
        --pred /runs/admet_cls/query_or_test_preds.csv \
        --truth /data/admet/cls/test.csv \
        --kind classification
    # and again for the regression panel with --kind regression
"""

import argparse
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

RDLogger.DisableLog("rdApp.*")
_CHOOSER = rdMolStandardize.LargestFragmentChooser()

# ---------------------------------------------------------------------------
# The authoritative per-task TDC metric, and TxGemma-27B-Predict Table 1 values
# (verified from arXiv:2504.06196 v1). tx = (metric, score, lo, hi, tdc_test_n).
# 'higher_better' is implied: rho/auroc/auprc up, mae/rmse down.
# ---------------------------------------------------------------------------
TDC_METRIC = {
    # classification panel
    "BBB": "auroc", "Pgp": "auroc", "HIA": "auroc", "F20": "auroc",
    "CYP2D6": "auprc", "CYP3A4": "auprc", "CYP2C9": "auprc",
    "hERG": "auroc", "AMES": "auroc", "DILI": "auroc",
    # regression panel
    "Caco2": "mae", "logD": "mae", "logS": "mae", "PPBR": "mae", "LD50": "mae",
    "VDss": "spearman", "t_half": "spearman", "CL_hep": "spearman",
}

TXGEMMA = {  # metric, score, lo, hi, tdc_test_n
    "BBB":   ("auroc", 0.908, 0.872, 0.938, 406),
    "Pgp":   ("auroc", 0.937, 0.904, 0.964, 245),
    "HIA":   ("auroc", 0.988, 0.972, 0.999, 117),
    "F20":   ("auroc", 0.694, 0.575, 0.801, 384),
    "CYP2D6":("auprc", 0.683, 0.639, 0.726, 2626),
    "CYP3A4":("auprc", 0.854, 0.836, 0.872, 2467),
    "CYP2C9":("auprc", 0.798, 0.767, 0.826, 2419),
    "hERG":  ("auroc", 0.885, 0.813, 0.946, 132),
    "AMES":  ("auroc", 0.816, 0.795, 0.838, 1457),
    "DILI":  ("auroc", 0.886, 0.810, 0.947, 96),
    "Caco2": ("mae", 0.401, 0.358, 0.449, 182),
    "logD":  ("mae", 0.538, 0.507, 0.570, 840),   # Lipophilicity AstraZeneca
    "logS":  ("mae", 0.907, 0.870, 0.948, 1996),  # Solubility AqSolDB
    "PPBR":  ("mae", 9.048, 8.141, 10.111, 559),
    "LD50":  ("mae", 0.627, 0.597, 0.660, 1478),
    "VDss":  ("spearman", 0.559, 0.457, 0.655, 226),
    "t_half":("spearman", 0.458, 0.306, 0.594, 135),
    "CL_hep":("spearman", 0.260, 0.129, 0.384, 243),  # hardest reg task in ADMET
}


def canonical(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None
    m = _CHOOSER.choose(m)
    if m is None or m.GetNumHeavyAtoms() == 0:
        return None
    return Chem.MolToSmiles(m, isomericSmiles=True)


def score_one(metric, y, p):
    """Point estimate for one task's metric. y, p are 1-D float arrays."""
    if metric == "auroc":
        return roc_auc_score(y, p)
    if metric == "auprc":
        return average_precision_score(y, p)
    if metric == "spearman":
        return spearmanr(y, p).correlation
    if metric == "mae":
        return float(np.mean(np.abs(y - p)))
    if metric == "rmse":
        return float(np.sqrt(np.mean((y - p) ** 2)))
    raise ValueError(metric)


def bootstrap_ci(metric, y, p, n_boot=2000, seed=0):
    """95% percentile CI. Returns (lo, hi) or (nan, nan) if undefined."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        # AUROC/AUPRC undefined if a resample is single-class
        if metric in ("auroc", "auprc") and len(np.unique(yb)) < 2:
            continue
        try:
            out.append(score_one(metric, yb, pb))
        except Exception:
            continue
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def load_joined(pred_path, truth_path):
    """KERMT preds are indexed by SMILES, one col per task, no ground truth.
    Ground truth lives in the test CSV. Join on canonical SMILES."""
    pred = pd.read_csv(pred_path)
    pred = pred.rename(columns={pred.columns[0]: "smiles"})
    truth = pd.read_csv(truth_path)

    pred["_key"] = pred["smiles"].map(canonical)
    truth["_key"] = truth["smiles"].map(canonical)
    pred = pred.dropna(subset=["_key"]).drop_duplicates("_key")
    truth = truth.dropna(subset=["_key"]).drop_duplicates("_key")

    n_p, n_t = len(pred), len(truth)
    tasks = [c for c in truth.columns if c not in ("smiles", "_key")]
    merged = truth.merge(pred, on="_key", suffixes=("_true", "_pred"), how="inner")
    if len(merged) < min(n_p, n_t):
        print(f"  note: joined {len(merged)} rows "
              f"(pred {n_p}, truth {n_t}) -- SMILES that failed to match were dropped",
              file=sys.stderr)
    return merged, tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="KERMT predictions CSV")
    ap.add_argument("--truth", required=True, help="ground-truth test CSV")
    ap.add_argument("--kind", choices=["classification", "regression"], required=True)
    ap.add_argument("--min-test", type=int, default=50,
                    help="flag tasks with fewer usable test rows")
    ap.add_argument("--min-minority", type=int, default=15,
                    help="flag classification tasks with fewer minority-class labels")
    a = ap.parse_args()

    merged, tasks = load_joined(a.pred, a.truth)

    rows = []
    for t in tasks:
        metric = TDC_METRIC.get(t)
        if metric is None:
            print(f"  skip {t}: no TDC metric mapping", file=sys.stderr)
            continue
        yt, pp = f"{t}_true", f"{t}_pred"
        if yt not in merged or pp not in merged:
            # single-task or unsuffixed layout fallback
            yt = t + "_true" if t + "_true" in merged else t
            pp = t + "_pred" if t + "_pred" in merged else t
        sub = merged[[yt, pp]].dropna()
        y = sub[yt].to_numpy(float)
        p = sub[pp].to_numpy(float)
        n = len(y)
        if n == 0:
            continue

        flags = []
        if n < a.min_test:
            flags.append(f"n={n}<{a.min_test}")
        n_min = None
        if metric in ("auroc", "auprc"):
            if len(np.unique(y)) < 2:
                rows.append((t, metric, float("nan"), None, None, n, "ALL ONE CLASS"))
                continue
            n_min = int(min((y == 0).sum(), (y == 1).sum()))
            if n_min < a.min_minority:
                flags.append(f"minority={n_min}")

        s = score_one(metric, y, p)
        lo, hi = bootstrap_ci(metric, y, p)
        rows.append((t, metric, s, lo, hi, n, ";".join(flags)))

    # ---- render ----
    hib = {"auroc", "auprc", "spearman"}  # higher is better
    print(f"\n{'='*82}\n{a.kind.upper()} PANEL  (union scaffold split -- DIRECTIONAL vs TxGemma, not leaderboard)\n{'='*82}")
    print(f"{'task':7} {'metric':8} {'KERMT (95% CI)':>24} {'TxGemma-27B':>20} {'Δ':>7}  flags")
    print("-"*82)
    for t, metric, s, lo, hi, n, flag in rows:
        if np.isnan(s):
            print(f"{t:7} {metric:8} {'-- '+flag:>24}")
            continue
        kerr = f"{s:.3f} [{lo:.3f},{hi:.3f}]"
        tx = TXGEMMA.get(t)
        if tx and tx[0] == metric:
            _, txs, txlo, txhi, txn = tx
            better = (s > txs) if metric in hib else (s < txs)
            arrow = "KERMT" if better else "TxG"
            overlap = not (hi < txlo or lo > txhi)
            delta = f"{s-txs:+.3f}"
            txcol = f"{txs:.3f}[{txlo:.2f},{txhi:.2f}]"
            note = "overlap" if overlap else f"**{arrow}**"
            print(f"{t:7} {metric:8} {kerr:>24} {txcol:>20} {delta:>7}  {note} {flag}")
        elif tx and tx[0] != metric:
            print(f"{t:7} {metric:8} {kerr:>24} {'METRIC MISMATCH':>20} {'--':>7}  TxG uses {tx[0]}; {flag}")
        else:
            print(f"{t:7} {metric:8} {kerr:>24} {'(no ref)':>20} {'--':>7}  {flag}")

    print("-"*82)
    print("overlap = KERMT 95% CI overlaps TxGemma 95% CI (no significant difference).")
    print("**KERMT**/**TxG** = point estimate better AND CIs may still overlap; check n.")
    print("Split differs from TxGemma's official TDC splits -> treat as directional only.")
    print("\nFor leaderboard-grade numbers: rerun single-task via tdc.benchmark_group")
    print("admet_group (5 seeds). It returns the correct metric per task automatically")
    print("and yields mean +/- std comparable to TxGemma Table 1 and the TDC leaderboard.")


if __name__ == "__main__":
    main()
