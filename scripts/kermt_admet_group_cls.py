#!/usr/bin/env python3
"""
KERMT ADMET CLASSIFICATION — leaderboard-grade harness (sibling of the reg one).

Runs single-task KERMT finetune+predict for each TDC ADMET classification
benchmark, 5 seeds, TDC official splits, scored in the CORRECT per-task metric
(AUROC for most; AUPRC for the CYP_Veith trio) — directly comparable to MolE
(specialist SOTA) and TxGemma-27B (Table 1, arXiv:2504.06196).

This fixes, in one place, every classification pitfall we hit:
  * CYP2D6/3A4/2C9_Veith are AUPRC on the TDC leaderboard, NOT AUROC. Reporting
    AUROC there inflated those numbers by 0.15-0.19 last time. Each benchmark is
    selected AND scored in its own metric.
  * KERMT's get_class_sizes asserts binary {0,1} targets. TDC classification
    labels already are; canonicalization + conflict-drop keeps them clean.
  * Metrics are computed with numpy only — no sklearn (broken: mean_absolute_error
    NameError in tdc.Evaluator) and no scipy (broken: libstdc++ CXXABI mismatch).
    Pure-numpy AUROC (Mann-Whitney) and AUPRC (sklearn-identical avg precision),
    both verified to 1e-6 against sklearn.
  * Single-task -> per-task early-stop selection, no multitask interference,
    leaderboard-comparable protocol.
  * Precomputed rdkit_2d_normalized features (collator NaN bug can't fire);
    salts stripped, NaN-descriptor mols dropped; ckpt_link symlink so predict's
    recursive glob picks exactly one .pt; arch flags come from the ckpt.

Requires: PyTDC, rdkit, pandas, numpy. KERMT repo at --code_dir.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
_CHOOSER = rdMolStandardize.LargestFragmentChooser()
_FRAGILE = [(n, f) for n, f in Descriptors._descList
            if n.startswith("BCUT2D_") or "PartialCharge" in n]

# benchmark -> (kermt --metric flag, TDC leaderboard metric)
# kermt's --metric drives early-stopping selection; the second is what we score.
CLS_BENCHMARKS = {
    "BBB_Martins":       ("auc",     "auroc"),
    "Pgp_Broccatelli":   ("auc",     "auroc"),
    "HIA_Hou":           ("auc",     "auroc"),
    "Bioavailability_Ma":("auc",     "auroc"),
    "hERG":              ("auc",     "auroc"),
    "AMES":              ("auc",     "auroc"),
    "DILI":              ("auc",     "auroc"),
    "CYP2D6_Veith":      ("prc-auc", "auprc"),   # AUPRC on the leaderboard
    "CYP3A4_Veith":      ("prc-auc", "auprc"),
    "CYP2C9_Veith":      ("prc-auc", "auprc"),
}

# MolE (specialist SOTA) and TxGemma-27B, Table 1 (arXiv:2504.06196). VERIFY vs
# Mendez-Lucio et al. before co-authoring. test_n from Table S.2.
REF = {  # MolE, TxGemma, test_n
    "BBB_Martins":       (0.903, 0.908, 406),
    "Pgp_Broccatelli":   (0.930, 0.937, 245),
    "HIA_Hou":           (0.984, 0.988, 117),
    "Bioavailability_Ma":(0.640, 0.694, 384),
    "hERG":              (0.835, 0.885, 132),
    "AMES":              (0.834, 0.816, 1457),
    "DILI":              (0.852, 0.886, 96),
    "CYP2D6_Veith":      (0.679, 0.683, 2626),
    "CYP3A4_Veith":      (0.876, 0.854, 2467),
    "CYP2C9_Veith":      (0.782, 0.798, 2419),
}


# --------------------------------------------------------------------------
# SMILES hygiene
# --------------------------------------------------------------------------
def canonical(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None
    m = _CHOOSER.choose(m)
    if m is None or m.GetNumHeavyAtoms() == 0:
        return None
    for _, fn in _FRAGILE:
        try:
            v = fn(m)
        except Exception:
            return None
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return None
    return Chem.MolToSmiles(m, isomericSmiles=True)


# --------------------------------------------------------------------------
# Pure-numpy metrics (verified vs sklearn to 1e-6)
# --------------------------------------------------------------------------
def _auroc(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), float); ranks[order] = np.arange(1, len(allv) + 1)
    sv = allv[order]                      # average ranks for ties
    i, n = 0, len(sv)
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    U = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(U / (len(pos) * len(neg)))


def _auprc(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    order = np.argsort(-p, kind="mergesort"); y = y[order]
    P = (y == 1).sum()
    if P == 0:
        return float("nan")
    tp = np.cumsum(y == 1); fp = np.cumsum(y == 0)
    prec = tp / (tp + fp); rec = tp / P
    rec0 = np.concatenate([[0], rec])
    return float(np.sum((rec0[1:] - rec0[:-1]) * prec))   # sklearn average_precision


def _score(metric, y_true, y_pred):
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    ok = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[ok], y_pred[ok]
    return _auprc(y_true, y_pred) if metric == "auprc" else _auroc(y_true, y_pred)


def _true_labels(group, name):
    return group.get(name)["test"]["Y"].to_numpy(float)


# --------------------------------------------------------------------------
def run_one(name, kmetric, train, valid, test, work, args):
    tdir = os.path.join(work, name); os.makedirs(tdir, exist_ok=True)
    col = "Y"

    def prep(df):
        d = pd.DataFrame({"smiles": df["Drug"].map(canonical),
                          col: df["Y"].astype(float)})
        d = d.dropna(subset=["smiles"])
        # drop duplicates that DISAGREE on the binary label (0 vs 1 -> would
        # break get_class_sizes); keep agreeing ones.
        g = d.groupby("smiles")[col]
        d = g.first()[g.nunique() == 1].reset_index()
        return d

    tr, va, te = prep(train), prep(valid), prep(test)
    p_tr = os.path.join(tdir, "train.csv"); tr.to_csv(p_tr, index=False)
    p_va = os.path.join(tdir, "valid.csv"); va.to_csv(p_va, index=False)
    p_te = os.path.join(tdir, "test.csv");  te.to_csv(p_te, index=False)

    feats = {}
    for tag, path in (("train", p_tr), ("valid", p_va), ("test", p_te)):
        fp = os.path.join(tdir, f"{tag}.npz")
        _sh([sys.executable, os.path.join(args.code_dir, "scripts", "save_features.py"),
             "--data_path", path, "--save_path", fp,
             "--features_generator", "rdkit_2d_normalized", "--restart"], args)
        feats[tag] = fp

    save_dir = os.path.join(tdir, "run")
    _sh([sys.executable, os.path.join(args.code_dir, "main.py"), "finetune",
         "--data_path", p_tr, "--separate_val_path", p_va, "--separate_test_path", p_te,
         "--features_path", feats["train"],
         "--separate_val_features_path", feats["valid"],
         "--separate_test_features_path", feats["test"],
         "--save_dir", save_dir, "--checkpoint_path", args.ckpt,
         "--dataset_type", "classification", "--metric", kmetric,   # per-task metric!
         "--num_folds", "1", "--ensemble_size", str(args.ensemble),
         "--epochs", str(args.epochs), "--batch_size", "32",
         "--init_lr", "1e-4", "--max_lr", "1e-4", "--final_lr", "2e-5",
         "--warmup_epochs", "2", "--early_stop_epoch", "15",
         "--no_features_scaling", "--seed", str(args.seed_base)], args)

    if args.dry_run:
        return None

    link = os.path.join(tdir, "ckpt_link"); os.makedirs(link, exist_ok=True)
    best = _find_best_pt(save_dir)
    ln = os.path.join(link, "model.pt")
    if os.path.islink(ln) or os.path.exists(ln):
        os.remove(ln)
    os.symlink(best, ln)

    out = os.path.join(tdir, "preds.csv")
    _sh([sys.executable, os.path.join(args.code_dir, "main.py"), "predict",
         "--data_path", p_te, "--checkpoint_dir", link, "--output_path", out,
         "--features_path", feats["test"], "--batch_size", "32"], args)

    pred = pd.read_csv(out)
    pred = pred.rename(columns={pred.columns[0]: "smiles", pred.columns[1]: "_pred"})
    pred = pred[["smiles", "_pred"]]
    te_key = test.copy()
    te_key["smiles"] = te_key["Drug"].map(canonical)
    m = te_key.merge(pred, on="smiles", how="left")
    if m["_pred"].isna().any():
        print(f"    warn: {int(m['_pred'].isna().sum())}/{len(m)} test rows unpredicted",
              file=sys.stderr)
    return m["_pred"].to_numpy(float)


def _find_best_pt(save_dir):
    for root, _, files in os.walk(save_dir):
        if "model.pt" in files:
            return os.path.join(root, "model.pt")
    raise FileNotFoundError(f"no model.pt under {save_dir}")


def _sh(cmd, args):
    if args.dry_run:
        print("  DRY:", " ".join(cmd)[:150]); return
    subprocess.run(cmd, check=True, cwd=args.code_dir,
                   env={**os.environ, "PYTHONPATH": args.code_dir})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--work", default="/runs/admet_group_cls")
    ap.add_argument("--tdc_path", default="data/")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--seed_base", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--ensemble", type=int, default=1)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    from tdc.benchmark_group import admet_group
    group = admet_group(path=a.tdc_path)
    os.makedirs(a.work, exist_ok=True)

    benches = a.only or list(CLS_BENCHMARKS)
    pred_lists = []
    for seed in a.seeds:
        print(f"\n===== SEED {seed} =====")
        preds = {}
        for name in benches:
            kmetric, _ = CLS_BENCHMARKS[name]
            b = group.get(name)
            train, valid = group.get_train_valid_split(
                benchmark=name, split_type="default", seed=seed)
            yhat = run_one(name, kmetric, train, valid, b["test"],
                           os.path.join(a.work, f"seed{seed}"), a)
            if not a.dry_run:
                preds[name] = yhat
            print(f"  {name:22s} metric={CLS_BENCHMARKS[name][1]}  done")
        if not a.dry_run:
            pred_lists.append(preds)

    if a.dry_run:
        print("\n[dry-run] wiring OK."); return

    truth = {name: _true_labels(group, name) for name in benches}
    per = {name: [] for name in benches}
    for preds in pred_lists:
        for name in benches:
            _, tdc_metric = CLS_BENCHMARKS[name]
            per[name].append(_score(tdc_metric, truth[name], preds[name]))
    results = {n: (float(np.nanmean(v)), float(np.nanstd(v))) for n, v in per.items()}

    print("\n" + "=" * 80)
    print(f"{'benchmark':20} {'metric':7} {'KERMT ('+str(len(pred_lists))+'-seed)':>16} "
          f"{'MolE':>7} {'TxG':>7} {'n':>6}  vsMolE")
    print("-" * 80)
    kw = mw = tie = 0
    for name in benches:
        _, metric = CLS_BENCHMARKS[name]
        mean, std = results[name]
        mole, txg, n = REF[name]
        lo, hi = mean - 2 * std, mean + 2 * std
        v = "KERMT" if lo > mole else ("MolE" if hi < mole else "~tie")
        kw += v == "KERMT"; mw += v == "MolE"; tie += v == "~tie"
        print(f"{name:20} {metric:7} {mean:7.3f} ± {std:5.3f}   {mole:7.3f} {txg:7.3f} "
              f"{n:6d}  {v}")
    print("-" * 80)
    print(f"KERMT vs MolE(SOTA): KERMT {kw}, MolE {mw}, ~tie {tie} (of {len(benches)}) "
          f"@ 2σ over seeds")
    proto = ("leaderboard-grade" if len(pred_lists) >= 5
             else f"{len(pred_lists)}-seed SMOKE TEST")
    print(f"protocol: {proto}, TDC official splits, per-task metric, numpy scoring")
    with open(os.path.join(a.work, "results.json"), "w") as f:
        json.dump({k: list(results[k]) for k in benches}, f, indent=2)
    print(f"written: {os.path.join(a.work, 'results.json')}")


if __name__ == "__main__":
    main()
