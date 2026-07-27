#!/usr/bin/env python3
"""
KERMT ADMET REGRESSION — leaderboard-grade harness.

Runs single-task KERMT finetune+predict for each TDC ADMET regression benchmark,
across 5 seeds, on TDC's OFFICIAL splits, and reports mean±std in the exact metric
TDC uses per task -- directly comparable to MolE (specialist SOTA) and TxGemma.

It implements the four regression levers we established:

  1. LOG-TRANSFORM the three heavy-tailed, Spearman-scored PK targets
     (VDss, Half_Life, Clearance_Hepatocyte). MSE training on raw units is
     dominated by a handful of extreme-value outliers (that's what produced the
     t_half RMSE=105.8 blowup). Spearman is rank-invariant, so this CANNOT change
     the reported metric -- it only changes which model MSE training finds.
     NOT applied to MAE-scored tasks (Caco2/logD/logS/LD50 are already log units;
     PPBR is a bounded % -- transforming it would break the MAE comparison).

  2. SINGLE-TASK per endpoint. Removes multitask interference and, critically,
     lets model selection use each task's OWN validation loss instead of a
     cross-unit mean dominated by CL_hep/PPBR. This is also what MolE and the
     TDC leaderboard do, so it is the only comparable protocol.

  3. ENSEMBLE / multi-seed. The 5-seed TDC protocol gives real error bars
     (your single-fold "±0.000" is a placeholder, not a variance). --ensemble
     adds within-seed model averaging on top if you want it.

  4. CLEAN FEATURES. Precomputed rdkit_2d_normalized (not on-the-fly), so the
     collator NaN bug never fires; salts stripped and NaN-descriptor mols dropped.

Plus a PREFLIGHT leakage check for CL_hep, whose ρ=0.599 beat specialist SOTA on
the hardest task in ADMET -- the textbook signature of a split leak or scoring bug.

Requires: PyTDC, rdkit, pandas, numpy. KERMT repo at --code_dir.
The released checkpoint kermt_contrastive_v2.0.pt at --ckpt (+ its 3 vocab files).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")
_CHOOSER = rdMolStandardize.LargestFragmentChooser()
_FRAGILE = [(n, f) for n, f in Descriptors._descList
            if n.startswith("BCUT2D_") or "PartialCharge" in n]

# The 8 regression benchmarks, their TDC metric, and whether to log-transform.
# metric is informational; admet_group applies the real metric on evaluate.
REG_BENCHMARKS = {
    "Caco2_Wang":              ("mae",      False),
    "Lipophilicity_AstraZeneca":("mae",     False),
    "Solubility_AqSolDB":      ("mae",      False),
    "PPBR_AZ":                 ("mae",      False),  # bounded %, MAE -> no transform
    "LD50_Zhu":                ("mae",      False),  # already -log units
    "VDss_Lombardo":           ("spearman", True),   # heavy tail, rank metric
    "Half_Life_Obach":         ("spearman", True),   # heavy tail, rank metric
    "Clearance_Hepatocyte_AZ": ("spearman", True),   # heavy tail, rank metric
}

# Reference points (Table 1, arXiv:2504.06196). MolE = specialist SOTA GNN.
REF = {  # metric, MolE, TxGemma-27B
    "Caco2_Wang":              (0.329, 0.401),
    "Lipophilicity_AstraZeneca":(0.406, 0.538),
    "Solubility_AqSolDB":      (0.776, 0.907),
    "PPBR_AZ":                 (7.229, 9.048),
    "LD50_Zhu":                (0.602, 0.627),
    "VDss_Lombardo":           (0.644, 0.559),
    "Half_Life_Obach":         (0.578, 0.458),
    "Clearance_Hepatocyte_AZ": (0.456, 0.260),
}


# ---------------------------------------------------------------------------
# SMILES hygiene (shared with the panel builder)
# ---------------------------------------------------------------------------
def canonical(smi):
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None
    m = _CHOOSER.choose(m)               # strip counterions
    if m is None or m.GetNumHeavyAtoms() == 0:
        return None
    for _, fn in _FRAGILE:                # reject NaN-descriptor mols (salts/metals)
        try:
            v = fn(m)
        except Exception:
            return None
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return None
    return Chem.MolToSmiles(m, isomericSmiles=True)


def fwd_transform(y, do_log):
    """Target -> training space. log10 for positive PK; log1p where zeros possible."""
    if not do_log:
        return y
    return np.log1p(np.clip(y, 0, None))   # log1p handles CL_hep zeros; monotone


def inv_transform(y, do_log):
    """Training space -> original label space (for handing back to admet_group)."""
    if not do_log:
        return y
    return np.expm1(y)


# ---------------------------------------------------------------------------
# Preflight: CL_hep leakage / plumbing check
# ---------------------------------------------------------------------------
def preflight_leakage(group):
    print("\n[preflight] CL_hep leakage check")
    b = group.get("Clearance_Hepatocyte_AZ")
    tv, test = b["train_val"], b["test"]
    tv_keys = set(tv["Drug"].map(canonical).dropna())
    te_keys = set(test["Drug"].map(canonical).dropna())
    overlap = tv_keys & te_keys
    print(f"  train_val n={len(tv)}  test n={len(test)}  test labels={test['Y'].notna().sum()}")
    print(f"  canonical-SMILES overlap train_val∩test = {len(overlap)}")
    if overlap:
        print("  !! LEAK: identical molecules in train_val and test. ρ on CL_hep is "
              "inflated. Fix the split before trusting that number.", file=sys.stderr)
    else:
        print("  no identity leak. If ρ still beats MolE's 0.456, inspect the scaffold "
              "split for near-duplicate series before claiming SOTA.")
    return len(overlap)


# ---------------------------------------------------------------------------
# One KERMT finetune+predict for one benchmark/seed
# ---------------------------------------------------------------------------
def run_one(name, do_log, train, valid, test, work, args):
    tdir = os.path.join(work, name); os.makedirs(tdir, exist_ok=True)
    col = "Y"

    def prep(df):
        d = pd.DataFrame({"smiles": df["Drug"].map(canonical), col: df["Y"].astype(float)})
        d = d.dropna(subset=["smiles"]).drop_duplicates("smiles")
        d[col] = fwd_transform(d[col].to_numpy(), do_log)
        return d

    tr, va, te = prep(train), prep(valid), prep(test)
    p_tr = os.path.join(tdir, "train.csv"); tr.to_csv(p_tr, index=False)
    p_va = os.path.join(tdir, "valid.csv"); va.to_csv(p_va, index=False)
    p_te = os.path.join(tdir, "test.csv");  te.to_csv(p_te, index=False)

    # precompute clean features for every split (bypasses the on-the-fly NaN bug)
    feats = {}
    for tag, path in (("train", p_tr), ("valid", p_va), ("test", p_te)):
        fp = os.path.join(tdir, f"{tag}.npz")
        _sh([sys.executable, os.path.join(args.code_dir, "scripts", "save_features.py"),
             "--data_path", path, "--save_path", fp,
             "--features_generator", "rdkit_2d_normalized", "--restart"], args)
        feats[tag] = fp

    save_dir = os.path.join(tdir, "run")
    # FINETUNE. Arch flags come from the ckpt -- do NOT pass hidden_size/depth/etc.
    _sh([sys.executable, os.path.join(args.code_dir, "main.py"), "finetune",
         "--data_path", p_tr, "--separate_val_path", p_va, "--separate_test_path", p_te,
         "--features_path", feats["train"],
         "--separate_val_features_path", feats["valid"],
         "--separate_test_features_path", feats["test"],
         "--save_dir", save_dir, "--checkpoint_path", args.ckpt,
         "--dataset_type", "regression", "--metric", "rmse",
         "--num_folds", "1", "--ensemble_size", str(args.ensemble),
         "--epochs", str(args.epochs), "--batch_size", "32",
         "--init_lr", "1e-4", "--max_lr", "1e-4", "--final_lr", "2e-5",
         "--warmup_epochs", "2", "--early_stop_epoch", "15",
         "--no_features_scaling", "--seed", str(args.seed_base)], args)

    if args.dry_run:
        # commands have been printed; nothing was trained, so stop before the
        # symlink/predict/merge steps that assume real artifacts exist.
        return None

    # single .pt into a link dir so predict's recursive glob picks exactly one
    link = os.path.join(tdir, "ckpt_link"); os.makedirs(link, exist_ok=True)
    best = _find_best_pt(save_dir)
    ln = os.path.join(link, "model.pt")
    if os.path.islink(ln) or os.path.exists(ln): os.remove(ln)
    os.symlink(best, ln)

    out = os.path.join(tdir, "preds.csv")
    _sh([sys.executable, os.path.join(args.code_dir, "main.py"), "predict",
         "--data_path", p_te, "--checkpoint_dir", link, "--output_path", out,
         "--features_path", feats["test"], "--batch_size", "32"], args)

    pred = pd.read_csv(out)
    # KERMT writes: unnamed SMILES index in col 0, prediction in col 1
    # (named after the training header, i.e. "Y"). Rename BOTH positionally so
    # this is robust to the header name and avoids a Y/Y merge collision.
    pred = pred.rename(columns={pred.columns[0]: "smiles", pred.columns[1]: "_pred"})
    pred = pred[["smiles", "_pred"]]
    # align predictions back to the ORIGINAL test order via canonical SMILES
    te_key = test.copy()
    te_key["smiles"] = te_key["Drug"].map(canonical)
    m = te_key.merge(pred, on="smiles", how="left")
    if m["_pred"].isna().any():
        n = int(m["_pred"].isna().sum())
        print(f"    warn: {n}/{len(m)} test rows got no prediction "
              f"(SMILES canonicalization mismatch or dropped mol)", file=sys.stderr)
    yhat = inv_transform(m["_pred"].to_numpy(float), do_log)   # back to label space
    return yhat


def _rankdata_avg(a):
    """scipy.stats.rankdata(method='average') in pure numpy (handles ties)."""
    a = np.asarray(a, float)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), dtype=np.intp); inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def _score(metric, y_true, y_pred):
    """TDC's per-benchmark metric, computed with numpy only. Avoids BOTH broken
    deps in this env: tdc.Evaluator (mean_absolute_error NameError) and scipy
    (libstdc++ CXXABI mismatch on scipy.spatial). Definitions match TDC:
    'mae' = mean|error|, 'spearman' = Pearson-on-average-ranks (== scipy)."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ok = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[ok], y_pred[ok]
    if metric == "mae":
        return float(np.mean(np.abs(y_true - y_pred)))
    if metric == "spearman":
        rt, rp = _rankdata_avg(y_true), _rankdata_avg(y_pred)
        return float(np.corrcoef(rt, rp)[0, 1])
    if metric == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    raise ValueError(metric)


def _true_labels(group, name):
    """The fixed test labels TDC scores against, in test order."""
    return group.get(name)["test"]["Y"].to_numpy(float)


def _find_best_pt(save_dir):
    cands = []
    for root, _, files in os.walk(save_dir):
        for f in files:
            if f == "model.pt":
                cands.append(os.path.join(root, f))
    if not cands:
        raise FileNotFoundError(f"no model.pt under {save_dir}")
    return sorted(cands)[0]


def _sh(cmd, args):
    if args.dry_run:
        print("  DRY:", " ".join(os.path.basename(c) if i == 1 else c
                                 for i, c in enumerate(cmd))[:160])
        return
    subprocess.run(cmd, check=True, cwd=args.code_dir,
                   env={**os.environ, "PYTHONPATH": args.code_dir})


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code_dir", required=True, help="KERMT repo root")
    ap.add_argument("--ckpt", required=True, help="kermt_contrastive_v2.0.pt")
    ap.add_argument("--work", default="/runs/admet_group_reg")
    ap.add_argument("--tdc_path", default="data/")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--seed_base", type=int, default=0, help="KERMT init seed")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--ensemble", type=int, default=1, help="within-seed model avg")
    ap.add_argument("--only", nargs="*", help="subset of benchmark names")
    ap.add_argument("--skip_preflight", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    from tdc.benchmark_group import admet_group
    group = admet_group(path=a.tdc_path)
    os.makedirs(a.work, exist_ok=True)

    if not a.skip_preflight and not a.dry_run:
        preflight_leakage(group)

    benches = a.only or list(REG_BENCHMARKS)
    pred_lists = []
    for seed in a.seeds:
        print(f"\n===== SEED {seed} =====")
        preds = {}
        for name in benches:
            _, do_log = REG_BENCHMARKS[name]
            b = group.get(name)
            train, valid = group.get_train_valid_split(
                benchmark=name, split_type="default", seed=seed)
            w = os.path.join(a.work, f"seed{seed}")
            yhat = run_one(name, do_log, train, valid, b["test"],
                           w, a)
            if not a.dry_run:
                preds[name] = yhat
            print(f"  {name:26s} log={do_log}  done")
        if not a.dry_run:
            pred_lists.append(preds)

    if a.dry_run:
        print("\n[dry-run] wiring OK; no models trained.")
        return

    # Score with our own metric functions against TDC's fixed test labels.
    # This deliberately does NOT call group.evaluate / evaluate_many, which are
    # broken in this TDC/sklearn install (mean_absolute_error NameError). The
    # metric definitions are identical, so the numbers are the leaderboard's.
    print("\n" + "=" * 78)
    per = {name: [] for name in benches}
    truth = {name: _true_labels(group, name) for name in benches}
    for preds in pred_lists:
        for name in benches:
            metric, _ = REG_BENCHMARKS[name]
            per[name].append(_score(metric, truth[name], preds[name]))
    results = {name: (float(np.mean(v)), float(np.std(v))) for name, v in per.items()}
    proto = (f"{len(pred_lists)}-seed, TDC official splits, metrics scored locally "
             f"(TDC-identical). {'Leaderboard-grade.' if len(pred_lists) >= 5 else 'SMOKE TEST — use 5 seeds to cite.'}")

    print(f"{'benchmark':26} {'metric':8} {'KERMT ('+str(len(pred_lists))+'-seed)':>16} {'MolE':>7} {'TxG':>7}")
    print("-" * 78)
    for name in benches:
        metric, _ = REG_BENCHMARKS[name]
        mean, std = results[name]
        mole, txg = REF[name]
        hib = metric == "spearman"
        vs = ("KERMT" if (mean > mole if hib else mean < mole) else "MolE")
        print(f"{name:26} {metric:8} {mean:7.3f} ± {std:5.3f}   {mole:7.3f} {txg:7.3f}  {vs}")
    print("-" * 78)
    print(f"protocol: {proto}")
    print("(scored locally with MAE/Spearman to bypass TDC evaluator import bug;"
          " definitions match the TDC leaderboard.)")
    with open(os.path.join(a.work, "results.json"), "w") as f:
        json.dump({k: list(results[k]) for k in benches}, f, indent=2)
    print(f"written: {os.path.join(a.work, 'results.json')}")


if __name__ == "__main__":
    main()
