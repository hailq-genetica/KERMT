#!/usr/bin/env python3
"""
Build multitask ADMET panels for KERMT finetuning from Therapeutics Data Commons.

Why this script exists (all three are silent-failure traps in KERMT):

1. KERMT has ONE global --dataset_type. KermtFinetuneTask.get_loss_func picks
   BCEWithLogitsLoss *or* MSELoss for ALL tasks. You cannot mix classification
   and regression endpoints in one finetune. -> two panels, two runs.

2. Missing labels must be EMPTY CELLS. moldataset.py:75 does
       targets = [float(x) if x != '' else None for x in line[1:]]
   and molgraph.py:584 builds  mask = [x is not None].
   A literal "nan"/"NaN" string parses to float('nan'), which is NOT None,
   so mask=1 and the NaN poisons the whole loss. pandas to_csv writes NaN as
   '' by default -- do not pass na_rep, and never fillna(0).

3. Per-endpoint TDC splits LEAK across a multitask panel: a molecule can be
   train for Caco2 and test for BBB. The union must be scaffold-split ONCE.
   (This is why the KERMT paper published its own multitask ADMET splits.)

Regression scale mixing is NOT a problem: StandardScaler.fit uses np.nanmean /
np.nanstd per column, so each endpoint is z-scored independently.

Usage:
    pip install PyTDC rdkit pandas
    python build_admet_panels.py --out /data/admet
"""

import argparse
import math
import os
import sys
from collections import defaultdict

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

_CHOOSER = rdMolStandardize.LargestFragmentChooser()

# Descriptors that depend on Gasteiger partial charges. These are the ONLY ones
# that realistically go NaN, and they do it for salts / metal complexes
# (Na+, Li+, Pt...). Checking ~10 is cheap; checking all 217 is not.
_FRAGILE = [(n, f) for n, f in Descriptors._descList
            if n.startswith("BCUT2D_") or "PartialCharge" in n]


def _descriptors_ok(mol):
    """Reject molecules whose RDKit 2D descriptors are NaN.

    This matters because of a real KERMT bug: MolCollator.__call__
    (molgraph.py:566-574) computes features ON THE FLY for
    'rdkit_2d_normalized_onthefly' and 'rdkit_2d_normalized_cuik_molmaker',
    which BYPASSES the NaN -> 0 fix in MoleculeDatapoint.__init__
    (moldataset.py:70-72). Only the `[d.features for d in batch]` branches get
    cleaned. A NaN feature flows into the FFN, BCEWithLogitsLoss does not assert
    on it, so NaN weights propagate silently through training -- and you find out
    at the first validation, when the eval path's F.binary_cross_entropy trips
    `input_val >= zero && input_val <= one` as a CUDA device-side assert.
    """
    for _, fn in _FRAGILE:
        try:
            v = fn(mol)
        except Exception:
            return False
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return False
    return True


def canonical(smi):
    """Canonicalize with stereo preserved, strip counterions, reject NaN-prone mols.

    Salt stripping is correct science independent of the bug above: the sodium
    counterion is not the thing that crosses the blood-brain barrier, and TDC
    ADMET sets (AqSolDB and AMES especially) carry a lot of salts. Canonical
    SMILES is also the merge key for the wide pivot, so stripping has to happen
    BEFORE the join or the same parent lands in two rows.
    """
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    m = _CHOOSER.choose(m)
    if m is None or m.GetNumHeavyAtoms() == 0:
        return None
    if not _descriptors_ok(m):
        return None
    return Chem.MolToSmiles(m, isomericSmiles=True)

# ---------------------------------------------------------------------------
# Panel definitions.
#
# NOTE: TDC dataset names are VERIFY items -- they drift between releases and
# are not stable knowledge. The script validates every name against TDC's own
# registry before downloading and tells you what it actually found.
#
# (tdc_module, tdc_name, column_name_in_output)
# ---------------------------------------------------------------------------

CLS_PANEL = [
    ("ADME", "BBB_Martins",        "BBB"),       # CNS penetration -- existential for a PD chaperone
    ("ADME", "Pgp_Broccatelli",    "Pgp"),       # efflux; gates real brain exposure
    ("ADME", "HIA_Hou",            "HIA"),
    ("ADME", "Bioavailability_Ma", "F20"),
    ("ADME", "CYP2D6_Veith",       "CYP2D6"),
    ("ADME", "CYP3A4_Veith",       "CYP3A4"),
    ("ADME", "CYP2C9_Veith",       "CYP2C9"),
    ("Tox",  "hERG",               "hERG"),      # basic amine + lipophilic aromatic = live risk
    ("Tox",  "AMES",               "AMES"),      # the ambroxol aniline is a classic mutagenicity alert
    ("Tox",  "DILI",               "DILI"),
]

REG_PANEL = [
    ("ADME", "Caco2_Wang",                 "Caco2"),
    ("ADME", "Lipophilicity_AstraZeneca",  "logD"),
    ("ADME", "Solubility_AqSolDB",         "logS"),
    ("ADME", "PPBR_AZ",                    "PPBR"),
    ("ADME", "VDss_Lombardo",              "VDss"),
    ("ADME", "Half_Life_Obach",            "t_half"),
    ("ADME", "Clearance_Hepatocyte_AZ",    "CL_hep"),
    ("Tox",  "LD50_Zhu",                   "LD50"),
]


def scaffold_of(smi):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
    except Exception:
        return ""


def scaffold_split(smiles, frac=(0.8, 0.1, 0.1), seed=0):
    """Balanced Murcko scaffold split over the UNION of the panel.

    Mirrors chemprop/KERMT 'scaffold_balanced': scaffold groups larger than half
    a test-set go to train first, so val/test stay scaffold-diverse rather than
    being dominated by one big series.
    """
    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[scaffold_of(s)].append(i)

    n = len(smiles)
    n_tr, n_va = int(frac[0] * n), int(frac[1] * n)

    big, small = [], []
    cutoff = max(1, n_va // 2)
    for idx in groups.values():
        (big if len(idx) > cutoff else small).append(idx)

    import random
    random.Random(seed).shuffle(small)
    ordered = big + small  # big scaffold families into train

    train, val, test = [], [], []
    for idx in ordered:
        if len(train) + len(idx) <= n_tr:
            train += idx
        elif len(val) + len(idx) <= n_va:
            val += idx
        else:
            test += idx
    return train, val, test


def resolve(module_name, ds_name):
    """Validate a TDC name against the live registry before downloading."""
    from tdc.utils import retrieve_dataset_names
    try:
        available = retrieve_dataset_names(module_name)
    except Exception:
        return True, None  # registry unavailable; let the loader try anyway
    lowered = {a.lower(): a for a in available}
    if ds_name.lower() in lowered:
        return True, lowered[ds_name.lower()]
    return False, available


def load_endpoint(module_name, ds_name, col, binary):
    ok, info = resolve(module_name, ds_name)
    if not ok:
        near = [a for a in info if a.lower()[:4] == ds_name.lower()[:4]]
        print(f"  !! '{ds_name}' not in TDC {module_name} registry. "
              f"Close matches: {near or '(none)'}", file=sys.stderr)
        return None
    real = info or ds_name

    import tdc.single_pred as sp
    cls = getattr(sp, module_name)
    df = cls(name=real).get_data()          # columns: Drug_ID, Drug, Y
    df = df[["Drug", "Y"]].copy()
    df["smiles"] = df["Drug"].map(canonical)
    df = df.dropna(subset=["smiles"])

    # Canonicalization collapses salt forms / tautomer spellings onto a single
    # SMILES, so duplicate rows are common. A BINARY endpoint must NOT be
    # averaged: 0 and 1 -> 0.5, which trips the
    #     assert set(np.unique(task_targets)) <= {0, 1}
    # in utils.get_class_sizes -- and it trips AFTER data loading, so you only
    # find out minutes into the run. Keep duplicates that agree; drop the ones
    # that disagree rather than inventing a label the source contradicts itself
    # on. A dropped molecule just becomes an empty cell -> mask 0.
    n_raw = len(df)
    g = df.groupby("smiles")["Y"]
    if binary:
        nun = g.nunique()
        n_conflict = int((nun > 1).sum())
        df = g.first()[nun == 1].reset_index()
    else:
        n_conflict = 0
        df = g.mean().reset_index()
    df = df.rename(columns={"Y": col})
    note = ""
    if n_raw != len(df):
        note = f"  (from {n_raw:,} rows; {n_conflict} conflicting dropped)"
    print(f"  {col:8s} <- {real:32s} {len(df):6,d} mols{note}")
    return df


def build(panel, kind, outdir, seed=0):
    print(f"\n=== {kind.upper()} panel ===")
    binary = kind == "classification"
    frames = []
    for module_name, ds_name, col in panel:
        f = load_endpoint(module_name, ds_name, col, binary)
        if f is not None:
            frames.append(f)
    if not frames:
        print(f"  no endpoints resolved for {kind}; skipping", file=sys.stderr)
        return None

    # Outer-join into the sparse wide matrix. Unmeasured -> NaN -> empty cell.
    wide = frames[0]
    for f in frames[1:]:
        wide = wide.merge(f, on="smiles", how="outer")

    if binary:
        # Hard gate. KERMT's get_class_sizes asserts set(unique) <= {0,1} and
        # dies with a bare AssertionError well into the run. Fail here instead,
        # naming the offending column and the offending values.
        bad = {}
        for c in wide.columns.drop("smiles"):
            off = sorted(set(wide[c].dropna().unique()) - {0.0, 1.0})
            if off:
                bad[c] = off[:5]
        if bad:
            for c, off in bad.items():
                print(f"  !! {c} NON-BINARY: {off} -- either it belongs in the "
                      f"regression panel, or its duplicates were averaged",
                      file=sys.stderr)
            raise SystemExit(
                "refusing to write a classification panel that would fail at "
                "get_class_sizes")

    cols = ["smiles"] + [c for c in wide.columns if c != "smiles"]
    wide = wide[cols]

    tr, va, te = scaffold_split(wide["smiles"].tolist(), seed=seed)
    os.makedirs(outdir, exist_ok=True)

    for split_name, idx in (("train", tr), ("valid", va), ("test", te)):
        part = wide.iloc[idx]
        path = os.path.join(outdir, f"{split_name}.csv")
        # na_rep defaults to '' -- that is exactly what KERMT needs. Do not change.
        part.to_csv(path, index=False)
        print(f"  {split_name:5s} {len(part):6,d} rows -> {path}")

    dens = wide.drop(columns=["smiles"]).notna().mean()
    print(f"\n  union: {len(wide):,} molecules x {len(cols)-1} tasks")
    print(f"  label density: {dens.mean():.1%} overall")
    for c, d in dens.items():
        print(f"    {c:8s} {d:6.1%}")
    return len(cols) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/admet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", choices=["classification", "regression"])
    a = ap.parse_args()

    if a.only != "regression":
        build(CLS_PANEL, "classification", os.path.join(a.out, "cls"), a.seed)
    if a.only != "classification":
        build(REG_PANEL, "regression", os.path.join(a.out, "reg"), a.seed)

    print("\nNext: two separate finetunes (one global --dataset_type each).")
    print(f"  cls -> {a.out}/cls   --dataset_type classification --metric auc")
    print(f"  reg -> {a.out}/reg   --dataset_type regression     --metric rmse")


if __name__ == "__main__":
    main()
