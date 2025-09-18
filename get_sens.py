#!/usr/bin/env python3
"""
Scan a directory for files named:
    {model}_{train}_{test}_per_class_thresh.csv
Each file must contain per-class counts with columns including TP, FP, TN, FN.
For each (model, train, test), print macro- and micro- averaged Sens, Spec, F1.

Usage:
  python summarize_metrics.py /path/to/dir
  # If no path given, uses current directory.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

SUFFIX = "_per_class_thresh.csv"

def extract_model_train_test(filename: str):
    """
    Allow underscores inside the model name by stripping the known suffix
    and taking the last two underscore-separated tokens as train/test.
    """
    if not filename.endswith(SUFFIX):
        return None
    stem = filename[: -len(SUFFIX)]
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    train = parts[-2]
    test = parts[-1]
    model = "_".join(parts[:-2])
    return model, train, test

def safe_div(num, den):
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.true_divide(num, den)
        out[~np.isfinite(out)] = np.nan
    return out

def summarize_file(path: Path):
    # Load CSV (comma-separated). If your files use another delimiter, set sep accordingly.
    df = pd.read_csv(path)

    # Map columns case-insensitively for robustness
    colmap = {c.lower(): c for c in df.columns}
    required = ["tp", "fp", "tn", "fn"]
    missing = [k for k in required if k not in colmap]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}. Found: {list(df.columns)}")

    # Coerce to numeric
    for k in required:
        df[colmap[k]] = pd.to_numeric(df[colmap[k]], errors="coerce")

    TP = df[colmap["tp"]].to_numpy(dtype=float)
    FP = df[colmap["fp"]].to_numpy(dtype=float)
    TN = df[colmap["tn"]].to_numpy(dtype=float)
    FN = df[colmap["fn"]].to_numpy(dtype=float)

    # Per-class metrics
    sens = safe_div(TP, TP + FN)                  # TPR
    spec = safe_div(TN, TN + FP)                  # TNR
    f1   = safe_div(2*TP, 2*TP + FP + FN)

    # Macro (unweighted mean across classes)
    macro_sens = np.nanmean(sens)
    macro_spec = np.nanmean(spec)
    macro_f1   = np.nanmean(f1)

    # Micro (aggregate counts then compute)
    TP_sum, FP_sum, TN_sum, FN_sum = TP.sum(), FP.sum(), TN.sum(), FN.sum()
    micro_sens = TP_sum / (TP_sum + FN_sum) if (TP_sum + FN_sum) > 0 else np.nan
    micro_spec = TN_sum / (TN_sum + FP_sum) if (TN_sum + FP_sum) > 0 else np.nan
    micro_f1   = (2*TP_sum) / (2*TP_sum + FP_sum + FN_sum) if (2*TP_sum + FP_sum + FN_sum) > 0 else np.nan

    return {
        "macro_sens": macro_sens, "macro_spec": macro_spec, "macro_f1": macro_f1,
        "micro_sens": micro_sens, "micro_spec": micro_spec, "micro_f1": micro_f1,
    }

def fmt(x):
    return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.4f}"

def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    files = sorted(root.rglob(f"*{SUFFIX}"))

    if not files:
        print(f"No files found under {root.resolve()} matching '*{SUFFIX}'", file=sys.stderr)
        return

    rows = []
    for f in files:
        mtt = extract_model_train_test(f.name)
        if not mtt:
            continue
        model, train, test = mtt
        try:
            met = summarize_file(f)
        except Exception as e:
            print(f"[WARN] Skipping {f.name}: {e}", file=sys.stderr)
            continue

        rows.append({
            "model": model, "train": train, "test": test,
            **met
        })

    # Sort by model, then train, then test for stable output
    rows.sort(key=lambda r: (r["model"], r["train"], r["test"]))

    # Pretty print header
    header = [
        "MODEL", "TRAIN", "TEST",
        "MACRO_SENS", "MACRO_SPEC", "MACRO_F1",
        "MICRO_SENS", "MICRO_SPEC", "MICRO_F1"
    ]
    print("\t".join(header))
    for r in rows:
        print("\t".join([
            r["model"], r["train"], r["test"],
            fmt(r["macro_sens"]), fmt(r["macro_spec"]), fmt(r["macro_f1"]),
            fmt(r["micro_sens"]), fmt(r["micro_spec"]), fmt(r["micro_f1"]),
        ]))

if __name__ == "__main__":
    main()

