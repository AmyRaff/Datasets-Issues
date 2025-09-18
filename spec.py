#!/usr/bin/env python3
"""
6-class evaluator with per-class (one-vs-rest) threshold optimization.

- Folder layout (ImageFolder):
    test/
      class0/
      class1/
      ...
      class5/

- Model selection matches your logic (resnet101/50, resnext, efficient, dense, convs, convb)
  and loads weights from the mapped checkpoint path.

Outputs:
  * f1_threshold_sweep.csv          # threshold -> per-class F1
  * per_class_thresholds.csv        # chosen t_k and metrics at those thresholds
  * f1_curves_by_class.png          # all class curves + vertical lines at best t_k
  * Console summary (per-class + macro/micro metrics using per-class thresholds)
"""

import argparse
import os
import csv
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.models import (
    resnet101, resnet50, resnext101_32x8d, efficientnet_v2_s,
    densenet161, convnext_small, convnext_base
)
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def safe_div(n, d):
    return float(n) / float(d) if d != 0 else 0.0


def confusion_from_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """Return TP, FP, TN, FN for 0/1 vectors."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)              # sensitivity / TPR
    specificity = safe_div(tn, tn + fp)         # TNR
    f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "specificity": specificity, "f1": f1}


def set_classifier_out_features(model: nn.Module, num_classes: int):
    """Fallback head edits for older torchvision versions."""
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Linear):
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        elif isinstance(model.classifier, nn.Sequential):
            # efficientnet_v2_s: classifier[1], convnext: classifier[2]
            for idx in reversed(range(len(model.classifier))):
                if isinstance(model.classifier[idx], nn.Linear):
                    in_f = model.classifier[idx].in_features
                    model.classifier[idx] = nn.Linear(in_f, num_classes)
                    break
    else:
        raise RuntimeError("Could not locate classifier head to set num_classes.")


def create_model(arch_fn, num_classes: int):
    """Create model with API compatibility across torchvision versions."""
    try:
        # torchvision >= 0.13 style
        return arch_fn(weights=None, num_classes=num_classes)
    except TypeError:
        # older API
        m = arch_fn(pretrained=False)
        set_classifier_out_features(m, num_classes)
        return m


# -----------------------------
# Model builder (your mapping)
# -----------------------------
def build_and_load_model(model_name: str, train_dir: str, device: torch.device):
    if model_name == 'resnet101':
        if train_dir == 'mimic':
            model_path = os.path.join('mimic_models', 'models', 'resnet101', 'model_18')
        elif train_dir == 'chex':
            model_path = os.path.join('chex_models', 'models', 'resnet101', 'model_24')
        elif train_dir == 'cxr14':
            model_path = 'cxr14_models/models/resnet101/model_23'
        elif train_dir == 'padchest':
            model_path = 'pad_models/models/resnet101/model_26'
        else:
            raise Exception('train set not instantiated')
        model = create_model(resnet101, 6)

    elif model_name == 'resnet50':
        if train_dir == 'mimic':
            model_path = os.path.join('mimic_models', 'models', 'resnet50', 'model_12')
        elif train_dir == 'chex':
            model_path = os.path.join('chex_models', 'models', 'resnet50', 'model_27')
        elif train_dir == 'cxr14':
            model_path = 'cxr14_models/models/resnet50/model_22'
        elif train_dir == 'padchest':
            model_path = 'pad_models/models/resnet50/model_19'
        else:
            raise Exception('train set not instantiated')
        model = create_model(resnet50, 6)

    elif model_name == 'resnext':
        if train_dir == 'mimic':
            model_path = os.path.join('mimic_models', 'models', 'resnext', 'model_13')
        elif train_dir == 'chex':
            model_path = os.path.join('chex_models', 'models', 'resnext', 'model_15')
        elif train_dir == 'cxr14':
            model_path = 'cxr14_models/models/resnext/model_24'
        elif train_dir == 'padchest':
            model_path = 'pad_models/models/resnext/model_13'
        else:
            raise Exception('train set not instantiated')
        model = create_model(resnext101_32x8d, 6)

    elif model_name == 'efficient':
        if train_dir == 'mimic':
            model_path = os.path.join('mimic_models', 'models', 'efficient', 'model_10')
        elif train_dir == 'chex':
            model_path = os.path.join('chex_models', 'models', 'efficient', 'model_41')
        elif train_dir == 'cxr14':
            model_path = 'cxr14_models/models/efficient/model_14'
        elif train_dir == 'padchest':
            model_path = 'pad_models/models/efficient/model_10'
        else:
            raise Exception('train set not instantiated')
        model = create_model(efficientnet_v2_s, 6)

    elif model_name == 'dense':
        if train_dir == 'mimic':
            model_path = os.path.join('mimic_models', 'models', 'densenet', 'model_14')
        elif train_dir == 'chex':
            model_path = os.path.join('chex_models', 'models', 'densenet', 'model_28')
        elif train_dir == 'cxr14':
            model_path = 'cxr14_models/models/densenet/model_21'
        elif train_dir == 'padchest':
            model_path = 'pad_models/models/densenet/model_10'
        else:
            raise Exception('train set not instantiated')
        model = create_model(densenet161, 6)

    elif model_name == 'convs':
        if train_dir == 'mimic':
            model_path = os.path.join('mimic_models', 'models', 'convnext_small', 'model_45')
        elif train_dir == 'chex':
            model_path = os.path.join('chex_models', 'models', 'convnext_small', 'model_47')
        elif train_dir == 'cxr14':
            model_path = 'cxr14_models/models/convnext_small/model_37'
        elif train_dir == 'padchest':
            model_path = 'pad_models/models/convnext_small/model_48'
        else:
            raise Exception('train set not instantiated')
        model = create_model(convnext_small, 6)

    elif model_name == 'convb':
        if train_dir == 'mimic':
            model_path = os.path.join('mimic_models', 'models', 'conv_base', 'model_27')
        elif train_dir == 'chex':
            model_path = os.path.join('chex_models', 'models', 'conv_base', 'model_25')
        elif train_dir == 'cxr14':
            model_path = 'cxr14_models/models/conv_base/model_16'
        elif train_dir == 'padchest':
            model_path = 'pad_models/models/conv_base/model_41'
        else:
            raise Exception('train set not instantiated')
        model = create_model(convnext_base, 6)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    state = torch.load(os.path.normpath(model_path), map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, model_path


# -----------------------------
# Core evaluation
# -----------------------------
def gather_probs_labels(loader: DataLoader, model: nn.Module, device: torch.device):
    probs_all, labels_all = [], []
    with torch.no_grad():
        for imgs, y in loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)              # [N, 6]
            probs = F.softmax(logits, dim=1)  # [N, 6]
            probs_all.append(probs.cpu().numpy())
            labels_all.append(y.cpu().numpy())
    probs = np.concatenate(probs_all, axis=0)
    y_int = np.concatenate(labels_all, axis=0)
    y_onehot = np.eye(probs.shape[1], dtype=np.int32)[y_int]
    return probs, y_onehot, y_int


def choose_per_class_thresholds(probs: np.ndarray, y_onehot: np.ndarray, thresholds: np.ndarray):
    """
    For each class k, pick t_k that maximizes F1 of (probs[:,k] >= t) vs y[:,k].
    Returns:
        best_t (C,), best_metrics (list of dict per class), f1_grid (T, C) for plotting/CSV
    """
    N, C = probs.shape
    T = len(thresholds)
    f1_grid = np.zeros((T, C), dtype=float)
    best_t = np.zeros(C, dtype=float)
    best_metrics = []

    for k in range(C):
        yk = y_onehot[:, k].astype(np.int32)
        pk = probs[:, k]
        best_idx, best_f1 = 0, -1.0
        best_m = None

        for i, t in enumerate(thresholds):
            yhat = (pk >= t).astype(np.int32)
            tp, fp, tn, fn = confusion_from_binary(yk, yhat)
            m = metrics_from_counts(tp, fp, tn, fn)
            f1_grid[i, k] = m["f1"]
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best_idx = i
                best_m = {"tp": tp, "fp": fp, "tn": tn, "fn": fn, **m}

        best_t[k] = thresholds[best_idx]
        best_metrics.append(best_m)

    return best_t, best_metrics, f1_grid


def apply_per_class_thresholds(probs: np.ndarray, best_t: np.ndarray):
    """Return binary predictions [N,C] where pred[:,k] = (probs[:,k] >= best_t[k])."""
    return (probs >= best_t.reshape(1, -1)).astype(np.int32)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-name', required=True,
                    choices=['resnet101', 'resnet50', 'resnext', 'efficient', 'dense', 'convs', 'convb'])
    ap.add_argument('--train-dir', choices=['mimic', 'chex', 'cxr14', 'padchest'])
    ap.add_argument('--test-dir', required=True)
    ap.add_argument('--val-dir', default=None, help="Optional: choose thresholds on this set, then evaluate on test_dir.")
    ap.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--threshold_steps', type=int, default=21)  # number of thresholds in [0,1]
    ap.add_argument('--out_sweep_csv', default='f1_threshold_sweep.csv')
    ap.add_argument('--out_best_csv', default='per_class_thresholds.csv')
    ap.add_argument('--out_plot', default='f1_curves_by_class.png')
    args = ap.parse_args()
    
    args.out_sweep_csv = f'{args.model_name}_{args.train_dir}_{args.test_dir.split("_")[0]}_f1_sweep.csv'
    args.out_best_csv = f'{args.model_name}_{args.train_dir}_{args.test_dir.split("_")[0]}_per_class_thresh.csv'
    args.out_plot = f'{args.model_name}_{args.train_dir}_{args.test_dir.split("_")[0]}_f1_curves_by_class.png'

    device = torch.device('cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu')

    # Data transforms
    tfms = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Datasets & loaders
    def make_loader(root):
        ds = datasets.ImageFolder(root, transform=tfms)
        if len(ds.classes) != 6:
            raise ValueError(f"Expected 6 classes in {root}, found {len(ds.classes)}. Classes: {ds.classes}")
        return ds, DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    test_ds, test_loader = make_loader(args.test_dir)
    class_names = test_ds.classes

    model, model_path = build_and_load_model(args.model_name, args.train_dir, device)
    print(f"Loaded weights from: {model_path}")

    # Threshold source (validation or test)
    thresholds = np.linspace(0.0, 1.0, max(2, args.threshold_steps))

    if args.val_dir:
        val_ds, val_loader = make_loader(args.val_dir)
        print(f"Choosing thresholds on validation dir: {args.val_dir}")
        probs_val, y1h_val, _ = gather_probs_labels(val_loader, model, device)
        best_t, best_metrics_val, f1_grid = choose_per_class_thresholds(probs_val, y1h_val, thresholds)
    else:
        print("WARNING: No --val_dir provided. Choosing thresholds on TEST (may leak).")
        probs_test_tmp, y1h_test_tmp, _ = gather_probs_labels(test_loader, model, device)
        best_t, best_metrics_val, f1_grid = choose_per_class_thresholds(probs_test_tmp, y1h_test_tmp, thresholds)

    # Save sweep CSV (threshold -> F1 for each class)
    sweep_header = ["threshold"] + [f"f1_{cls}" for cls in class_names]
    with open(args.out_sweep_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(sweep_header)
        for i, t in enumerate(thresholds):
            row = [float(t)] + [float(f1_grid[i, k]) for k in range(len(class_names))]
            w.writerow(row)
    print(f"Saved sweep to: {args.out_sweep_csv}")

    # Evaluate on TEST using the chosen per-class thresholds
    probs_test, y1h_test, yint_test = gather_probs_labels(test_loader, model, device)
    ypred_bin = apply_per_class_thresholds(probs_test, best_t)

    # Per-class metrics at chosen thresholds
    per_class_rows = []
    tp_sum = fp_sum = tn_sum = fn_sum = 0
    for k, cls in enumerate(class_names):
        tp, fp, tn, fn = confusion_from_binary(y1h_test[:, k], ypred_bin[:, k])
        m = metrics_from_counts(tp, fp, tn, fn)
        per_class_rows.append({
            "class": cls,
            "threshold": float(best_t[k]),
            "F1": m["f1"],
            "Sensitivity_TPR": m["recall"],
            "Specificity_TNR": m["specificity"],
            "Precision": m["precision"],
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        })
        tp_sum += tp; fp_sum += fp; tn_sum += tn; fn_sum += fn

    # Macro / micro
    macro_f1 = float(np.mean([r["F1"] for r in per_class_rows]))
    macro_sens = float(np.mean([r["Sensitivity_TPR"] for r in per_class_rows]))
    macro_spec = float(np.mean([r["Specificity_TNR"] for r in per_class_rows]))
    micro_prec = safe_div(tp_sum, tp_sum + fp_sum)
    micro_rec  = safe_div(tp_sum, tp_sum + fn_sum)
    micro_f1   = safe_div(2 * micro_prec * micro_rec, micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    # Save per-class thresholds CSV
    best_header = ["class", "threshold", "F1", "Sensitivity_TPR", "Specificity_TNR", "Precision", "TP", "FP", "TN", "FN"]
    with open(args.out_best_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=best_header)
        w.writeheader()
        for r in per_class_rows:
            w.writerow(r)
    print(f"Saved per-class thresholds/metrics to: {args.out_best_csv}")

    # Plot: F1 curves for all classes + vertical lines at chosen thresholds
    plt.figure(figsize=(8, 6))
    for k, cls in enumerate(class_names):
        plt.plot(thresholds, f1_grid[:, k], label=f"{cls} F1")
        plt.axvline(best_t[k], linestyle="--")
    plt.xlabel("Threshold")
    plt.ylabel("F1")
    plt.title("Per-class F1 vs Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=150)
    print(f"Saved curves to: {args.out_plot}")

    # Console summary
    print("\n=== Evaluation on TEST with per-class thresholds ===")
    for r in per_class_rows:
        print(f"{r['class']:>15s}  t*={r['threshold']:.3f} | F1={r['F1']:.4f}  Sens={r['Sensitivity_TPR']:.4f}  "
              f"Spec={r['Specificity_TNR']:.4f}  Prec={r['Precision']:.4f}  TP/FP/TN/FN={r['TP']}/{r['FP']}/{r['TN']}/{r['FN']}")
    print("----------------------------------------------------")
    print(f"Macro: F1={macro_f1:.4f}  Sens={macro_sens:.4f}  Spec={macro_spec:.4f}")
    print(f"Micro: F1={micro_f1:.4f}  Prec={micro_prec:.4f}  Rec={micro_rec:.4f}")
    print("====================================================\n")


if __name__ == "__main__":
    main()
