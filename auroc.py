import argparse
import csv
import os
from typing import Dict, List

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import convnext_small, convnext_base, efficientnet_v2_s, resnet50
from torchvision.models import resnext101_32x8d, densenet161, resnet101

from sklearn.metrics import roc_auc_score, average_precision_score

# --------- Helpers ---------
def get_transforms():
    # Standard ResNet preprocessing (ImageNet stats)
    return transforms.Compose([
        transforms.ToTensor(),
    ])

def compute_metrics(y_true: torch.Tensor, y_prob: torch.Tensor, class_names: List[str]) -> Dict:
    """
    y_true: (N, C) one-hot
    y_prob: (N, C) softmax probabilities
    """
    y_true_np = y_true.numpy()
    y_prob_np = y_prob.numpy()
    C = y_true_np.shape[1]

    per_class = []
    for c in range(C):
        row = {"class": class_names[c]}
        # AUROC per class (OvR). Requires at least one pos and one neg.
        try:
            row["auroc"] = roc_auc_score(y_true_np[:, c], y_prob_np[:, c])
        except ValueError:
            row["auroc"] = float("nan")

        # AUPRC (average precision) per class
        try:
            row["auprc"] = average_precision_score(y_true_np[:, c], y_prob_np[:, c])
        except ValueError:
            row["auprc"] = float("nan")

        per_class.append(row)

    # Macro/micro averages
    metrics = {
        "per_class": per_class,
        "macro": {},
        "micro": {},
    }

    # Macro: average of per-class scores (ignoring NaNs)
    import math
    macro_auroc_vals = [r["auroc"] for r in per_class if not math.isnan(r["auroc"])]
    macro_auprc_vals = [r["auprc"] for r in per_class if not math.isnan(r["auprc"])]

    metrics["macro"]["auroc"] = sum(macro_auroc_vals) / len(macro_auroc_vals) if macro_auroc_vals else float("nan")
    metrics["macro"]["auprc"] = sum(macro_auprc_vals) / len(macro_auprc_vals) if macro_auprc_vals else float("nan")

    # Micro: treat as one big binary problem across classes
    try:
        metrics["micro"]["auroc"] = roc_auc_score(y_true_np.ravel(), y_prob_np.ravel())
    except ValueError:
        metrics["micro"]["auroc"] = float("nan")
    try:
        metrics["micro"]["auprc"] = average_precision_score(y_true_np.ravel(), y_prob_np.ravel())
    except ValueError:
        metrics["micro"]["auprc"] = float("nan")

    return metrics

def save_csv(metrics: Dict, out_path: str):
    fieldnames = ["class", "auroc", "auprc"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics["per_class"]:
            writer.writerow(row)
        # Add macro/micro rows
        writer.writerow({"class": "macro_avg", "auroc": metrics["macro"]["auroc"], "auprc": metrics["macro"]["auprc"]})
        writer.writerow({"class": "micro_avg", "auroc": metrics["micro"]["auroc"], "auprc": metrics["micro"]["auprc"]})


# --------- Main ---------
def main():
    
    parser = argparse.ArgumentParser(description="Compute AUROC and AUPRC for a multi-class ResNet101.")
    parser.add_argument("--test-dir", type=str)
    parser.add_argument("--train-dir", type=str)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)  # 0 is safest across platforms
    args = parser.parse_args()

    model_dir = None
    model_name = args.model_name
    train_dir = args.train_dir
    device = 'cuda'
    
    if model_name == 'resnet101':
        if train_dir == 'mimic':
            model_dir = 'mimic_models/models/resnet101/model_18'
        elif train_dir == 'chex':
            model_dir = 'chex_models/models/resnet101/model_24'
        elif train_dir == 'cxr14':
            model_dir = 'cxr14_models/models/resnet101/model_23'
        elif train_dir == 'padchest':
            model_dir = 'pad_models/models/resnet101/model_26'
        else:
            raise Exception('train set not instantiated')
        saved_model = resnet101(num_classes=6)
    elif model_name == 'resnet50':
        if train_dir == 'mimic':
            model_dir = 'mimic_models/models/resnet50/model_12'
        elif train_dir == 'chex':
            model_dir = 'chex_models/models/resnet50/model_27'
        elif train_dir == 'cxr14':
            model_dir = 'cxr14_models/models/resnet50/model_22'
        elif train_dir == 'padchest':
            model_dir = 'pad_models/models/resnet50/model_19'
        else:
            raise Exception('train set not instantiated')
        saved_model = resnet50(num_classes=6)
    elif model_name == 'resnext':
        if train_dir == 'mimic':
            model_dir = 'mimic_models/models/resnext/model_13'
        elif train_dir == 'chex':
            model_dir = 'chex_models/models/resnext/model_15'
        elif train_dir == 'cxr14':
            model_dir = 'cxr14_models/models/resnext/model_24'
        elif train_dir == 'padchest':
            model_dir = 'pad_models/models/resnext/model_13'
        else:
            raise Exception('train set not instantiated')
        saved_model = resnext101_32x8d(num_classes=6)
    elif model_name == 'efficient':
        if train_dir == 'mimic':
            model_dir = 'mimic_models/models/efficient/model_10'
        elif train_dir == 'chex':
            model_dir = 'chex_models/models/efficient/model_41'
        elif train_dir == 'cxr14':
            model_dir = 'cxr14_models/models/efficient/model_14'
        elif train_dir == 'padchest':
            model_dir = 'pad_models/models/efficient/model_10'
        else:
            raise Exception('train set not instantiated')
        saved_model = efficientnet_v2_s(num_classes=6)
    elif model_name == 'dense':
        if train_dir == 'mimic':
            model_dir = 'mimic_models/models/densenet/model_14'
        elif train_dir == 'chex':
            model_dir = 'chex_models/models/densenet/model_28'
        elif train_dir == 'cxr14':
            model_dir = 'cxr14_models/models/densenet/model_21'
        elif train_dir == 'padchest':
            model_dir = 'pad_models/models/densenet/model_10'
        else:
            raise Exception('train set not instantiated')
        saved_model = densenet161(num_classes=6)
    elif model_name == 'convs':
        if train_dir == 'mimic':
            model_dir = 'mimic_models/models/convnext_small/model_45'
        elif train_dir == 'chex':
            model_dir = 'chex_models/models/convnext_small/model_47'
        elif train_dir == 'cxr14':
            model_dir = 'cxr14_models/models/convnext_small/model_37'
        elif train_dir == 'padchest':
            model_dir = 'pad_models/models/convnext_small/model_48'
        else:
            raise Exception('train set not instantiated')
        saved_model = convnext_small(num_classes=6)
    elif model_name == 'convb':
        if train_dir == 'mimic':
            model_dir = 'mimic_models/models/conv_base/model_27'
        elif train_dir == 'chex':
            model_dir = 'chex_models/models/conv_base/model_25'
        elif train_dir == 'cxr14':
            model_dir = 'cxr14_models/models/conv_base/model_16'
        elif train_dir == 'padchest':
            model_dir = 'pad_models/models/conv_base/model_41'
        else:
            raise Exception('train set not instantiated')
        saved_model = convnext_base(num_classes=6)
    assert model_dir != None
    
    saved_model.load_state_dict(torch.load(model_dir, map_location=torch.device(device)))
    saved_model.to('cuda')
    saved_model.eval()
    
    # Dataset & loader
    transform = get_transforms()
    test_ds = datasets.ImageFolder(root=args.test_dir, transform=transform)
    class_names = test_ds.classes
    num_classes = len(class_names)
    if num_classes != 6:
        print(f"[Warning] Detected {num_classes} classes from folder names, but model was defined with 6.")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    model = saved_model
    # Model (exactly as you described)
    #model = torch.load(model_dir, map_location=torch.device("cpu"))
    #model.load_state_dict(state)
    #model.eval()

    #device = torch.device("cpu")  # weights were loaded on CPU; keep it consistent
    #model.to(device)

    # Inference
    all_probs = []
    all_targets = []
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for imgs, targets in tqdm(test_loader):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = softmax(logits).cpu()  # (B, C)
            all_probs.append(probs)

            # One-hot encode targets
            one_hot = torch.zeros(len(targets), num_classes, dtype=torch.float32)
            one_hot[torch.arange(len(targets)), targets] = 1.0
            all_targets.append(one_hot)

    if len(all_probs) == 0:
        raise RuntimeError("No data found in the test loader. Check your --test-dir path and subfolders.")

    y_prob = torch.cat(all_probs, dim=0)      # (N, C)
    y_true = torch.cat(all_targets, dim=0)    # (N, C)

    # Metrics
    metrics = compute_metrics(y_true, y_prob, class_names)

    # Pretty print
    print("\nPer-class metrics:")
    width = max(len(c) for c in class_names + ["macro_avg", "micro_avg"])
    print(f"{'Class'.ljust(width)}  AUROC      AUPRC")
    print("-" * (width + 21))
    for row in metrics["per_class"]:
        print(f"{row['class'].ljust(width)}  {row['auroc']:.6f}  {row['auprc']:.6f}")
    print("-" * (width + 21))
    print(f"{'macro_avg'.ljust(width)}  {metrics['macro']['auroc']:.6f}  {metrics['macro']['auprc']:.6f}")
    print(f"{'micro_avg'.ljust(width)}  {metrics['micro']['auroc']:.6f}  {metrics['micro']['auprc']:.6f}")

    # Save CSV
    out_csv = f"{args.train_dir}_{args.test_dir.split('_')[0]}_{model_name}_metrics_auroc_auprc.csv"
    save_csv(metrics, out_csv)
    print(f"\nSaved: {os.path.abspath(out_csv)}")

if __name__ == "__main__":
    main()
