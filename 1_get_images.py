import argparse
import os
import shutil
from collections import defaultdict
from tqdm import tqdm

import pandas as pd


CLASSES = [
    "no finding",
    "cardiomegaly",
    "pleural effusion",
    "lung cancer",
    "pneumonia",
    "pneumothorax",
]


def parse_labels_to_targets(raw_label_str):
    """
    Map NIH ChestX-ray14 labels to the requested 6-class set.
    - Effusion -> pleural effusion
    - Mass or Nodule -> lung cancer
    - Keep Cardiomegaly, Pneumonia, Pneumothorax
    - 'No Finding' stays 'no finding' ONLY if it's the sole label
    Returns a sorted list of target class names (could be multiple).
    """
    if not isinstance(raw_label_str, str) or not raw_label_str.strip():
        return []

    parts = [p.strip().lower() for p in raw_label_str.split("|") if p.strip()]
    parts_set = set(parts)

    # If it's exactly "no finding", treat as that single class
    if parts_set == {"no finding"}:
        return ["no finding"]

    targets = set()
    if "cardiomegaly" in parts_set:
        targets.add("cardiomegaly")
    if "effusion" in parts_set:
        targets.add("pleural effusion")
    if "pneumonia" in parts_set:
        targets.add("pneumonia")
    if "pneumothorax" in parts_set:
        targets.add("pneumothorax")
    if "mass" in parts_set or "nodule" in parts_set:
        targets.add("lung cancer")

    return sorted(list(targets))


def ensure_dirs(root_out):
    os.makedirs(root_out, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(root_out, c), exist_ok=True)


def main():

    # Read CSV
    df = pd.read_csv('cxr14_data/Data_Entry_2017.csv')

    # Standard column names in NIH CSV
    # Image Index, Finding Labels, Follow-up #, Patient ID, View Position, ...
    if "View Position" not in df.columns or "Patient ID" not in df.columns or "Image Index" not in df.columns:
        raise ValueError("CSV missing required columns: 'Image Index', 'Patient ID', 'View Position'.")

    # Keep only PA view
    df = df[df["View Position"].astype(str).str.upper() == "PA"].copy()

    # One image per patient: choose earliest Follow-up #
    #if "Follow-up #" in df.columns:
        #df["Follow-up #"] = pd.to_numeric(df["Follow-up #"], errors="coerce").fillna(0).astype(int)
        #df = df.sort_values(["Patient ID", "Follow-up #", "Image Index"])
        #df = df.drop_duplicates(subset=["Patient ID"], keep="first")
    #else:
        # If column missing, just keep first occurrence per patient after sorting by Image Index
    #    df = df.sort_values(["Patient ID", "Image Index"]).drop_duplicates(subset=["Patient ID"], keep="first")

    # Map labels to target classes
    df["targets"] = df["Finding Labels"].apply(parse_labels_to_targets)

    # Keep rows that map to at least one of the 6 classes
    df = df[df["targets"].map(len) > 0].copy()
    print(len(df))

    # Prepare output folders
    ensure_dirs('images')

    # Copy/move files
    per_class_counts = defaultdict(int)
    missing_files = []
    operations = []

    for _, row in tqdm(df.iterrows()):
        img_name = row["Image Index"]
        src_path = os.path.join('cxr14_data\images-224\images-224/', img_name)

        if not os.path.isfile(src_path):
            missing_files.append(img_name)
            continue

        age = row["Patient Age"]
        gender = row["Patient Gender"]
        for target in row["targets"]:
            new_name = f'{age}_{gender}_{img_name}'
            dst_path = os.path.join('images', target, new_name)
            # If file already exists at destination, skip to be idempotent
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)  # copy first so we can still place into multiple folders
            per_class_counts[target] += 1
            operations.append(
                {
                    "image": new_name,
                    "patient_id": row["Patient ID"],
                    "view": row.get("View Position", ""),
                    "follow_up": row.get("Follow-up #", ""),
                    "finding_labels_raw": row.get("Finding Labels", ""),
                    "targets": "|".join(row["targets"]),
                    "dest": dst_path,
                }
            )

    # Report
    total_copied = sum(per_class_counts.values())
    print("\nDone.")
    print(f"Images processed (one per patient, PA only): {len(df)}")
    print(f"Images placed (counting multi-folder copies): {total_copied}")
    print("\nPer-class counts:")
    for c in CLASSES:
        print(f"  {c:>16}: {per_class_counts[c]}")

    if missing_files:
        print(f"\nMissing in '{'all'}' (skipped): {len(missing_files)}")
        # Show a few examples
        for m in missing_files[:10]:
            print("  -", m)

    # Save an audit CSV of what we did
    audit_path = "build_audit.csv"
    pd.DataFrame(operations).to_csv(audit_path, index=False)
    print(f"\nAudit written to: {audit_path}")


if __name__ == "__main__":
    main()
