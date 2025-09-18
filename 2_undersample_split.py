#!/usr/bin/env python3
"""
Balance a folder dataset using OneSidedSelection (imblearn), then 80/10/10 stratified split.
Folder layout expected:

images/
  ├── no finding/
  ├── cardiomegaly/
  ├── lung cancer/
  ├── pneumonia/
  ├── pneumothorax/
  └── pleural effusion/

Result:
balanced_split/
  ├── train/{six labels}/files...
  ├── val/{six labels}/files...
  └── test/{six labels}/files...

Notes:
- We never open image contents; we only manipulate filenames and paths.
- OSS needs numerical features. Since we aren't reading images, we create
  deterministic pseudo-features from the file path hash (good enough for
  indexable prototype selection).
"""

import os
import sys
import shutil
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

# imbalanced-learn
from imblearn.under_sampling import OneSidedSelection
# Optional: to force perfectly equal class counts after OSS (uncomment to use)
# from imblearn.under_sampling import RandomUnderSampler


# ------------------------- Configuration -------------------------

INPUT_DIR = Path("images")                 # source root (with 6 class subfolders)
OUTPUT_DIR = Path("cxr_split")        # destination root
MOVE_FILES = False                         # False = copy (default). True = move.
RANDOM_STATE = 42                          # reproducibility
ALLOWED_EXTS = None                        # e.g., {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# If class folder names differ in capitalization/spacing, list them explicitly in the order you prefer.
EXPECTED_CLASSES = [
    "no finding",
    "cardiomegaly",
    "lung cancer",
    "pneumonia",
    "pneumothorax",
    "pleural effusion",
]

# ------------------------- Helpers -------------------------

def filename_to_features(p: Path) -> np.ndarray:
    """
    Map a path to a 4D numeric vector using a SHA-256 hash.
    This gives deterministic pseudo-features for OSS without reading files.
    """
    h = hashlib.sha256(str(p).encode("utf-8")).digest()  # 32 bytes
    # Split into 4 unsigned 64-bit ints, then scale to [0, 1]
    parts = [int.from_bytes(h[i:i+8], byteorder="big", signed=False) for i in range(0, 32, 8)]
    denom = (1 << 64) - 1
    return np.array([x / denom for x in parts], dtype=np.float64)


def gather_files(root: Path, allowed_exts=None):
    """
    Return lists: files (Path), labels (str), classes (sorted or EXPECTED_CLASSES).
    """
    if not root.exists():
        print(f"ERROR: Input dir not found: {root}", file=sys.stderr)
        sys.exit(1)

    # Determine class folders
    class_dirs = []
    if EXPECTED_CLASSES:
        for cname in EXPECTED_CLASSES:
            cpath = root / cname
            if cpath.is_dir():
                class_dirs.append(cpath)
    else:
        # Auto-discover subdirs
        class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    if not class_dirs:
        print("ERROR: No class subdirectories found.", file=sys.stderr)
        sys.exit(1)

    files, labels = [], []
    for cdir in class_dirs:
        label = cdir.name
        for fp in cdir.rglob("*"):
            if fp.is_file():
                if allowed_exts is None or fp.suffix.lower() in allowed_exts:
                    files.append(fp)
                    labels.append(label)

    if not files:
        print("ERROR: No files found under the given class folders.", file=sys.stderr)
        sys.exit(1)

    classes = [d.name for d in class_dirs]
    return files, labels, classes


def ensure_dirs(base: Path, classes):
    for split in ("train", "val", "test"):
        for c in classes:
            (base / split / c).mkdir(parents=True, exist_ok=True)


def move_or_copy(src: Path, dst: Path, move=False):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


# ------------------------- Main logic -------------------------

def main():
    np.random.seed(RANDOM_STATE)

    files, labels, classes = gather_files(INPUT_DIR, ALLOWED_EXTS)
    print("Original class counts:")
    orig_counts = Counter(labels)
    for c in classes:
        print(f"  {c:>18}: {orig_counts.get(c, 0)}")

    # Build pseudo-features for OSS
    X = np.vstack([filename_to_features(p) for p in files])
    #X = np.array(files)
    y = np.array(labels)

    # Run OneSidedSelection to under-sample majority classes
    oss = OneSidedSelection(random_state=RANDOM_STATE)
    X_res, y_res = oss.fit_resample(X, y)

    # Recover which original indices were kept by matching feature rows
    # (hash-based features are effectively unique; we map row tuples to indices)
    feat_to_indices = defaultdict(list)
    for idx, row in enumerate(X):
        feat_to_indices[tuple(row)].append(idx)

    kept_indices = []
    for row in X_res:
        key = tuple(row)
        idx_list = feat_to_indices.get(key, [])
        if not idx_list:
            raise RuntimeError("Failed to map resampled row back to original indices.")
        kept_indices.append(idx_list.pop())

    kept_indices = np.array(kept_indices, dtype=int)
    files_kept = [files[i] for i in kept_indices]
    labels_kept = y_res.tolist()

    print("\nAfter OneSidedSelection:")
    oss_counts = Counter(labels_kept)
    for c in classes:
        print(f"  {c:>18}: {oss_counts.get(c, 0)}")

    # (Optional) Force perfect balance to the smallest class count after OSS.
    # Uncomment the block below if you want exact equal counts per class.
    """
    rus = RandomUnderSampler(sampling_strategy='auto', random_state=RANDOM_STATE)
    # Use simple integer indices as features for RUS to preserve mapping precisely
    idx_array = kept_indices.reshape(-1, 1)
    idx_res, y_res2 = rus.fit_resample(idx_array, np.array(labels_kept))
    files_kept = [files[int(i)] for i in idx_res.ravel()]
    labels_kept = y_res2.tolist()
    print("\nAfter additional RandomUnderSampler (perfect balance):")
    bal_counts = Counter(labels_kept)
    for c in classes:
        print(f"  {c:>18}: {bal_counts.get(c, 0)}")
    """

    # Stratified 80/10/10 split on the kept set
    file_arr = np.array(files_kept)
    label_arr = np.array(labels_kept)

    f_train, f_temp, y_train, y_temp = train_test_split(
        file_arr, label_arr, test_size=0.2, random_state=RANDOM_STATE, stratify=label_arr
    )
    f_val, f_test, y_val, y_test = train_test_split(
        f_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )

    print("\nSplit sizes (files):")
    print(f"  train: {len(f_train)}")
    print(f"    val: {len(f_val)}")
    print(f"   test: {len(f_test)}")

    # Prepare output directories
    ensure_dirs(OUTPUT_DIR, classes)

    # Copy/move files into split/label structure
    def place(split_name, f_list, y_list):
        for fp, lbl in zip(f_list, y_list):
            dst = OUTPUT_DIR / split_name / lbl / fp.name
            # If filename collision occurs within a label split, append an index
            if dst.exists():
                stem, ext = os.path.splitext(fp.name)
                k = 1
                while True:
                    cand = OUTPUT_DIR / split_name / lbl / f"{stem}__dup{k}{ext}"
                    if not cand.exists():
                        dst = cand
                        break
                    k += 1
            move_or_copy(Path(fp), dst, move=MOVE_FILES)

    place("train", f_train, y_train)
    place("val",   f_val,   y_val)
    place("test",  f_test,  y_test)

    # Final sanity report
    def count_by_class(split):
        counts = Counter()
        for c in classes:
            d = OUTPUT_DIR / split / c
            if d.exists():
                counts[c] = sum(1 for _ in d.iterdir() if _.is_file())
        return counts

    print("\nFinal counts on disk:")
    for split in ("train", "val", "test"):
        counts = count_by_class(split)
        tot = sum(counts.values())
        breakdown = ", ".join(f"{c}: {counts.get(c,0)}" for c in classes)
        print(f"  {split:>5} (total {tot}): {breakdown}")

    print(f"\nDone. Output written to: {OUTPUT_DIR.resolve()}")
    if MOVE_FILES:
        print("Note: files were MOVED from the source. Set MOVE_FILES=False to copy instead.")

if __name__ == "__main__":
    main()
