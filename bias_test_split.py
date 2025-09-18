from pathlib import Path
import shutil
import re
from collections import defaultdict

# --- CONFIG ---
TEST_DIR = Path("cxr_split/test")  # root containing the six label subdirectories
# Subgroup folder names (created next to TEST_DIR / in CWD)
SUBGROUPS = ["male", "female", "under40", "40-64", "65plus"]

# Overwrite behavior: if a file already exists at destination, set to True to overwrite
OVERWRITE = False
# ---------------

# Regex for filenames like "034Y_M_filename" (with any extension)
# Captures: age (3 digits) + 'Y', sex (M/F), and the rest of the name
FNAME_RE = re.compile(r"^(?P<age>\d{3})Y_(?P<sex>[MF])_(?P<rest>.+)$", re.IGNORECASE)

def parse_metadata(stem: str):
    """
    Given a filename stem (no extension), return (age_int, sex_upper) or None if no match.
    """
    m = FNAME_RE.match(stem)
    if not m:
        return None
    age = int(m.group("age"))
    sex = m.group("sex").upper()
    return age, sex

def age_bucket(age: int) -> list[str]:
    """Return the age-based subgroup(s) for a given age."""
    buckets = []
    if age < 40:
        buckets.append("under40")
    elif 40 <= age <= 64:
        buckets.append("40-64")
    else:
        buckets.append("65plus")
    return buckets

def sex_bucket(sex: str) -> list[str]:
    """Return the sex-based subgroup for M/F."""
    return ["male"] if sex == "M" else (["female"] if sex == "F" else [])

def ensure_dirs(labels: list[str]):
    """Create subgroup/label directories."""
    for sg in SUBGROUPS:
        for label in labels:
            (Path(sg) / label).mkdir(parents=True, exist_ok=True)

def main():
    if not TEST_DIR.exists():
        raise SystemExit(f"Input directory not found: {TEST_DIR.resolve()}")

    # Discover label subdirectories under test/ (use actual folders present)
    labels = sorted([p.name for p in TEST_DIR.iterdir() if p.is_dir()])
    if not labels:
        raise SystemExit(f"No label subdirectories found in {TEST_DIR}")

    ensure_dirs(labels)

    copied = defaultdict(int)
    skipped = defaultdict(int)
    malformed = []

    # Walk each label directory and process files
    for label in labels:
        label_dir = TEST_DIR / label
        for path in label_dir.rglob("*"):
            if path.is_dir():
                continue
            stem = path.stem  # filename without extension
            ext = path.suffix  # keep original extension (e.g., .png, .jpg)

            meta = parse_metadata(stem)
            if not meta:
                malformed.append(str(path))
                continue

            age, sex = meta
            targets = sex_bucket(sex) + age_bucket(age)

            for sg in targets:
                dest = Path(sg) / label / path.name
                if dest.exists() and not OVERWRITE:
                    skipped[(sg, label)] += 1
                    continue
                try:
                    shutil.copy2(path, dest)
                    copied[(sg, label)] += 1
                except Exception as e:
                    print(f"Failed to copy {path} -> {dest}: {e}")

    # Summary
    print("=== Done ===")
    print("Labels found:", ", ".join(labels))
    print("\nCopied counts by subgroup/label:")
    for (sg, label), n in sorted(copied.items()):
        print(f"  {sg}/{label}: {n}")

    if skipped:
        print("\nSkipped (already existed) by subgroup/label:")
        for (sg, label), n in sorted(skipped.items()):
            print(f"  {sg}/{label}: {n}")

    if malformed:
        print("\nFiles with unexpected filenames (not matching '###Y_[M|F]_rest'):")
        for p in malformed:
            print("  ", p)

if __name__ == "__main__":
    main()
