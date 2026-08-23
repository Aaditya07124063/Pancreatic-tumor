"""
Shared dataset utilities for pancreatic CT classification.

Loads images from ./train and ./test, deduplicates by content hash to prevent
leakage from duplicate files (including cross-class duplicates), and provides
stratified 80/10/10 train/val/test splits.
"""

import hashlib
import os
import pickle
from collections import defaultdict

import numpy as np
from sklearn.model_selection import train_test_split

CLASSES = ["normal", "pancreatic_tumor"]
DATA_DIRS = ["./train", "./test"]
SEEDS = [42, 7, 21, 99, 123]
EPOCHS = 50
PATIENCE = 15
CACHE_PATH = "./.dataset_cache.pkl"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_entries(data_dirs, classes):
    entries = []
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            continue
        for label, cls in enumerate(classes):
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    entries.append((os.path.join(cls_dir, fname), label))
    return entries


def _dedupe_entries(entries):
    by_hash = defaultdict(list)
    for path, label in entries:
        by_hash[md5(path)].append((path, label))

    paths, labels = [], []
    skipped_conflicts = 0
    skipped_dupes = 0

    for group in by_hash.values():
        unique_labels = {label for _, label in group}
        if len(unique_labels) > 1:
            skipped_conflicts += 1
            continue
        path, label = sorted(group, key=lambda x: x[0])[0]
        paths.append(path)
        labels.append(label)
        skipped_dupes += len(group) - 1

    if skipped_dupes or skipped_conflicts:
        print(
            f"  [data_utils] Deduped dataset: removed {skipped_dupes} duplicate files, "
            f"excluded {skipped_conflicts} cross-label conflict groups."
        )

    return np.array(paths), np.array(labels, dtype=np.int64)


def load_merged_dataset(data_dirs=None, classes=None, dedupe=True, use_cache=True):
    """
    Load all images from train and test folders.

    When dedupe=True, keeps one file per unique content hash. Images that appear
    under conflicting labels are excluded entirely.
    """
    data_dirs = data_dirs or DATA_DIRS
    classes = classes or CLASSES

    if use_cache and os.path.isfile(CACHE_PATH):
        with open(CACHE_PATH, "rb") as fh:
            cached = pickle.load(fh)
        if cached.get("data_dirs") == data_dirs and cached.get("classes") == classes:
            paths = cached["paths"]
            labels = cached["labels"]
            if not dedupe or cached.get("deduped"):
                return paths, labels

    entries = _collect_entries(data_dirs, classes)
    if not dedupe:
        paths = np.array([e[0] for e in entries])
        labels = np.array([e[1] for e in entries], dtype=np.int64)
    else:
        paths, labels = _dedupe_entries(entries)
        if use_cache:
            with open(CACHE_PATH, "wb") as fh:
                pickle.dump({
                    "data_dirs": data_dirs,
                    "classes": classes,
                    "deduped": True,
                    "paths": paths,
                    "labels": labels,
                }, fh)

    return paths, labels


def stratified_split(paths, labels, seed):
    """Return 80/10/10 stratified train, val, test splits."""
    tr_p, tmp_p, tr_l, tmp_l = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=seed
    )
    va_p, te_p, va_l, te_l = train_test_split(
        tmp_p, tmp_l, test_size=0.5, stratify=tmp_l, random_state=seed
    )
    return tr_p, va_p, te_p, tr_l, va_l, te_l


def print_split_summary(tr_l, va_l, te_l, classes=None):
    classes = classes or CLASSES
    total = len(tr_l) + len(va_l) + len(te_l)
    print(
        f"  Split sizes -> Train: {len(tr_l)} ({100*len(tr_l)/total:.1f}%) | "
        f"Val: {len(va_l)} ({100*len(va_l)/total:.1f}%) | "
        f"Test: {len(te_l)} ({100*len(te_l)/total:.1f}%)"
    )
    for split_name, split_labels in [("Train", tr_l), ("Val", va_l), ("Test", te_l)]:
        counts = [int((split_labels == i).sum()) for i in range(len(classes))]
        print(f"    {split_name}: " + " | ".join(f"{c}={n}" for c, n in zip(classes, counts)))
