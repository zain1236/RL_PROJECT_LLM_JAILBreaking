"""
data_utils.py
--------------
Downloads the AdvBench "harmful strings" subset (Zou et al. 2023) from the
official llm-attacks repo and produces the small-scale (40-string) sample
used for this demo run, with an 80/20 train/test split (consistent with
both papers).
"""

import csv
import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import requests

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_strings.csv"
)


def download_advbench(dest: Path) -> List[str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return load_strings_csv(dest)

    resp = requests.get(ADVBENCH_URL, timeout=30)
    resp.raise_for_status()
    dest.write_text(resp.text, encoding="utf-8")
    return load_strings_csv(dest)


def load_strings_csv(path: Path) -> List[str]:
    strings = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # First row is the header ("target"); some rows may be quoted.
    for row in rows[1:]:
        if row and row[0].strip():
            strings.append(row[0].strip())
    return strings


@dataclass
class DatasetSplit:
    train: List[str]
    test: List[str]


def make_small_split(all_strings: List[str], n_total: int = 40, train_frac: float = 0.8,
                      seed: int = 42) -> DatasetSplit:
    rng = random.Random(seed)
    pool = all_strings.copy()
    rng.shuffle(pool)
    subset = pool[:n_total]
    n_train = int(round(n_total * train_frac))
    return DatasetSplit(train=subset[:n_train], test=subset[n_train:])


def save_split(split: DatasetSplit, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.txt").write_text("\n".join(split.train), encoding="utf-8")
    (out_dir / "test.txt").write_text("\n".join(split.test), encoding="utf-8")


def load_split(out_dir: Path) -> DatasetSplit:
    train = (out_dir / "train.txt").read_text(encoding="utf-8").splitlines()
    test = (out_dir / "test.txt").read_text(encoding="utf-8").splitlines()
    return DatasetSplit(train=[t for t in train if t.strip()], test=[t for t in test if t.strip()])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--n_total", type=int, default=40)
    parser.add_argument("--train_frac", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_csv = data_dir / "raw" / "harmful_strings.csv"
    all_strings = download_advbench(raw_csv)
    print(f"Downloaded {len(all_strings)} harmful strings from AdvBench.")

    split = make_small_split(all_strings, n_total=args.n_total, train_frac=args.train_frac, seed=args.seed)
    save_split(split, data_dir / "small_40")
    print(f"Saved small split: {len(split.train)} train / {len(split.test)} test -> {data_dir / 'small_40'}")
