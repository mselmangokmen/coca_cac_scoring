#!/usr/bin/env python
"""Patient-level train / test splits of both COCA arms, stratified by CAC-DRS band.

    python scripts/make_splits.py --gated-reference data/coca_gated_reference.csv \
        --nongated-reference data/coca_nongated_reference.csv

Per cohort, patients (in patient-id order) are grouped by the risk band of their reference score; each
band is shuffled with numpy RandomState(seed) and its first round(n * train_ratio) patients go to train,
the rest to test. Every patient has one series, so a patient-level split is also a volume-level one.

With the defaults (train_ratio 0.8, seed 42) the non-gated split is the one used in the accompanying
study (165 train / 40 test patients). Both splits are shipped in splits/; rerunning this script
reproduces them.
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coca_cac.agatston import CAC_DRS, cac_drs  # noqa: E402
from coca_cac.cohorts import read_gated_reference, read_nongated_reference, write_split  # noqa: E402


def stratified_split(scores, train_ratio, seed):
    """{pid: score} -> [(pid, split, score)]."""
    rng = np.random.RandomState(seed)
    buckets = {}
    for pid in sorted(scores):
        buckets.setdefault(cac_drs(scores[pid]), []).append(pid)
    rows = []
    for pids in buckets.values():                  # bands in order of first appearance
        pids = list(pids); rng.shuffle(pids)
        ntr = int(round(len(pids) * train_ratio))
        rows += [(p, "train" if j < ntr else "test", scores[p]) for j, p in enumerate(pids)]
    return sorted(rows)


def describe(name, rows):
    print(f"{name}: {len(rows)} patients")
    for side in ("train", "test"):
        c = Counter(cac_drs(s) for _, sp, s in rows if sp == side)
        print(f"  {side:5s} {sum(c.values()):4d}   " + "  ".join(f"{k} {c[k]}" for k in CAC_DRS))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--gated-reference", default="data/coca_gated_reference.csv")
    ap.add_argument("--nongated-reference", default="data/coca_nongated_reference.csv")
    ap.add_argument("--out-dir", default="splits")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    gated = read_gated_reference(a.gated_reference)
    rows = stratified_split(gated, a.train_ratio, a.seed)
    write_split(os.path.join(a.out_dir, "coca_gated_split.csv"), rows)
    describe("gated", rows)

    rows = stratified_split(read_nongated_reference(a.nongated_reference), a.train_ratio, a.seed)
    write_split(os.path.join(a.out_dir, "coca_nongated_split.csv"), rows)
    describe("non-gated", rows)


if __name__ == "__main__":
    main()
