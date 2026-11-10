#!/usr/bin/env python
"""Export the reference Agatston scores of the non-gated COCA arm to a CSV, with each series' geometry.

    python scripts/export_nongated_reference.py --data-root /data/coca --out data/coca_nongated_reference.csv

The non-gated arm ships its scores (scores.xlsx, column `total`) and no contours, so nothing is
recomputed here: the scores are Stanford's own. Only patients whose series directory exists are kept.
Columns: patient_id, total, n_slices, spacing, thickness (as read by coca_cac/dicom_io.py), series
(the series directory relative to the arm's root).
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coca_cac.cohorts import cohort_root, nongated_reference, nongated_series  # noqa: E402
from coca_cac.dicom_io import load_scan  # noqa: E402


def geometry(d):
    slices, spacing, thickness = load_scan(d)
    return len(slices), spacing, thickness


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-root", required=True, help="COCA release root (or its deidentified_nongated directory)")
    ap.add_argument("--out", default="data/coca_nongated_reference.csv")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    root = cohort_root(a.data_root, "nongated")
    ref, dirs = nongated_reference(root), nongated_series(root)
    pids = sorted(dirs)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        geo = dict(zip(pids, ex.map(geometry, [dirs[p] for p in pids])))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "total", "n_slices", "spacing", "thickness", "series"])
        for p in pids:
            n, sp, th = geo[p]
            w.writerow([p, repr(ref[p]), n, f"{sp:.6f}", f"{th:.3f}", os.path.relpath(dirs[p], root)])
    print(f"{len(pids)} of {len(ref)} scored patients have a series -> {a.out}")


if __name__ == "__main__":
    main()
