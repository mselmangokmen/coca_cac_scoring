#!/usr/bin/env python
"""Recompute the reference Agatston score of every annotated gated COCA patient from its OsiriX contours.

    python scripts/build_gated_reference.py --data-root /data/coca --out data/coca_gated_reference.csv

The gated arm ships contours but no scores (see coca_cac/agatston.py for the definition). Columns:
patient_id, total (the reference), cross_check (per-ROI score without dilation, which reproduces the
annotator's own OsiriX values), n_slices, spacing, thickness, n_annotated_slices, out_of_range (ROIs
whose ImageIndex falls outside the stack; such patients are excluded downstream). Patients whose
series or contour file cannot be read are listed on stdout and left out.
"""
import argparse
import csv
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coca_cac.agatston import CAC_DRS, agatston_from_masks, agatston_from_rois, cac_drs  # noqa: E402
from coca_cac.cohorts import cohort_root, gated_series  # noqa: E402
from coca_cac.dicom_io import load_volume, parse_osirix_rois  # noqa: E402


def one(rec):
    pid, series, xml = rec
    try:
        hu, spacing, thickness = load_volume(series)
        rois = parse_osirix_rois(xml)
        total, oor = agatston_from_masks(hu, rois, spacing)
        cross, _ = agatston_from_rois(hu, rois, spacing)
        return pid, dict(total=total, cross_check=cross, n_slices=len(hu), spacing=spacing, thickness=thickness,
                         n_annotated_slices=len(rois), out_of_range=oor), None
    except Exception as e:
        return pid, None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-root", required=True, help="COCA release root (or its Gated_release_final directory)")
    ap.add_argument("--out", default="data/coca_gated_reference.csv")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    pats = gated_series(cohort_root(a.data_root, "gated"))
    print(f"{len(pats)} gated patients with a series and a contour file", flush=True)
    t0, done, failed = time.time(), {}, {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (pid, rec, err) in enumerate(ex.map(one, pats, chunksize=2)):
            (failed.__setitem__(pid, err) if err else done.__setitem__(pid, rec))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(pats)}  {time.time() - t0:.0f} s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cols = ["patient_id", "total", "cross_check", "n_slices", "spacing", "thickness", "n_annotated_slices",
            "out_of_range"]
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for pid in sorted(done):
            r = done[pid]
            w.writerow([pid, f"{r['total']:.4f}", f"{r['cross_check']:.4f}", r["n_slices"], f"{r['spacing']:.6f}",
                        f"{r['thickness']:.3f}", r["n_annotated_slices"], r["out_of_range"]])

    oor = sorted(p for p, r in done.items() if r["out_of_range"])
    ok = {p: r for p, r in done.items() if not r["out_of_range"]}
    sc = np.array([r["total"] for r in ok.values()]); cr = np.array([r["cross_check"] for r in ok.values()])
    print(f"\n{len(done)} scored in {time.time() - t0:.0f} s -> {a.out}")
    print(f"unreadable: {len(failed)} {failed}")
    print(f"ROI outside the slice stack (excluded): {len(oor)} {oor}")
    print(f"usable reference scores: {len(ok)}; median {np.median(sc):.1f}, zeros {int((sc == 0).sum())}")
    c = Counter(cac_drs(s) for s in sc)
    for k in CAC_DRS:
        print(f"  {k:8s} {c[k]:4d}  ({100 * c[k] / len(sc):4.1f}%)")
    same = sum(cac_drs(x) == cac_drs(y) for x, y in zip(sc, cr))
    print(f"cross-check: score ratio median {np.median(sc / np.maximum(cr, 1e-9)):.3f}, "
          f"same CAC-DRS band {same}/{len(sc)}")


if __name__ == "__main__":
    main()
