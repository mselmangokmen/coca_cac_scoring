#!/usr/bin/env python
"""Frozen-encoder top-K patch features of every patient of one COCA arm (both split sides).

    python scripts/extract_features.py --encoder curia --weights weights/curia \
        --cohort gated --data-root /data/coca --split-csv splits/coca_gated_split.csv
    python scripts/extract_features.py --encoder meddinov3 --weights weights/meddinov3/model.pth \
        --dinov3-repo dinov3 --cohort nongated --data-root /data/coca --split-csv splits/coca_nongated_split.csv

Writes <out>/<encoder>_L<layer>/coca_<cohort>.npz (layout in coca_cac/features.py). The cache holds both
split sides; train_head.py separates them with the same split file. The encoder is never updated.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coca_cac.dataset import CocaVolumes, volume_loader  # noqa: E402
from coca_cac.encoders import LAYER, FrozenEncoder  # noqa: E402
from coca_cac.features import TOPK, volume_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--encoder", required=True, choices=sorted(LAYER))
    ap.add_argument("--weights", required=True, help="Curia: model directory or hub id; MedDINOv3: model.pth")
    ap.add_argument("--dinov3-repo", default=None, help="clone of facebookresearch/dinov3 (MedDINOv3 only)")
    ap.add_argument("--layer", type=int, default=None, help="default: the encoder's layer in coca_cac/encoders.py")
    ap.add_argument("--cohort", required=True, choices=["gated", "nongated"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--out", default="outputs/features")
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--batch", type=int, default=8, help="slices per forward pass")
    ap.add_argument("--workers", type=int, default=4, help="DICOM reading processes")
    ap.add_argument("--limit", type=int, default=0, help="first N patients only (smoke test)")
    a = ap.parse_args()

    layer = a.layer or LAYER[a.encoder]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = FrozenEncoder(a.encoder, a.weights, dev, a.dinov3_repo)
    print(f"{a.encoder}: {enc.n_layers} blocks, dim {enc.dim}, {enc.n_reg} register tokens, grid {enc.grid}x{enc.grid}, "
          f"read at L{layer}, top-{a.topk} patches", flush=True)
    ds = CocaVolumes(a.data_root, a.cohort, a.split_csv)
    if a.limit:
        ds.rows = ds.rows[:a.limit]

    ft, at, ti, sv = [], [], [], []
    vp, vg, vt, vs, vn, skipped = [], [], [], [], [], []
    t0 = time.time()
    for i, it in enumerate(volume_loader(ds, a.workers)):
        if "error" in it:
            skipped.append((it["pid"], it["error"])); print(f"  skip {it['pid']}: {it['error']}", flush=True)
            continue
        f, att, idx = volume_features(enc, it["hu"], layer, dev, a.topk, a.batch)
        ft.append(f); at.append(att); ti.append(idx); sv.append(np.full(len(f), len(vp), np.int32))
        vp.append(it["pid"]); vg.append(it["agatston"]); vt.append(it["thickness"]); vs.append(it["spacing"])
        vn.append(len(f))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(ds)} patients  {time.time() - t0:.0f} s", flush=True)

    out = os.path.join(a.out, f"{a.encoder}_L{layer}")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"coca_{a.cohort}{'_limit%d' % a.limit if a.limit else ''}.npz")
    np.savez(path, ftopk=np.concatenate(ft), atopk=np.concatenate(at), tidx=np.concatenate(ti), svol=np.concatenate(sv),
             vpid=np.asarray(vp, np.int64), vgt=np.asarray(vg, np.float32), vth=np.asarray(vt, np.float32),
             vsx=np.asarray(vs, np.float32), vsy=np.asarray(vs, np.float32), vnsl=np.asarray(vn, np.int64),
             meta_encoder=a.encoder, meta_layer=layer, meta_topk=a.topk, meta_grid=enc.grid, meta_cohort=a.cohort)
    print(f"{len(vp)} patients, {sum(vn)} slices in {time.time() - t0:.0f} s -> {path}"
          + (f"; skipped {len(skipped)}: {skipped}" if skipped else ""), flush=True)


if __name__ == "__main__":
    main()
