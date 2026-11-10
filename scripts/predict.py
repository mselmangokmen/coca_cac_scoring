#!/usr/bin/env python
"""End-to-end inference: DICOM series directory -> predicted Agatston score, with timing.

    python scripts/predict.py --encoder curia --weights weights/curia \
        --head outputs/heads/curia_L7/coca_gated/head.pt --series /data/coca/Gated_release_final/patient/0/<series> \
        [--series ...] [--save-maps outputs/maps]

Per series it prints the predicted score, its CAC-DRS band and the time spent reading the DICOM files,
running the encoder and running the head. --save-maps writes, per series, the additive decomposition of
the prediction: `contrib` (n_slices, grid, grid), whose sum is the predicted score, and the slice scores.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coca_cac.agatston import cac_drs  # noqa: E402
from coca_cac.dicom_io import find_series_dir, load_volume  # noqa: E402
from coca_cac.encoders import FrozenEncoder  # noqa: E402
from coca_cac.features import volume_features  # noqa: E402
from coca_cac.head import contributions, load_head, predict  # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dinov3-repo", default=None)
    ap.add_argument("--head", required=True)
    ap.add_argument("--series", action="append", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--save-maps", default=None)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = FrozenEncoder(a.encoder, a.weights, dev, a.dinov3_repo)
    ck = torch.load(a.head, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    if meta["encoder"] != a.encoder:
        raise SystemExit(f"the head was trained on {meta['encoder']} features, not {a.encoder}")
    model, mu, sd, cfg, _ = load_head(a.head, enc.dim, dev)
    volume_features(enc, np.zeros((a.batch, 512, 512), np.int16), meta["layer"], dev, meta["topk"], a.batch)  # warm-up
    print("series\tprediction\tband\tslices\tread_s\tencoder_s\thead_s")
    for s in a.series:
        t0 = time.perf_counter()
        hu, spacing, thickness = load_volume(find_series_dir(s))
        t1 = time.perf_counter()
        f, att, idx = volume_features(enc, hu, meta["layer"], dev, meta["topk"], a.batch)
        sync(); t2 = time.perf_counter()
        cache = dict(ftopk=f, atopk=att, svol=np.zeros(len(f), np.int32), vgt=np.zeros(1, np.float32),
                     vth=np.array([thickness], np.float32), vsx=np.array([spacing], np.float32),
                     vsy=np.array([spacing], np.float32))
        y = float(predict(model, cache, mu, sd, cfg["thickness_ref"], dev)[0])
        sync(); t3 = time.perf_counter()
        print(f"{s}\t{y:.1f}\t{cac_drs(y)}\t{len(hu)}\t{t1 - t0:.2f}\t{t2 - t1:.2f}\t{t3 - t2:.3f}", flush=True)
        if a.save_maps:
            c = contributions(model, cache, mu, sd, cfg["thickness_ref"], dev)       # (n_slices, K)
            g = enc.grid
            m = np.zeros((len(c), g * g), np.float32)
            np.put_along_axis(m, idx.astype(np.int64), c, axis=1)
            os.makedirs(a.save_maps, exist_ok=True)
            out = os.path.join(a.save_maps, os.path.basename(os.path.normpath(s)) + ".npz")
            np.savez(out, contrib=m.reshape(-1, g, g), slice_scores=m.sum(1), prediction=y)
            print(f"  contributions -> {out} (sum {m.sum():.1f})", flush=True)


if __name__ == "__main__":
    main()
