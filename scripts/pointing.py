#!/usr/bin/env python
"""Calcium localization of frozen encoders on gated COCA: pointing game and related metrics, every layer.

    python scripts/pointing.py --data-root /data/coca --split-csv splits/coca_gated_split.csv --side test \
        --encoder curia=weights/curia --encoder meddinov3=weights/meddinov3/model.pth --dinov3-repo dinov3

Three stages, each reusing the previous one's files under --out:
  pool     every calcium-bearing slice of the selected patients, plus, per patient, as many calcium-free
           slices as it has calcium-bearing ones, nearest in slice index (so patient identity and slice
           level do not predict the label). A slice is calcium-bearing when its calcium mask (annotated,
           filled and dilated contour AND HU > 130; coca_cac/agatston.py) is non-empty.
           -> pool_<side>.npz (HU slices, masks, patient ids, slice indices)
  encode   per encoder, the class token and the head-averaged class-token -> patch attention of every
           pool slice at every layer -> enc_<side>_<encoder>.npz
  metrics  metrics.json, report.txt, layers.png (definitions in coca_cac/localization.py), at each
           encoder's fixed layer (coca_cac/encoders.py LAYER, or KEY=PATH@LAYER) and, for pointing and the
           CLS probe, at every layer.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coca_cac.agatston import calcium_mask  # noqa: E402
from coca_cac.dataset import CocaVolumes, volume_loader  # noqa: E402
from coca_cac.encoders import DISPLAY, LAYER, FrozenEncoder  # noqa: E402
from coca_cac.features import TOPK, to_input  # noqa: E402
from coca_cac import localization as loc  # noqa: E402

BASE_GRID = 32          # grid of the HU and uniform baselines (the grid of both supported encoders)


def build_pool(a, path):
    sides = ("train", "test") if a.side == "all" else (a.side,)
    ds = CocaVolumes(a.data_root, "gated", a.split_csv, sides=sides, with_rois=True)
    hu_k, m_k, pid_k, sidx_k, pos_k = [], [], [], [], []
    short, t0 = 0, time.time()
    for i, it in enumerate(volume_loader(ds, a.workers)):
        if "error" in it:
            print(f"  skip {it['pid']}: {it['error']}", flush=True); continue
        hu, rois = it["hu"], it["rois"]
        masks = {j: calcium_mask(hu[j], rois[j]) for j in rois if 0 <= j < len(hu)}
        masks = {j: m for j, m in masks.items() if m.any()}
        if not masks:
            continue                                 # no calcium-bearing slice: the patient adds nothing
        pos = np.array([j in masks for j in range(len(hu))])
        keep, s = loc.matched_negatives(np.zeros(len(hu), int), np.arange(len(hu)), pos)
        short += s
        for j in keep:
            hu_k.append(hu[j]); m_k.append(masks.get(j, np.zeros(hu.shape[1:], bool)))
            pid_k.append(it["pid"]); sidx_k.append(j); pos_k.append(bool(pos[j]))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(ds)} patients  {time.time() - t0:.0f} s", flush=True)
    pos = np.array(pos_k)
    m = np.stack(m_k)
    np.savez(path, hu=np.stack(hu_k), mask_bits=np.packbits(m, axis=-1), mask_width=m.shape[-1], pid=np.array(pid_k),
             slice_index=np.array(sidx_k), is_pos=pos, side=a.side)
    print(f"pool: {len(pos)} slices ({int(pos.sum())} calcium-bearing) from {len(set(pid_k))} patients; "
          f"negatives short of a 1:1 match: {short} -> {path}", flush=True)


def load_pool(path):
    z = np.load(path)
    masks = np.unpackbits(z["mask_bits"], axis=-1)[..., :int(z["mask_width"])].astype(bool)
    return z["hu"], masks, z["pid"], z["slice_index"], z["is_pos"]


def encode(a, name, weights, pool, path):
    hu, masks, pid, _, ispos = pool
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = FrozenEncoder(name, weights, dev, a.dinov3_repo)
    layers = list(range(1, enc.n_layers + 1))
    cls, att = [], []
    t0 = time.time()
    for j in range(0, len(hu), a.batch):
        out = enc.forward(to_input(hu[j:j + a.batch], dev), layers)
        cls.append(torch.stack([out[L][0] for L in layers], 1).float().cpu().numpy())
        att.append(torch.stack([out[L][2].mean(1) for L in layers], 1).half().cpu().numpy())
    pm = np.stack([loc.patch_mask(m, enc.grid) for m in masks[ispos]])
    np.savez(path, cls=np.concatenate(cls), attn=np.concatenate(att), pos_mask=pm, grid=enc.grid,
             layers=np.array(layers), dim=enc.dim)
    print(f"[{name}] {len(hu)} slices, {enc.n_layers} layers, grid {enc.grid}x{enc.grid} in {time.time() - t0:.0f} s "
          f"-> {path}", flush=True)
    del enc; torch.cuda.empty_cache()


def encoder_metrics(z, ispos, pid, layer, topk):
    g = int(z["grid"]); layers = [int(L) for L in z["layers"]]
    M = z["pos_mask"].reshape(-1, g, g)
    y = ispos.astype(int)
    li = layers.index(layer)
    A = z["attn"][:, li, :].astype(np.float32) * (g * g)           # uniform attention = 1.0
    m = loc.detector_metrics(A[ispos].reshape(-1, g, g), M, A[~ispos].reshape(-1, g, g), loc.TAUS_ATTN)
    m["cls_auroc"], m["cls_auroc_sd"] = loc.cls_auroc(z["cls"][:, li, :].astype(np.float32), y, pid)
    m["topk_coverage"] = loc.topk_coverage(A[ispos], M, topk)
    m.update(layer=layer, grid=g)
    by = {}
    keep = [i for i in range(len(M)) if M[i].any()]
    for j, L in enumerate(layers):
        Aj = z["attn"][:, j, :].astype(np.float32)[ispos]
        pt = [bool(M[i].reshape(-1)[Aj[i].argmax()]) for i in keep]
        by[L] = dict(pointing=float(np.mean(pt)), cls_auroc=loc.cls_auroc(z["cls"][:, j, :].astype(np.float32), y, pid)[0])
    m["by_layer"] = by
    m["peak_pointing_layer"] = max(by, key=lambda L: by[L]["pointing"])
    return m


def baseline_metrics(hu, masks, ispos, topk):
    gB = BASE_GRID
    H = np.stack([loc.patch_max(h, gB) for h in hu])
    MB = np.stack([loc.patch_mask(m, gB) for m in masks[ispos]]).reshape(-1, gB, gB)
    out = {}
    mh = loc.detector_metrics(H[ispos].reshape(-1, gB, gB), MB, H[~ispos].reshape(-1, gB, gB), loc.TAUS_HU)
    mh["topk_coverage"] = loc.topk_coverage(H[ispos], MB, topk)
    out["HU>130"] = mh
    U = np.random.default_rng(0).random((len(hu), gB * gB)).astype(np.float32)
    mu = loc.detector_metrics(U[ispos].reshape(-1, gB, gB), MB, U[~ispos].reshape(-1, gB, gB), loc.TAUS_UNI)
    mu["topk_coverage"] = loc.topk_coverage(U[ispos], MB, topk)
    out["uniform"] = mu
    for v in out.values():
        v.update(grid=gB, cls_auroc=float("nan"))
    return out


def write_report(res, path):
    P = res["pool"]
    L = ["=" * 100, "Calcium localization on gated Stanford COCA (frozen encoders, class-token attention)", "=" * 100,
         f"pool: {P['n_slices']} slices ({P['n_pos']} calcium-bearing, {P['n_neg']} calcium-free) from "
         f"{P['n_patients']} patients, split side '{P['side']}'", ""]
    cov = f"top-{res['topk']} cov"
    hdr = (f"  {'encoder':<11}{'L':>4}{'grid':>7}{'CLS AUROC':>11}{'attn AUROC':>12}{'pointing':>10}"
           f"{'lesion rec':>12}{cov:>12}{'FA@sens.8':>11}")
    L += [hdr, "  " + "-" * (len(hdr) - 2)]
    rows = [(DISPLAY.get(k, k), v) for k, v in res["encoders"].items()] + list(res["baselines"].items())
    for k, m in rows:
        f = lambda x: "n/a" if x != x else f"{x:.3f}"
        L.append(f"  {k:<11}{('L%d' % m['layer']) if 'layer' in m else '-':>4}{'%dx%d' % (m['grid'], m['grid']):>7}"
                 f"{f(m['cls_auroc']):>11}{f(m['attn_auroc']):>12}{f(m['pointing']):>10}{f(m['lesion_recall']):>12}"
                 f"{f(m['topk_coverage']['mean']):>12}{f(m['false_alarm_at_sens08']):>11}")
    L += ["", "pointing / CLS AUROC by layer"]
    for k, m in res["encoders"].items():
        L.append(f"  {DISPLAY.get(k, k):<11} pointing " + " ".join(f"L{int(Lq)}:{v['pointing']:.3f}" for Lq, v in m["by_layer"].items())
                 + f"   peak L{m['peak_pointing_layer']}")
        L.append(f"  {'':<11} CLS      " + " ".join(f"L{int(Lq)}:{v['cls_auroc']:.3f}" for Lq, v in m["by_layer"].items()))
    L += ["", "The HU>130 baseline scores each patch by its peak HU; the uniform baseline is a constant map with a",
          "fixed random tie-break. A coarser grid makes a pointing hit easier, so compare encoders on one grid."]
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def plot_layers(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, m in res["encoders"].items():
        Ls = [int(L) for L in m["by_layer"]]
        ax[0].plot(Ls, [m["by_layer"][L]["pointing"] for L in m["by_layer"]], marker="o", label=DISPLAY.get(k, k))
        ax[1].plot(Ls, [m["by_layer"][L]["cls_auroc"] for L in m["by_layer"]], marker="o", label=DISPLAY.get(k, k))
    for x, t in zip(ax, ("pointing game", "CLS-to-CAC AUROC (5-fold, patient-grouped)")):
        x.set_xlabel("transformer layer"); x.set_ylabel(t); x.grid(alpha=0.3); x.legend()
        x.set_xticks(Ls)
    ax[0].set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split-csv", default="splits/coca_gated_split.csv")
    ap.add_argument("--side", default="test", choices=["test", "train", "all"])
    ap.add_argument("--encoder", action="append", default=[], metavar="NAME=WEIGHTS[@LAYER]",
                    help="repeatable; NAME is curia or meddinov3; LAYER defaults to coca_cac/encoders.py LAYER")
    ap.add_argument("--dinov3-repo", default=None)
    ap.add_argument("--out", default="outputs/pointing")
    ap.add_argument("--stage", default="all", choices=["pool", "encode", "metrics", "all"])
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8,
                    help="CPU threads of the metrics stage (unbounded BLAS / OpenMP threads oversubscribe the CPU)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    encs = []
    for spec in a.encoder:
        name, _, rest = spec.partition("=")
        weights, _, lay = rest.partition("@")
        if name not in LAYER or not weights:
            ap.error(f"--encoder {spec!r}: expected NAME=WEIGHTS[@LAYER] with NAME in {sorted(LAYER)}")
        encs.append((name, weights, int(lay) if lay else LAYER[name]))

    pool_path = os.path.join(a.out, f"pool_{a.side}.npz")
    if a.stage in ("pool", "all") and not os.path.exists(pool_path):
        build_pool(a, pool_path)
    if a.stage == "pool":
        return
    pool = load_pool(pool_path)
    hu, masks, pid, _, ispos = pool
    for name, weights, _ in encs:
        p = os.path.join(a.out, f"enc_{a.side}_{name}.npz")
        if a.stage in ("encode", "all") and not os.path.exists(p):
            encode(a, name, weights, pool, p)
    if a.stage == "encode":
        return
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(a.threads)
    t0 = time.time()
    with threadpool_limits(limits=a.threads):
        res = dict(topk=a.topk, pool=dict(side=a.side, n_slices=int(len(ispos)), n_pos=int(ispos.sum()),
                                          n_neg=int((~ispos).sum()), n_patients=int(len(np.unique(pid)))),
                   encoders={}, baselines=baseline_metrics(hu, masks, ispos, a.topk))
        for name, _, layer in encs:
            z = np.load(os.path.join(a.out, f"enc_{a.side}_{name}.npz"))
            res["encoders"][name] = encoder_metrics(z, ispos, pid, layer, a.topk)
    print(f"metrics in {time.time() - t0:.0f} s", flush=True)
    json.dump(res, open(os.path.join(a.out, f"metrics_{a.side}.json"), "w"), indent=1)
    write_report(res, os.path.join(a.out, f"report_{a.side}.txt"))
    if res["encoders"]:
        plot_layers(res, os.path.join(a.out, f"layers_{a.side}.png"))
    print("->", a.out)


if __name__ == "__main__":
    main()
