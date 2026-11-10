#!/usr/bin/env python
"""Train the additive scoring head on the TRAIN side of a split and evaluate it on the TEST side.

    python scripts/train_head.py --features outputs/features/curia_L7/coca_gated.npz \
        --split-csv splits/coca_gated_split.csv --out outputs/heads/curia_L7/coca_gated

Writes to --out: head.pt (weights, feature mean / sd, settings), test_predictions.csv (patient_id,
reference, prediction and their bands), metrics.json, report.txt, results.png, train.log. The test
patients are not read until the head is trained. Training settings: coca_cac/head.py.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coca_cac import metrics  # noqa: E402
from coca_cac.cohorts import read_split  # noqa: E402
from coca_cac.features import load_cache, select_patients  # noqa: E402
from coca_cac.head import DEFAULTS, fit_head, predict, save_head  # noqa: E402


def side(cache, split_rows, name):
    pids = [p for p, s, _ in split_rows if s == name]
    sub = select_patients(cache, pids)
    ref = {p: g for p, s, g in split_rows if s == name}
    missing = sorted(set(pids) - set(sub["vpid"].tolist()))
    if missing:
        print(f"  WARNING: {len(missing)} {name} patients have no features (unreadable series): {missing}")
    bad = [p for p, g in zip(sub["vpid"].tolist(), sub["vgt"].tolist()) if abs(g - np.float32(ref[p])) > 1e-3]
    if bad:
        raise SystemExit(f"reference scores of the cache and of the split file differ for {bad[:5]}")
    return sub


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--features", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--out", required=True)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v) if not isinstance(v, bool) else int, default=v)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lines = []

    def log(m):
        print(m, flush=True); lines.append(m)

    cache = load_cache(a.features)
    rows = read_split(a.split_csv)
    enc, layer, cohort = str(cache["meta_encoder"]), int(cache["meta_layer"]), str(cache["meta_cohort"])
    log(f"features {a.features}: encoder {enc} L{layer}, cohort {cohort}")
    tr, te = side(cache, rows, "train"), side(cache, rows, "test")
    log(f"train {len(tr['vpid'])} patients / {len(tr['svol'])} slices, test {len(te['vpid'])} patients / "
        f"{len(te['svol'])} slices")
    hp = {k: getattr(a, k) for k in DEFAULTS}
    hp["tail_weight"] = bool(hp["tail_weight"])
    log("settings: " + ", ".join(f"{k}={v}" for k, v in hp.items()))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mu, sd, cfg = fit_head(tr, dev, log, **hp)
    pred = predict(model, te, mu, sd, cfg["thickness_ref"], dev)
    save_head(os.path.join(a.out, "head.pt"), model, mu, sd, cfg,
              dict(encoder=enc, layer=layer, topk=int(cache["meta_topk"]), cohort=cohort,
                   features=a.features, split=a.split_csv))

    gt = te["vgt"].astype(np.float64)
    metrics.write_predictions(os.path.join(a.out, "test_predictions.csv"), te["vpid"], gt, pred)
    res = metrics.evaluate(gt, pred)
    res.update(encoder=enc, layer=layer, cohort=cohort, n_train=int(len(tr["vpid"])))
    metrics.save(res, os.path.join(a.out, "metrics.json"))
    title = f"COCA {cohort}, test side: head trained on the {cohort} train side, encoder {enc} L{layer}"
    txt = metrics.report(res, title)
    open(os.path.join(a.out, "report.txt"), "w").write(txt + "\n")
    metrics.plot(gt, pred, res, os.path.join(a.out, "results.png"), title)
    log(txt)
    open(os.path.join(a.out, "train.log"), "w").write("\n".join(lines) + "\n")
    print("->", a.out)


if __name__ == "__main__":
    main()
