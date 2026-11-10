"""Scoring metrics, computed per patient.

  CAC-DRS bands (<=10, 11-100, 101-400, >400): accuracy, linearly weighted Cohen's kappa, per-band
  sensitivity / specificity / precision / F1, confusion matrix.
  Presence: AUROC of the continuous prediction for a reference score > 10.
  Errors: MAE and RMSE in Agatston units, RMSE of log(1 + score).
  Continuous agreement on log(1 + score): ICC(A,1) (two-way, absolute agreement, single measurement),
  R^2, Bland-Altman bias and 95% limits of agreement (prediction minus reference).
  95% intervals for kappa and ICC: non-parametric patient-level bootstrap, 20,000 resamples, central
  (percentile) interval; random generator seeds 3 (kappa) and 0 (ICC), the seeds of the accompanying
  study, so that its intervals are reproduced.
"""
import json

import numpy as np

from .agatston import CAC_DRS, cac_drs

AUC_THRESHOLD = 10.0
N_BOOT, KAPPA_SEED, ICC_SEED = 20000, 3, 0


def _band_metrics(cm, i):
    TP = cm[i, i]; FP = cm[:, i].sum() - TP; FN = cm[i, :].sum() - TP; TN = cm.sum() - TP - FP - FN
    sens = TP / (TP + FN) if TP + FN else 0.0
    spec = TN / (TN + FP) if TN + FP else 0.0
    ppv = TP / (TP + FP) if TP + FP else 0.0
    f1 = 2 * ppv * sens / (ppv + sens) if ppv + sens else 0.0
    return dict(sensitivity=float(sens), specificity=float(spec), precision=float(ppv), f1=float(f1),
                support=int(TP + FN))


def kappa(gt, pred):
    from sklearn.metrics import cohen_kappa_score
    gi = [CAC_DRS.index(cac_drs(s)) for s in gt]
    pi = [CAC_DRS.index(cac_drs(s)) for s in pred]
    return float(cohen_kappa_score(gi, pi, weights="linear")) if len(set(gi)) > 1 else 0.0


def icc_a1(gt, pred):
    """ICC(A,1) on log(1 + score)."""
    X = np.c_[np.log1p(gt), np.log1p(pred)]; n, k = X.shape; gm = X.mean()
    ssr = k * ((X.mean(1) - gm) ** 2).sum(); ssc = n * ((X.mean(0) - gm) ** 2).sum()
    sse = ((X - gm) ** 2).sum() - ssr - ssc
    msr = ssr / (n - 1); msc = ssc / (k - 1); mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + (k / n) * (msc - mse)))


def bootstrap_ci(fn, gt, pred, seed, n_boot=N_BOOT):
    rng = np.random.default_rng(seed); n = len(gt); out = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.integers(0, n, n)
        out[b] = fn(gt[ix], pred[ix])
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return [float(lo), float(hi)]


def evaluate(gt, pred, with_ci=True):
    from sklearn.metrics import confusion_matrix, roc_auc_score
    gt = np.asarray(gt, float); pred = np.clip(np.asarray(pred, float), 0, None)
    gi = [CAC_DRS.index(cac_drs(s)) for s in gt]
    pi = [CAC_DRS.index(cac_drs(s)) for s in pred]
    cm = confusion_matrix(gi, pi, labels=range(4))
    pos = gt > AUC_THRESHOLD
    auc = float(roc_auc_score(pos.astype(int), pred)) if len(set(pos)) > 1 else float("nan")
    err = gt - pred
    lg, lp = np.log1p(gt), np.log1p(pred)
    d = lp - lg
    res = dict(n=int(len(gt)), accuracy=float(np.trace(cm) / cm.sum()), kappa=kappa(gt, pred), auc_gt10=auc,
               MAE=float(np.abs(err).mean()), RMSE=float(np.sqrt((err ** 2).mean())),
               logRMSE=float(np.sqrt((d ** 2).mean())),
               icc=icc_a1(gt, pred), r2=float(np.corrcoef(lg, lp)[0, 1] ** 2),
               bias=float(d.mean()), loa=[float(d.mean() - 1.96 * d.std(ddof=1)), float(d.mean() + 1.96 * d.std(ddof=1))],
               confusion=cm.tolist(), bands={})
    for i, c in enumerate(CAC_DRS):
        m = _band_metrics(cm, i)
        sel = np.array(gi) == i
        m["MAE"] = float(np.abs(err[sel]).mean()) if sel.any() else float("nan")
        res["bands"][c] = m
    res["macro_f1"] = float(np.mean([res["bands"][c]["f1"] for c in CAC_DRS]))
    if with_ci:
        res["kappa_ci95"] = bootstrap_ci(kappa, gt, pred, KAPPA_SEED)
        res["icc_ci95"] = bootstrap_ci(icc_a1, gt, pred, ICC_SEED)
    return res


def report(res, title):
    L = ["=" * 92, title, "=" * 92]
    ci = lambda k: (f" [{res[k + '_ci95'][0]:.3f}, {res[k + '_ci95'][1]:.3f}]" if k + "_ci95" in res else "")
    L.append(f"patients {res['n']}   kappa {res['kappa']:.3f}{ci('kappa')}   accuracy {res['accuracy']:.3f}   "
             f"AUROC(>10) {res['auc_gt10']:.3f}   macro-F1 {res['macro_f1']:.3f}")
    L.append(f"MAE {res['MAE']:.1f}   RMSE {res['RMSE']:.1f}   logRMSE {res['logRMSE']:.3f}   "
             f"ICC(A,1) {res['icc']:.3f}{ci('icc')}   R2 {res['r2']:.3f}   "
             f"bias {res['bias']:+.2f}  LoA [{res['loa'][0]:.2f}, {res['loa'][1]:.2f}]")
    L.append(f"  {'band':<9}{'n':>5}{'sens':>8}{'spec':>8}{'prec':>8}{'F1':>8}{'MAE':>9}")
    for c in CAC_DRS:
        m = res["bands"][c]
        L.append(f"  {c:<9}{m['support']:>5}{m['sensitivity']:>8.3f}{m['specificity']:>8.3f}"
                 f"{m['precision']:>8.3f}{m['f1']:>8.3f}{m['MAE']:>9.1f}")
    L.append("  confusion (rows = reference, columns = prediction): " + "  ".join(CAC_DRS))
    for i, c in enumerate(CAC_DRS):
        L.append(f"    {c:<9}" + "".join(f"{v:>7d}" for v in res["confusion"][i]))
    return "\n".join(L)


def write_predictions(path, pid, gt, pred):
    with open(path, "w") as f:
        f.write("patient_id,reference,prediction,reference_band,predicted_band\n")
        for p, g, y in zip(pid, gt, pred):
            f.write(f"{int(p)},{float(g):.4f},{float(y):.4f},{cac_drs(g)},{cac_drs(y)}\n")


def save(res, path):
    with open(path, "w") as f:
        json.dump(res, f, indent=2)


def plot(gt, pred, res, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gt = np.asarray(gt, float); pred = np.clip(np.asarray(pred, float), 0, None)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    cm = np.array(res["confusion"])
    ax[0].imshow(cm, cmap="Blues")
    ax[0].set_xticks(range(4)); ax[0].set_yticks(range(4))
    ax[0].set_xticklabels(CAC_DRS, rotation=45, ha="right"); ax[0].set_yticklabels(CAC_DRS)
    for a in range(4):
        for b in range(4):
            ax[0].text(b, a, int(cm[a, b]), ha="center", va="center", color="white" if cm[a, b] > cm.max() / 2 else "black")
    ax[0].set_xlabel("predicted band"); ax[0].set_ylabel("reference band")
    ax[0].set_title(f"kappa {res['kappa']:.3f}, accuracy {res['accuracy']:.3f}")
    ax[1].scatter(gt + 1, pred + 1, s=16, alpha=0.6, edgecolor="k", linewidth=0.3)
    mx = max(gt.max(), pred.max()) + 2
    ax[1].plot([1, mx], [1, mx], "k--", alpha=0.4)
    for t in (10, 100, 400):
        ax[1].axvline(t + 1, color="orange", ls=":", lw=1); ax[1].axhline(t + 1, color="orange", ls=":", lw=1)
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("reference Agatston + 1"); ax[1].set_ylabel("predicted Agatston + 1")
    ax[1].set_title(f"ICC(A,1) {res['icc']:.3f}, R2 {res['r2']:.3f}")
    lg, lp = np.log1p(gt), np.log1p(pred)
    ax[2].scatter((lg + lp) / 2, lp - lg, s=16, alpha=0.6, edgecolor="k", linewidth=0.3)
    ax[2].axhline(res["bias"], color="k"); ax[2].axhline(0, color="grey", ls=":")
    for v in res["loa"]:
        ax[2].axhline(v, color="k", ls="--")
    ax[2].set_xlabel("mean of log(1+score)"); ax[2].set_ylabel("prediction - reference, log(1+score)")
    ax[2].set_title(f"bias {res['bias']:+.2f}, LoA [{res['loa'][0]:.2f}, {res['loa'][1]:.2f}]")
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
