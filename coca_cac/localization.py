"""Localization metrics of a frozen encoder's class-token attention against calcium masks.

All maps are on the encoder's patch grid; a patch is a calcium patch when it contains at least one
calcium pixel (max-pooling of the pixel mask). On calcium-bearing slices:

  pointing         fraction of slices whose highest-attention patch is a calcium patch (raw argmax)
  lesion recall    with n = number of lesions (8-connected components of the patch mask) on the slice,
                   the fraction of lesions hit by one of the n highest-attention patches
  attn AUROC       patch-level AUROC of the attention against the patch mask, over all positive slices
  top-K coverage   fraction of a slice's calcium patches among its K highest-attention patches (the
                   scoring head's patch budget), averaged over slices
  false alarm      thresholding the attention map, a lesion counts as detected when a connected
                   candidate component touches it; the false-alarm rate is the fraction of calcium-FREE
                   slices with at least one candidate, read at lesion sensitivity 0.8 (linear
                   interpolation over the threshold sweep)
  CLS AUROC        slice-level calcium detection from the class token: logistic regression,
                   standardised features, balanced classes, 5-fold patient-grouped stratified
                   cross-validation (seed 42), mean held-out AUROC over the folds

Baselines on the same protocol: the peak HU of each patch (HU > 130) and a uniform map with a fixed
random tie-break.
"""
import numpy as np
from scipy.ndimage import label as cc_label

STRUCT = np.ones((3, 3), int)                    # 8-connectivity
TAUS_ATTN = np.logspace(np.log10(0.3), np.log10(400), 44)    # in units of the uniform attention 1/P
TAUS_HU = np.unique(np.r_[np.geomspace(60.0, 3000.0, 33), 130.0])
TAUS_UNI = np.linspace(0.001, 0.999, 34)


def patch_mask(mask, grid):
    """(H, W) bool pixel mask -> (grid*grid,) bool patch mask (a patch with any calcium pixel)."""
    import torch
    import torch.nn.functional as F
    k = max(1, mask.shape[-1] // grid)
    m = torch.from_numpy(np.ascontiguousarray(mask)).float()[None, None]
    return (F.max_pool2d(m, kernel_size=k, stride=k).flatten() > 0).numpy()


def patch_max(img, grid):
    """(H, W) -> (grid*grid,) per-patch maximum (the HU baseline)."""
    import torch
    import torch.nn.functional as F
    k = max(1, img.shape[-1] // grid)
    return F.max_pool2d(torch.from_numpy(img.astype(np.float32))[None, None], kernel_size=k, stride=k).flatten().numpy()


def lesions(mask):
    lab, n = cc_label(mask > 0, STRUCT)
    return [lab == k for k in range(1, n + 1)]


def matched_negatives(pid, sidx, pos):
    """Every calcium-bearing slice, plus, per patient, as many calcium-free slices as it has
    calcium-bearing ones, nearest in slice index (ties: lower index). Each patient then contributes both
    classes in equal number at the same anatomical level, so neither patient identity nor slice level
    predicts the label. -> (sorted kept indices, number of negatives patients lacked)."""
    keep, short = list(np.flatnonzero(pos)), 0
    for p in np.unique(pid[pos]):
        P = np.flatnonzero((pid == p) & pos); N = np.flatnonzero((pid == p) & ~pos)
        d = np.abs(sidx[N][:, None] - sidx[P][None, :]).min(1)
        take = N[np.lexsort((sidx[N], d))[:len(P)]]
        short += len(P) - len(take); keep += list(take)
    return np.sort(np.array(keep)), short


def detector_metrics(SP, MP, SN, taus):
    """SP (npos, g, g) score maps of calcium-bearing slices, MP (npos, g, g) their patch masks, SN
    (nneg, g, g) score maps of calcium-free slices; `taus` thresholds in the unit of the maps."""
    from sklearn.metrics import roc_auc_score
    keep = np.array([i for i in range(len(MP)) if MP[i].any()])
    SP, MP = SP[keep], MP[keep]
    tot = budget = 0; pointing = []
    for A, M in zip(SP, MP):
        les = lesions(M)
        nl = len(les); tot += nl
        flat = A.reshape(-1)
        pointing.append(bool((M > 0).reshape(-1)[flat.argmax()]))
        topn = set(np.argsort(-flat)[:nl].tolist())
        for comp in les:
            if topn & set(np.flatnonzero(comp.reshape(-1)).tolist()):
                budget += 1
    attn_auroc = float(roc_auc_score((MP > 0).reshape(-1), SP.reshape(-1).astype(np.float64)))
    sens, negfire = [], []
    for tau in taus:
        det = nf = 0
        for A, M in zip(SP, MP):
            lab, _ = cc_label(A > tau, STRUCT)
            for comp in lesions(M):
                if np.any(lab[comp] > 0):
                    det += 1
        for A in SN:
            _, nd = cc_label(A > tau, STRUCT)
            nf += int(nd > 0)
        sens.append(det / max(tot, 1)); negfire.append(nf / max(len(SN), 1))
    return dict(n_pos=int(len(SP)), n_neg=int(len(SN)), n_lesions=int(tot),
                pointing=float(np.mean(pointing)), lesion_recall=budget / max(tot, 1), attn_auroc=attn_auroc,
                false_alarm_at_sens08=at_sens(sens, negfire, 0.8),
                sweep=dict(taus=[float(t) for t in taus], sens=sens, negfire=negfire))


def at_sens(sens, values, s):
    """`values` at lesion sensitivity s (linear interpolation); nan if the sweep never reaches s."""
    se = np.array(sens); v = np.array(values); o = np.argsort(se)
    return float("nan") if se.max() < s else float(np.interp(s, se[o], v[o]))


def cls_auroc(X, y, pid):
    """5-fold patient-grouped logistic probe on the class token -> (mean, sd) of the held-out AUROC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler
    aucs = []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, pid):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def topk_coverage(A_pos, M, k):
    """A_pos (n, g*g) attention of calcium-bearing slices, M (n, g, g) patch masks -> coverage summary."""
    Mf = M.reshape(len(M), -1) > 0
    ncal = Mf.sum(1)
    keep = np.flatnonzero(ncal > 0)
    cov = np.empty(len(keep))
    for j, i in enumerate(keep):
        top = np.argpartition(-A_pos[i], k - 1)[:k]
        cov[j] = np.isin(top, np.flatnonzero(Mf[i])).sum() / ncal[i]
    return dict(k=int(k), n_slices=int(len(keep)), mean=float(cov.mean()), median=float(np.median(cov)),
                frac_full=float((cov >= 1.0).mean()), max_calcium_patches=int(ncal[keep].max()))
