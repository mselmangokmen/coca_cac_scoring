"""Top-K patch features: the frozen input of the scoring head.

For every slice, the K patches with the highest class-token attention at the encoder's layer are kept,
with their features and their attention values renormalised by a softmax over those K. A feature cache
holds every slice of every patient of one cohort, concatenated patient by patient in patient-id order:

    ftopk (N_slices, K, D) float16    patch features
    atopk (N_slices, K)    float16    softmax-renormalised attention of those patches
    tidx  (N_slices, K)    int16      their patch indices on the grid (row-major), for contribution maps
    svol  (N_slices,)      int32      which patient (row of the per-patient arrays) a slice belongs to
    vpid, vgt, vth, vsx, vsy, vnsl    per patient: id, reference score, slice thickness (mm),
                                      pixel spacing x / y (mm), number of slices
"""
import numpy as np
import torch
import torch.nn.functional as F

TOPK = 32


def normalize_slice(hu):
    """Per-slice min-max normalisation of a HU slice to [-1, 1] (no HU window, no clipping)."""
    lo, hi = float(hu.min()), float(hu.max())
    if hi - lo < 1e-6:
        return np.zeros_like(hu, dtype=np.float32)
    return (((hu - lo) / (hi - lo)) * 2.0 - 1.0).astype(np.float32)


def to_input(hu_slices, device):
    """(n, H, W) HU -> (n, 1, 512, 512) normalised float32 tensor (bilinear resize if not 512)."""
    x = np.stack([normalize_slice(s.astype(np.float32)) for s in hu_slices])[:, None]
    x = torch.from_numpy(x).float().to(device)
    if x.shape[-2:] != (512, 512):
        x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
    return x


@torch.no_grad()
def topk_features(encoder, x, layer, k=TOPK):
    """(b, 1, 512, 512) -> (features (b, K, D) fp16, attention (b, K) fp16, patch index (b, K) int16)."""
    _, feat, att = encoder.forward(x, [layer])[layer]
    att = att.mean(1)                                            # mean over heads
    tv, ti = torch.topk(att, min(k, feat.shape[1]), dim=1)
    fk = torch.gather(feat, 1, ti[:, :, None].expand(-1, -1, feat.shape[-1]))
    return (fk.half().cpu().numpy(), torch.softmax(tv, dim=1).half().cpu().numpy(),
            ti.to(torch.int16).cpu().numpy())


def volume_features(encoder, hu, layer, device, k=TOPK, batch=8):
    """All slices of one volume, `batch` slices per forward."""
    fs, as_, ts = [], [], []
    for j in range(0, len(hu), batch):
        f, a, t = topk_features(encoder, to_input(hu[j:j + batch], device), layer, k)
        fs.append(f); as_.append(a); ts.append(t)
    return np.concatenate(fs), np.concatenate(as_), np.concatenate(ts)


def load_cache(path):
    return dict(np.load(path, allow_pickle=False))


def select_patients(cache, pids):
    """The sub-cache of the patients `pids`, in the cache's own (patient-id) order."""
    want = set(int(p) for p in pids)
    keep = np.array([int(p) in want for p in cache["vpid"]])
    rows = keep[cache["svol"]]
    new_id = np.cumsum(keep) - 1
    out = {k: cache[k][rows] for k in ("ftopk", "atopk", "tidx")}
    out["svol"] = new_id[cache["svol"][rows]].astype(np.int32)
    for k in ("vpid", "vgt", "vth", "vsx", "vsy", "vnsl"):
        out[k] = cache[k][keep]
    for k in cache:
        if k.startswith("meta_"):
            out[k] = cache[k]
    return out
