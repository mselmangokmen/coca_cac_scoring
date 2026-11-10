"""The additive (DeepSets) scoring head and its training.

For slice i with top-K patches k = 1..K (features f_ik, renormalised attention a_ik):

    z_ik = phi([f_ik | s_x | s_y | a_ik])  >= 0        phi: MLP (d+3 -> 256 -> 256 -> 1), softplus output
    s_i  = sum_k a_ik z_ik                             slice score
    y    = (t / 3) sum_i s_i                           volume prediction, t = slice thickness in mm

so the prediction decomposes exactly into the non-negative patch contributions c_ik = (t/3) a_ik z_ik.
Features are standardised with the training set's mean / sd. The only supervision is the volume-level
Agatston score: Huber loss (delta 1) on log(1 + y), each volume weighted by 1 + log(1 + y_ref).
AdamW (lr 1e-3, weight decay 1e-4), cosine schedule, 16 volumes per batch, 100 epochs, seed 42,
no early stopping.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULTS = dict(hidden=256, dropout=0.1, epochs=100, lr=1e-3, batch_volumes=16, thickness_ref=3.0,
                tail_weight=True, seed=42)


class DeepSets(nn.Module):
    def __init__(self, d, hidden=256, p=0.1):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(d + 3, hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1), nn.Softplus())

    def patch_scores(self, feat, attn, sx, sy):
        """feat (M, K, D), attn (M, K), sx / sy (M,) -> z (M, K) >= 0."""
        M, K, _ = feat.shape
        chans = [feat, sx[:, None, None].expand(M, K, 1), sy[:, None, None].expand(M, K, 1), attn[:, :, None]]
        return self.phi(torch.cat(chans, dim=-1)).squeeze(-1)

    def slice_raw(self, feat, attn, sx, sy):
        return (self.patch_scores(feat, attn, sx, sy) * attn).sum(1)    # (M,) slice scores s_i


def to_device(cache, device):
    """Feature cache -> GPU tensors and the slice rows of every volume."""
    svol = torch.from_numpy(cache["svol"].astype(np.int64))
    vsx = torch.from_numpy(cache["vsx"].astype(np.float32))
    vsy = torch.from_numpy(cache["vsy"].astype(np.float32))
    nvol = len(cache["vgt"])
    return dict(FT=torch.from_numpy(cache["ftopk"]).to(device), AT=torch.from_numpy(cache["atopk"]).to(device),
                sx=vsx[svol].to(device), sy=vsy[svol].to(device),
                vth=torch.from_numpy(cache["vth"].astype(np.float32)).to(device),
                vgt=torch.from_numpy(cache["vgt"].astype(np.float32)).to(device), nvol=nvol,
                rows=[torch.where(svol == v)[0].to(device) for v in range(nvol)])


def seg_sum(x, counts):
    """Per-volume sum of contiguous slice rows through a prefix sum: deterministic, unlike the atomic
    `index_add_`, whose run-to-run float drift compounds over training."""
    cs = torch.cumsum(x, 0)
    ends = torch.cumsum(counts, 0) - 1
    tot = cs[ends]
    return tot - torch.cat([tot.new_zeros(1), tot[:-1]])


def forward_volumes(model, G, vol_ids, mu, sd, thick_ref, device):
    idx = torch.cat([G["rows"][v] for v in vol_ids])
    counts = torch.tensor([G["rows"][v].numel() for v in vol_ids], device=device)
    feat = (G["FT"][idx].float() - mu) / sd
    raw = model.slice_raw(feat, G["AT"][idx].float(), G["sx"][idx], G["sy"][idx])
    thick = G["vth"][torch.as_tensor(vol_ids, device=device)]
    return seg_sum(raw, counts) * (thick / thick_ref)


def fit_head(cache, device, log=print, **kw):
    """Train one head on every volume of `cache` -> (model, mu, sd, cfg)."""
    cfg = dict(DEFAULTS, **kw)
    G = to_device(cache, device)
    D = G["FT"].shape[-1]
    torch.manual_seed(int(cfg["seed"])); np.random.seed(int(cfg["seed"]))
    N = G["FT"].shape[0]                                         # mu / sd from a feature subsample
    sub = torch.randint(0, N, (min(50000, N),), device=G["FT"].device)
    pf = G["FT"][sub].float().reshape(-1, D)
    mu, sd = pf.mean(0), pf.std(0) + 1e-6
    del pf

    model = DeepSets(D, hidden=int(cfg["hidden"]), p=float(cfg["dropout"])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(cfg["epochs"]))
    gt = G["vgt"]
    w = (1.0 + torch.log1p(gt)) if cfg["tail_weight"] else torch.ones(len(gt), device=device)
    B, nvol, ids = int(cfg["batch_volumes"]), G["nvol"], np.arange(G["nvol"])
    log(f"training the head: {nvol} volumes, {int(cfg['epochs'])} epochs, {B} volumes per batch")
    for ep in range(int(cfg["epochs"])):
        model.train()
        order, tot = ids[np.random.permutation(nvol)], 0.0
        for j in range(0, nvol, B):
            vb = order[j:j + B].tolist()
            pred = forward_volumes(model, G, vb, mu, sd, float(cfg["thickness_ref"]), device)
            gv = torch.as_tensor(vb, device=device)
            loss = (F.huber_loss(torch.log1p(pred), torch.log1p(gt[gv]), reduction="none", delta=1.0) * w[gv]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(vb)
        sched.step()
        if ep % 10 == 0 or ep == int(cfg["epochs"]) - 1:
            log(f"  epoch {ep + 1:3d}/{int(cfg['epochs'])}  loss {tot / nvol:.4f}")
    return model, mu, sd, cfg


@torch.no_grad()
def predict(model, cache, mu, sd, thickness_ref, device):
    """Predicted Agatston score of every volume of `cache` (clipped at 0)."""
    model.eval()
    G = to_device(cache, device)
    out = [forward_volumes(model, G, list(range(j, min(j + 64, G["nvol"]))), mu, sd, thickness_ref, device).cpu().numpy()
           for j in range(0, G["nvol"], 64)]
    return np.clip(np.concatenate(out), 0, None)


@torch.no_grad()
def contributions(model, cache, mu, sd, thickness_ref, device):
    """Per-patch contributions c_ik = (t/3) a_ik z_ik of every slice of `cache`: (N_slices, K)."""
    model.eval()
    G = to_device(cache, device)
    z = model.patch_scores((G["FT"].float() - mu) / sd, G["AT"].float(), G["sx"], G["sy"])
    t = G["vth"][torch.from_numpy(cache["svol"].astype(np.int64)).to(device)] / thickness_ref
    return (z * G["AT"].float() * t[:, None]).cpu().numpy()


def save_head(path, model, mu, sd, cfg, meta):
    torch.save({"state_dict": model.state_dict(), "mu": mu.cpu(), "sd": sd.cpu(), "cfg": cfg, "meta": meta}, path)


def load_head(path, d, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = DeepSets(d, hidden=int(ck["cfg"]["hidden"]), p=0.0).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model, ck["mu"].to(device), ck["sd"].to(device), ck["cfg"], ck["meta"]
