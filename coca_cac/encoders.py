"""Frozen public encoders behind one interface: per-layer class token, patch tokens and the
class-token -> patch attention.

Two encoders are supported, both with their PUBLISHED weights and no adaptation of any kind:

  curia      Curia (Raidium), a DINOv2 ViT-B/16 radiology foundation model, single-channel input.
             Hugging Face: raidium/curia (gated: accept the licence on the model page, then
             `hf download raidium/curia config.json model.safetensors --local-dir weights/curia`).
             Loaded with transformers' AutoModel (Dinov2Model), eager attention.
  meddinov3  MedDINOv3 (ViT-B/16, CT-3M), a DINOv3 model with 4 storage tokens, 3-channel input (the CT
             slice is repeated). Hugging Face: ricklisz123/MedDINOv3-ViTB-16-CT-3M (model.pth). Needs
             the DINOv3 model code: `git clone https://github.com/facebookresearch/dinov3` and pass
             --dinov3-repo (or set DINOV3_REPO).

Input: (B, 1, 512, 512) float32, each slice min-max normalised to [-1, 1] (features.normalize_slice).
Both encoders use 16-px patches, i.e. a 32 x 32 patch grid.

Layer L means the output of the L-th transformer block (1-based, before any final normalisation), and
its attention is that block's attention; the mean over heads of the class-token row, restricted to the
patch tokens, is the attention map used for patch selection and for localization.

`LAYER` holds the layer each encoder is read at for scoring. These are the layers at which the
encoder's attention points at calcium best (the pointing peak over all 12 layers) in the accompanying
study; scripts/pointing.py reports every layer, so the choice can be re-made on COCA.
"""
import os
import sys

import torch

LAYER = {"curia": 7, "meddinov3": 11}
DISPLAY = {"curia": "Curia", "meddinov3": "MedDINOv3"}
HF_SOURCE = {"curia": "raidium/curia", "meddinov3": "ricklisz123/MedDINOv3-ViTB-16-CT-3M"}

# MedDINOv3 ViT-B/16 construction (strict load of its teacher backbone: 0 missing / 0 unexpected keys)
MEDDINOV3_KW = dict(
    patch_size=16, in_chans=3,
    pos_embed_rope_base=100, pos_embed_rope_normalize_coords="separate",
    pos_embed_rope_rescale_coords=2, pos_embed_rope_dtype="fp32",
    qkv_bias=False, mask_k_bias=False, layerscale_init=1.0e-05,
    norm_layer="layernormbf16", ffn_layer="mlp", ffn_bias=True, proj_bias=True,
    n_storage_tokens=4,
)


class FrozenEncoder:
    """`forward(x, layers)` -> {L: (cls (B, D), patches (B, P, D), attn (B, heads, P))}."""

    def __init__(self, name, weights, device, dinov3_repo=None):
        if name not in LAYER:
            raise ValueError(f"unknown encoder {name!r}; supported: {sorted(LAYER)}")
        self.name, self.device = name, device
        if name == "curia":
            self._load_hf(weights)
        else:
            self._load_meddinov3(weights, dinov3_repo)
        with torch.no_grad():
            out = self.forward(torch.zeros(1, 1, 512, 512, device=device), [self.n_layers])
        self.grid = int(round(out[self.n_layers][1].shape[1] ** 0.5))

    # ---------------------------------------------------------------- loading
    def _load_hf(self, weights):
        from transformers import AutoModel
        # eager attention: the SDPA / flash implementations return no attention weights
        self.m = AutoModel.from_pretrained(weights, attn_implementation="eager").to(self.device).eval()
        c = self.m.config
        self.n_layers, self.dim = int(c.num_hidden_layers), int(c.hidden_size)
        self.in_ch = int(getattr(c, "num_channels", 1))
        self.n_reg = int(getattr(c, "num_register_tokens", 0) or 0)
        self.kind = "hf"

    def _load_meddinov3(self, weights, dinov3_repo):
        repo = dinov3_repo or os.environ.get("DINOV3_REPO")
        if repo:
            sys.path.insert(0, os.path.abspath(repo))
        try:
            import dinov3.models.vision_transformer as vits
        except ImportError as e:
            raise SystemExit("MedDINOv3 needs the DINOv3 model code: git clone "
                             "https://github.com/facebookresearch/dinov3 and pass --dinov3-repo <clone>") from e
        path = os.path.join(weights, "model.pth") if os.path.isdir(weights) else weights
        vit = vits.vit_base(img_size=512, **MEDDINOV3_KW).eval()
        raw = torch.load(path, map_location="cpu", weights_only=False)
        tw = raw["teacher"] if isinstance(raw, dict) and "teacher" in raw else raw
        sd = {k[len("backbone."):]: v for k, v in tw.items() if k.startswith("backbone.") and "head" not in k}
        vit.load_state_dict(sd, strict=True)
        self.m = vit.to(self.device).eval()
        self.n_layers, self.dim = len(vit.blocks), int(vit.embed_dim)
        self.in_ch = int(vit.patch_embed.proj.in_channels)
        self.n_reg = int(vit.n_storage_tokens)
        self.kind = "dinov3"

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, x, layers):
        if self.in_ch > 1 and x.shape[1] == 1:
            x = x.repeat(1, self.in_ch, 1, 1)
        r = 1 + self.n_reg                                     # [CLS | registers | patches]
        if self.kind == "hf":
            out = self.m(x, output_hidden_states=True, output_attentions=True)
            hs, at = out.hidden_states, out.attentions
            return {L: (hs[L][:, 0], hs[L][:, r:], at[L - 1][:, :, 0, r:]) for L in layers}
        # DINOv3 runs fused scaled-dot-product attention, which returns no weights: the call is wrapped
        # and the class-token row of softmax(q k^T / sqrt(d)) is recomputed from the same q and k.
        cap = []
        orig = torch.nn.functional.scaled_dot_product_attention

        def patched(q, k, v, *a, **kw):
            A = torch.softmax((q.float() @ k.float().transpose(-2, -1)) * (q.shape[-1] ** -0.5), dim=-1)
            cap.append(A[:, :, 0, r:])
            return orig(q, k, v, *a, **kw)

        torch.nn.functional.scaled_dot_product_attention = patched
        try:
            outs = self.m.get_intermediate_layers(x, n=list(range(self.n_layers)), reshape=False,
                                                  norm=False, return_class_token=True)
        finally:
            torch.nn.functional.scaled_dot_product_attention = orig
        # get_intermediate_layers drops the storage tokens: outs[i] = (patches, cls) after block i+1
        return {L: (outs[L - 1][1], outs[L - 1][0], cap[L - 1]) for L in layers}
