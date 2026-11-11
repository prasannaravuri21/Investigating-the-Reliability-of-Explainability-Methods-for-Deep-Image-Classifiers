"""Week 5: the five attribution methods, behind one function.

    compute_attribution(method, wrapper, x, target, cfg) -> (B, H, W) tensor

Each map is returned at input resolution so ROAD and IDSDS can treat all five
identically. The five methods were chosen to span the main families:

    gradcam               gradient-based, coarse, architecture-dependent
    integrated_gradients  complete (attributions sum to the output difference)
    gradient_shap         complete, sampling-based
    lrp                   relevance propagated backwards by per-layer rules
    rise                  black-box, random masking

Sign convention
---------------
Deletion metrics need a single importance ranking per pixel. Grad-CAM and RISE
are non-negative by construction. For the signed methods the raw channel sum is
kept by default, so that pixels the method says push *away* from the class rank
as least important rather than most. `postprocess` exposes this choice because
Gevaert et al. (2024) show it can move the scores.
"""
from __future__ import annotations

import time
from typing import Callable

import torch
import torch.nn.functional as F

from .models import ModelWrapper
from .utils import get_logger, gpu_throttle

LOG = get_logger()

METHODS = ["gradcam", "integrated_gradients", "lrp", "rise", "gradient_shap"]
BUILTIN = "bcos_builtin"

DEFAULT_POSTPROCESS = {
    "gradcam": "relu",
    "integrated_gradients": "sum",
    "gradient_shap": "sum",
    "lrp": "sum",
    "rise": "raw",
    BUILTIN: "sum",
}


def _postprocess(attr: torch.Tensor, mode: str) -> torch.Tensor:
    """(B, C, H, W) or (B, H, W) -> (B, H, W)."""
    if attr.dim() == 4:
        attr = attr.sum(dim=1) if mode in {"sum", "raw"} else attr
        if attr.dim() == 4:  # abs / relu applied per channel first
            attr = attr.abs().sum(dim=1) if mode == "abs" else F.relu(attr).sum(dim=1)
    elif mode == "abs":
        attr = attr.abs()
    elif mode == "relu":
        attr = F.relu(attr)
    return attr.float()


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
class GradCAM:
    """Grad-CAM with an optional reshape step for token-based models.

    Written out rather than taken from a library because the ViT needs the
    token sequence folded back into a 14x14 grid, and keeping that explicit
    makes the architecture-specific choice auditable (Section 3.5 risk).
    """

    def __init__(self, wrapper: ModelWrapper):
        self.wrapper = wrapper
        self.layer = wrapper.target_layer()
        self.reshape: Callable | None = wrapper.gradcam_reshape
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            self.layer.register_forward_hook(self._save_activation),
            self.layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inp, out):
        self._activations = out

    def _save_gradient(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0]

    def close(self):
        for h in self._handles:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __call__(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        was_grad = [p.requires_grad for p in self.wrapper.model.parameters()]
        for p in self.wrapper.model.parameters():
            p.requires_grad_(True)  # full backward hooks need a grad path
        try:
            x = x.clone().requires_grad_(True)
            logits = self.wrapper.forward(x)
            score = logits.gather(1, target.view(-1, 1)).sum()
            self.wrapper.model.zero_grad(set_to_none=True)
            score.backward()

            acts, grads = self._activations, self._gradients
            if acts is None or grads is None:
                raise RuntimeError("Grad-CAM hooks captured nothing; check gradcam_layer")
            if self.reshape is not None:
                acts, grads = self.reshape(acts), self.reshape(grads)

            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return cam.squeeze(1).detach()
        finally:
            for p, flag in zip(self.wrapper.model.parameters(), was_grad):
                p.requires_grad_(flag)


# --------------------------------------------------------------------------- #
# Captum-based methods
# --------------------------------------------------------------------------- #
def integrated_gradients(wrapper: ModelWrapper, x: torch.Tensor, target: torch.Tensor,
                         n_steps: int = 32, baseline: str = "zeros") -> torch.Tensor:
    from captum.attr import IntegratedGradients

    base = torch.zeros_like(x) if baseline == "zeros" else _blur(x)
    ig = IntegratedGradients(wrapper.forward)
    attr = ig.attribute(x, baselines=base, target=target, n_steps=n_steps,
                        internal_batch_size=max(1, x.shape[0]))
    return attr.detach()


def gradient_shap(wrapper: ModelWrapper, x: torch.Tensor, target: torch.Tensor,
                  n_samples: int = 16, stdevs: float = 0.09,
                  baselines: torch.Tensor | None = None,
                  max_forward_batch: int = 32) -> torch.Tensor:
    from captum.attr import GradientShap

    # SHAP needs a reference distribution, not a single point. A black image
    # plus a white one plus the batch mean is a cheap, documented stand-in for
    # sampling the training set; swap in real training images if compute allows.
    # Built from the whole batch, before chunking, so chunking cannot change it.
    if baselines is None:
        baselines = torch.cat(
            [torch.zeros_like(x[:1]), torch.ones_like(x[:1]), x.mean(dim=0, keepdim=True)], dim=0
        )
    gs = GradientShap(wrapper.forward)
    # Captum expands each image into n_samples noisy copies in one backward pass
    # (batch 8 x 16 samples = 128 images), which exhausts RAM on the B-cos
    # network. Attributions are per-image, so chunking the images is exact.
    chunk = max(1, max_forward_batch // max(1, n_samples))
    parts = [
        gs.attribute(x[i: i + chunk], baselines=baselines, target=target[i: i + chunk],
                     n_samples=n_samples, stdevs=stdevs).detach()
        for i in range(0, x.shape[0], chunk)
    ]
    return torch.cat(parts)


def _blur(x: torch.Tensor, kernel: int = 31, sigma: float = 10.0) -> torch.Tensor:
    from torchvision.transforms.functional import gaussian_blur

    return gaussian_blur(x, kernel_size=[kernel, kernel], sigma=[sigma, sigma])


# --------------------------------------------------------------------------- #
# LRP (zennit)
# --------------------------------------------------------------------------- #
def lrp(wrapper: ModelWrapper, x: torch.Tensor, target: torch.Tensor,
        composite: str = "epsilon_plus_flat") -> torch.Tensor:
    """LRP through zennit, on the normalised input.

    LRP is computed on the raw network, so the input here is already
    normalised; only the ranking of the resulting relevance is used, so working
    in normalised space is harmless.

    The ViT and the B-cos network are refused rather than fudged. There is no
    settled relevance rule for attention, and B-cos layers are dynamic-linear
    with bias-free normalisation, so silently applying the CNN composite would
    produce a number that looks like LRP but is not. Section 3.5 records this as
    a known limitation; the transformer-specific alternative is Chefer et al.
    (2021), and for B-cos the built-in explanation plays LRP's role.
    """
    if not wrapper.lrp_supported:
        raise NotImplementedError(
            f"LRP is not defined for {wrapper.name} with the CNN composites "
            f"({wrapper.notes}). Report this cell as not-applicable, or substitute "
            "an architecture-specific relevance method and say so explicitly."
        )
    from zennit.attribution import Gradient
    from zennit.composites import EpsilonGammaBox, EpsilonPlusFlat
    from zennit.torchvision import ResNetCanonizer, VGGCanonizer

    canonizers = []
    if "resnet" in wrapper.name:
        canonizers = [ResNetCanonizer()]
    elif "vgg" in wrapper.name:
        canonizers = [VGGCanonizer()]

    if composite == "epsilon_plus_flat":
        comp = EpsilonPlusFlat(canonizers=canonizers)
    elif composite == "epsilon_gamma_box":
        comp = EpsilonGammaBox(low=-3.0, high=3.0, canonizers=canonizers)
    else:
        raise ValueError(f"unknown composite {composite!r}")

    xn = wrapper.normalise(x).clone().requires_grad_(True)
    # Infer the number of outputs rather than assuming 1000: the same code runs
    # on the 50-class FunnyBirds models.
    with torch.no_grad():
        n_out = wrapper.forward(x[:1]).shape[1]
    one_hot = torch.zeros(x.shape[0], n_out, device=x.device)
    one_hot.scatter_(1, target.view(-1, 1), 1.0)
    with Gradient(model=wrapper.model, composite=comp) as attributor:
        _, relevance = attributor(xn, one_hot)
    return relevance.detach()


# --------------------------------------------------------------------------- #
# RISE
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rise(wrapper: ModelWrapper, x: torch.Tensor, target: torch.Tensor,
         n_masks: int = 4000, s: int = 7, p: float = 0.5,
         batch_size: int = 64, generator: torch.Generator | None = None) -> torch.Tensor:
    """Randomised Input Sampling for Explanation (Petsiuk et al., 2018).

    Coarse binary masks are upsampled with bilinear interpolation and randomly
    shifted, which is what makes the masks smooth rather than blocky. The map
    is the mask-weighted average of the target-class probability.
    """
    b, _, h, w = x.shape
    device = x.device
    cell_h, cell_w = int(h / s) + 1, int(w / s) + 1
    up_h, up_w = (s + 1) * cell_h, (s + 1) * cell_w

    grid = (torch.rand(n_masks, 1, s, s, device=device, generator=generator) < p).float()
    masks = F.interpolate(grid, size=(up_h, up_w), mode="bilinear", align_corners=False)
    offsets = torch.randint(0, max(cell_h, 1), (n_masks, 2), device=device, generator=generator)
    offsets[:, 1] = torch.randint(0, max(cell_w, 1), (n_masks,), device=device, generator=generator)

    out = torch.zeros(b, h, w, device=device)
    for i in range(b):
        acc = torch.zeros(h, w, device=device)
        for start in range(0, n_masks, batch_size):
            burst = time.perf_counter()
            stop = min(start + batch_size, n_masks)
            chunk = torch.stack(
                [
                    masks[j, 0, offsets[j, 0]: offsets[j, 0] + h, offsets[j, 1]: offsets[j, 1] + w]
                    for j in range(start, stop)
                ]
            )
            masked = x[i: i + 1] * chunk.unsqueeze(1)
            probs = torch.softmax(wrapper.forward(masked), dim=1)[:, target[i]]
            acc += (probs.view(-1, 1, 1) * chunk).sum(dim=0)
            gpu_throttle(burst, device)
        out[i] = acc / (n_masks * p)
    return out


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def compute_attribution(method: str, wrapper: ModelWrapper, x: torch.Tensor,
                        target: torch.Tensor, cfg=None, postprocess: str | None = None) -> torch.Tensor:
    """Return a (B, H, W) importance map for `method` on `wrapper`."""
    get = (lambda k, d: cfg.get_path(k, d)) if cfg is not None else (lambda k, d: d)
    mode = postprocess or DEFAULT_POSTPROCESS.get(method, "sum")

    if method == "gradcam":
        with GradCAM(wrapper) as cam:
            raw = cam(x, target)
    elif method == "integrated_gradients":
        raw = integrated_gradients(
            wrapper, x, target,
            n_steps=get("attribution.integrated_gradients.n_steps", 32),
            baseline=get("attribution.integrated_gradients.baseline", "zeros"),
        )
    elif method == "gradient_shap":
        raw = gradient_shap(
            wrapper, x, target,
            n_samples=get("attribution.gradient_shap.n_samples", 16),
            stdevs=get("attribution.gradient_shap.stdevs", 0.09),
            max_forward_batch=get("attribution.gradient_shap.max_forward_batch", 32),
        )
    elif method == "lrp":
        raw = lrp(wrapper, x, target, composite=get("attribution.lrp.composite", "epsilon_plus_flat"))
    elif method == "rise":
        raw = rise(
            wrapper, x, target,
            n_masks=get("attribution.rise.n_masks", 4000),
            s=get("attribution.rise.s", 7),
            p=get("attribution.rise.p", 0.5),
            batch_size=get("attribution.rise.batch_size", 64),
        )
    elif method == BUILTIN:
        raw = wrapper.builtin_explanation(x, target)
    else:
        raise KeyError(f"unknown method {method!r}; available: {METHODS + [BUILTIN]}")

    return _postprocess(raw, mode)
