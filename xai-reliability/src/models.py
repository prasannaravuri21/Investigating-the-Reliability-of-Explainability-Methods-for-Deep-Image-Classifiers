"""Week 4: load the four classifiers behind one common interface.

Design decision that matters for the rest of the project
--------------------------------------------------------
Every perturbation in this study (ROAD imputation, IDSDS patch deletion)
happens in *image space*: a float tensor in [0, 1] with three channels. Each
model therefore owns its own normalisation and applies it inside `forward`.
Without this, "remove 30% of the pixels" would mean a different thing for a
ResNet (ImageNet mean/std) than for a B-cos network (which takes a six-channel
[x, 1-x] encoding), and the cross-architecture comparison in RQ2 would be
confounded by preprocessing.

No model is trained from scratch. ResNet-50, EfficientNet-B0 and ViT-B/16 come
from torchvision; the B-cos network comes from the authors' release
(https://github.com/B-cos/B-cos-v2), which provides an exact built-in
explanation and acts as the faithful reference point for RQ3.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
from torchvision import transforms as T

from .utils import get_logger

LOG = get_logger()

# Set XAI_PRETRAINED=0 to build the architectures with random weights. Only
# scripts/smoke_test.py should do this: it checks that the pipeline runs without
# downloading 500 MB of checkpoints first.
USE_PRETRAINED = os.environ.get("XAI_PRETRAINED", "1") != "0"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_space_transform(size: int = 224) -> Callable:
    """PIL -> float tensor in [0, 1], 3 x size x size. No normalisation."""
    return T.Compose(
        [
            T.Resize(int(size * 256 / 224)),
            T.CenterCrop(size),
            T.ToTensor(),
        ]
    )


def add_inverse(x: torch.Tensor) -> torch.Tensor:
    """B-cos input encoding: stack the image with its complement."""
    return torch.cat([x, 1.0 - x], dim=1)


def get_module(model: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted module path, allowing negative list indices.

    get_module(resnet, "layer4.-1")               -> last bottleneck block
    get_module(vit, "encoder.layers.-1.ln_1")     -> last block's first norm
    """
    node: nn.Module = model
    for part in path.split("."):
        if part.lstrip("-").isdigit():
            children = list(node.children()) if not isinstance(node, (nn.Sequential, nn.ModuleList)) else list(node)
            node = children[int(part)]
        else:
            node = getattr(node, part)
    return node


# --------------------------------------------------------------------------- #
# Wrapper
# --------------------------------------------------------------------------- #
@dataclass
class ModelWrapper:
    """One classifier plus everything the attribution code needs to know."""

    name: str
    model: nn.Module
    normalise: Callable[[torch.Tensor], torch.Tensor]
    gradcam_layer: str
    gradcam_reshape: Callable | None = None
    is_bcos: bool = False
    lrp_supported: bool = True
    image_size: int = 224
    notes: str = ""
    _device: torch.device = field(default=torch.device("cpu"), repr=False)

    # -- basic plumbing ---------------------------------------------------- #
    def to(self, device: torch.device) -> "ModelWrapper":
        self.model.to(device).eval()
        self._device = device
        return self

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def transform(self) -> Callable:
        return image_space_transform(self.image_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits from an image-space batch (B, 3, H, W) in [0, 1]."""
        return self.model(self.normalise(x))

    __call__ = forward

    def target_layer(self) -> nn.Module:
        return get_module(self.model, self.gradcam_layer)

    # -- B-cos built-in explanation ---------------------------------------- #
    def builtin_explanation(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """The B-cos model's own explanation, as a (B, H, W) map.

        A B-cos network is dynamically linear, logit = W(x)^T x, so the
        contribution map W(x) * x *is* the model's computation rather than an
        approximation of it. This mirrors the authors' `explain()`
        (bcos/common.py): two details matter and both were wrong in an earlier
        version of this function.

        1. explanation_mode() detaches the dynamic normalisation, so the input
           gradient equals W(x). Without it the gradient also contains dW/dx
           and grad * input is no longer the contribution map.
        2. The gradient is taken on the six-channel [x, 1-x] encoding the model
           actually sees. Taking it on the 3-channel image instead folds the
           two halves together with the wrong sign (d(1-x)/dx = -1).

        Checks (verified): the map matches the authors' explain() to ~1e-9, and
        it sums to the logit minus B-cos v2's constant logit bias of
        -ln(1000) = -6.908, the same offset for every image.
        """
        if not self.is_bcos:
            raise RuntimeError(f"{self.name} has no built-in explanation")
        encoded = self.normalise(x).detach().clone().requires_grad_(True)
        mode = getattr(self.model, "explanation_mode", None)
        if mode is None:
            raise RuntimeError("B-cos model has no explanation_mode(); check the hub version")
        with torch.enable_grad(), mode():
            logits = self.model(encoded)
            selected = logits.gather(1, target.view(-1, 1)).sum()
            (grad,) = torch.autograd.grad(selected, encoded)
        return (grad * encoded.detach()).sum(dim=1).detach()


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _torchvision_normalise() -> Callable:
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    return lambda x: norm(x)


def _build_resnet50() -> ModelWrapper:
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if USE_PRETRAINED else None).eval()
    return ModelWrapper(
        name="resnet50",
        model=model,
        normalise=_torchvision_normalise(),
        gradcam_layer="layer4.-1",
        notes="torchvision IMAGENET1K_V2 weights",
    )


def _build_efficientnet_b0() -> ModelWrapper:
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if USE_PRETRAINED else None).eval()
    return ModelWrapper(
        name="efficientnet_b0",
        model=model,
        normalise=_torchvision_normalise(),
        gradcam_layer="features.-1",
        # zennit has no EfficientNet canonizer: BatchNorm stays unfused and the
        # squeeze-excitation gates multiply activations, which the CNN rules do
        # not handle. A run without one broke relevance conservation (every
        # pixel positive, the map summing to ~1e6 x the logit, near-uniform),
        # so the output was not LRP and the cell is recorded not-applicable.
        lrp_supported=False,
        notes="Grad-CAM reads the final conv stage; depthwise-separable blocks "
        "make the map coarser than ResNet's. LRP not applicable: no canonizer for "
        "squeeze-excitation gates and unfused BatchNorm, so relevance is not conserved",
    )


def _vit_reshape(activations: torch.Tensor) -> torch.Tensor:
    """(B, 1 + N, C) tokens -> (B, C, sqrt(N), sqrt(N)), dropping the CLS token."""
    tokens = activations[:, 1:, :]
    b, n, c = tokens.shape
    side = int(round(n ** 0.5))
    if side * side != n:
        raise ValueError(f"cannot reshape {n} patch tokens into a square grid")
    return tokens.transpose(1, 2).reshape(b, c, side, side)


def _build_vit_b_16() -> ModelWrapper:
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if USE_PRETRAINED else None).eval()
    return ModelWrapper(
        name="vit_b_16",
        model=model,
        normalise=_torchvision_normalise(),
        # The norm *before* the last block's attention keeps the token
        # activations interpretable; taking the block output instead mixes in
        # the residual stream and gives a noticeably noisier map.
        gradcam_layer="encoder.layers.-1.ln_1",
        gradcam_reshape=_vit_reshape,
        lrp_supported=False,
        notes="LRP has no agreed rule set for attention; see attributions.lrp",
    )


def _build_bcos_resnet50() -> ModelWrapper:
    """B-cos ResNet-50 from the authors' hub (interpretable by design)."""
    model = torch.hub.load("B-cos/B-cos-v2", "resnet50", pretrained=USE_PRETRAINED)
    model.eval()

    # The published B-cos models take the six-channel [x, 1-x] encoding. Probe
    # once so the pipeline fails loudly here rather than mid-experiment.
    with torch.no_grad():
        probe = torch.zeros(1, 3, 224, 224)
        try:
            model(add_inverse(probe))
            normalise = add_inverse
        except RuntimeError:
            model(probe)
            normalise = lambda x: x  # noqa: E731
            LOG.warning("B-cos model accepted 3-channel input; AddInverse appears to be internal")

    return ModelWrapper(
        name="bcos_resnet50",
        model=model,
        normalise=normalise,
        gradcam_layer="layer4.-1",
        is_bcos=True,
        # B-cos layers are dynamic-linear (weights depend on the input) and use
        # bias-free normalisation, so the CNN LRP rules and the ResNet canonizer
        # do not apply. The built-in explanation is the model's own exact
        # contribution map and takes LRP's place for this architecture.
        lrp_supported=False,
        notes="B-cos v2 ResNet-50; forward pass is itself an exact explanation. "
              "LRP not applicable: dynamic-linear B-cos layers have no standard relevance rules",
    )


BUILDERS: dict[str, Callable[[], ModelWrapper]] = {
    "resnet50": _build_resnet50,
    "efficientnet_b0": _build_efficientnet_b0,
    "vit_b_16": _build_vit_b_16,
    "bcos_resnet50": _build_bcos_resnet50,
}


@torch.no_grad()
def resolve_target(wrapper: "ModelWrapper", x: torch.Tensor, labels: torch.Tensor,
                   mode: str = "predicted") -> torch.Tensor:
    """The class whose evidence is being explained.

    Faithfulness asks what the model computed, not whether it was right, so the
    model's own top-1 prediction is the defensible target. It also matters
    practically here: on a set where a model is only ~65% accurate, explaining
    the ground-truth label would mean a third of the maps describe evidence for
    a class the model did not choose, which quietly turns a faithfulness
    measurement into an accuracy measurement.
    """
    if mode == "ground_truth":
        return labels
    if mode != "predicted":
        raise ValueError(f"target_mode must be 'predicted' or 'ground_truth', got {mode!r}")
    return wrapper.forward(x).argmax(dim=1)


def load_model(name: str, device: torch.device | str = "cpu") -> ModelWrapper:
    if name not in BUILDERS:
        raise KeyError(f"unknown model {name!r}; available: {sorted(BUILDERS)}")
    wrapper = BUILDERS[name]()
    for p in wrapper.model.parameters():
        p.requires_grad_(False)
    return wrapper.to(torch.device(device))
