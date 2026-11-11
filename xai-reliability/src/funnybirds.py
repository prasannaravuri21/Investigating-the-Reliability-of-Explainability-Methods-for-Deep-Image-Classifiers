"""Optional third protocol: FunnyBirds part-level ground truth (Hesse et al., 2023).

Why this is worth the extra week
--------------------------------
Section 1.1 of the report states the problem plainly: there is no ground truth
for the correct explanation of a prediction. ROAD and IDSDS work around that by
measuring the model's sensitivity to deletion, which is a proxy. FunnyBirds
sidesteps it instead: the images are rendered synthetically, so the framework
knows which bird parts actually determine the class, and an attribution map can
be scored against that.

That turns RQ1 from "two proxies disagree" into something stronger: three
protocols, one of which has access to ground truth, and evidence about which of
the two deletion-based proxies the ground-truth test sides with. That is a
contribution rather than an observation.

What this module is
-------------------
An adapter, not a reimplementation. The FunnyBirds authors publish both the
dataset (about 1.6 GB, Apache-2.0, no access request) and the evaluation
framework, including trained checkpoints for ResNet-50, VGG16 and ViT-B/16 --
two of which are already in this study. Their framework runs the protocol; all
it needs from us is an explainer object. This module supplies one that routes
straight into `src.attributions.compute_attribution`, so the maps scored on
FunnyBirds are produced by exactly the same code as the maps scored by ROAD and
IDSDS. If the two differed, the comparison would be worthless.

Setup
-----
    git clone https://github.com/visinf/funnybirds-framework
    cd funnybirds-framework
    wget download.visinf.tu-darmstadt.de/data/funnybirds/FunnyBirds.zip
    unzip FunnyBirds.zip && rm FunnyBirds.zip
    # download the ResNet-50 and ViT-B/16 checkpoints linked in their README

Then see scripts/w8_funnybirds.py, which writes the small bridging file their
framework expects and prints the exact commands to run.

Caveat worth stating in the report: FunnyBirds is synthetic, so a method that
scores well here has been shown faithful on rendered birds, not on natural
images. Hesse et al. say as much. It is a controlled check that complements the
ImageNet results, not a replacement for them.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .attributions import compute_attribution
from .models import ModelWrapper, _torchvision_normalise, _vit_reshape
from .utils import get_logger

LOG = get_logger()

FUNNYBIRDS_CLASSES = 50
FUNNYBIRDS_SIZE = 256


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_funnybirds_model(arch: str, checkpoint: str | Path,
                          device: torch.device | str = "cpu") -> ModelWrapper:
    """Wrap a FunnyBirds checkpoint in the same interface as the ImageNet models.

    The checkpoints are the authors' own, trained on the 50 FunnyBirds classes.
    Loading them through this wrapper means every attribution method in
    src/attributions.py works on them without modification.
    """
    import torchvision.models as tvm

    if arch == "resnet50":
        model = tvm.resnet50(weights=None, num_classes=FUNNYBIRDS_CLASSES)
        layer, reshape, lrp_ok = "layer4.-1", None, True
    elif arch == "vit_b_16":
        model = tvm.vit_b_16(weights=None, num_classes=FUNNYBIRDS_CLASSES)
        layer, reshape, lrp_ok = "encoder.layers.-1.ln_1", _vit_reshape, False
    elif arch == "vgg16":
        model = tvm.vgg16(weights=None, num_classes=FUNNYBIRDS_CLASSES)
        layer, reshape, lrp_ok = "features.-1", None, True
    else:
        raise KeyError(f"FunnyBirds checkpoints cover resnet50, vit_b_16, vgg16; got {arch!r}")

    state = torch.load(str(checkpoint), map_location="cpu")
    state = state.get("state_dict", state.get("model", state))
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        # Loud rather than silent: a partly loaded model still runs and would
        # produce plausible-looking nonsense.
        LOG.warning("checkpoint %s: %d missing, %d unexpected keys",
                    Path(checkpoint).name, len(missing), len(unexpected))
        if len(missing) > 5:
            raise RuntimeError(
                f"{len(missing)} weights did not load from {checkpoint}. Check the architecture "
                "matches the checkpoint, and whether their modelwrapper renames keys."
            )

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return ModelWrapper(
        name=f"funnybirds_{arch}",
        model=model,
        normalise=_torchvision_normalise(),
        gradcam_layer=layer,
        gradcam_reshape=reshape,
        lrp_supported=lrp_ok,
        image_size=FUNNYBIRDS_SIZE,
        notes=f"FunnyBirds checkpoint, {FUNNYBIRDS_CLASSES} classes",
    ).to(torch.device(device))


# --------------------------------------------------------------------------- #
# Part-level scoring
# --------------------------------------------------------------------------- #
def part_importance_from_map(attribution: torch.Tensor, part_map: torch.Tensor,
                             colors_to_part: dict) -> dict[str, float]:
    """Sum an attribution map inside each labelled bird part.

    This is the same operation as `idsds.patch_attribution`, except the regions
    are semantic parts rather than a square grid -- which is the whole point:
    the framework knows which parts are causally responsible for the class.

    attribution : (H, W)
    part_map    : (3, H, W) colour-coded part segmentation, values in [0, 1]
    """
    if attribution.dim() == 3:
        attribution = attribution.squeeze(0)
    if part_map.dim() == 4:
        part_map = part_map.squeeze(0)

    # The part map may be rendered at a different resolution than the map.
    if part_map.shape[-2:] != attribution.shape[-2:]:
        part_map = F.interpolate(part_map.unsqueeze(0), size=attribution.shape[-2:],
                                 mode="nearest").squeeze(0)

    importance: dict[str, float] = {}
    for colour, part_name in colors_to_part.items():
        reference = torch.tensor([float(c) for c in colour], device=part_map.device).view(3, 1, 1)
        if reference.max() > 1.0:
            reference = reference / 255.0
        mask = (part_map - reference).abs().sum(dim=0) < 0.05
        if mask.sum() == 0:
            continue
        key = part_name[0] if isinstance(part_name, (list, tuple)) else part_name
        importance[key] = importance.get(key, 0.0) + float(attribution[mask].sum())
    return importance


def important_parts_at_thresholds(importance: dict[str, float], thresholds) -> list[list[str]]:
    """For each threshold, the parts whose share of total importance exceeds it.

    Shares are taken over positive importance only, so a part the method says
    argues *against* the class is not counted as important evidence for it.
    """
    positive = {k: max(v, 0.0) for k, v in importance.items()}
    total = sum(positive.values())
    if total <= 0:
        return [[] for _ in thresholds]
    shares = {k: v / total for k, v in positive.items()}
    return [[k for k, share in shares.items() if share > float(t)] for t in thresholds]


class FunnyBirdsExplainer:
    """The object the FunnyBirds framework asks for.

    Their `evaluate_explainability.py` calls `explain`, `get_part_importance`
    and `get_important_parts`. Everything below delegates to this project's own
    attribution code, so a method scored here is the same method scored by ROAD
    and IDSDS.

    Check the signatures against `explainers/explainer_wrapper.py` in the
    version of the framework you clone -- the three methods below follow the
    documented interface, but their abstract class is the authority, and the
    threshold convention in particular is worth confirming against their
    `AbstractAttributionExplainer`.
    """

    def __init__(self, wrapper: ModelWrapper, method: str, cfg=None):
        self.wrapper = wrapper
        self.method = method
        self.cfg = cfg

    def explain(self, image: torch.Tensor, target: torch.Tensor | int):
        if not torch.is_tensor(target):
            target = torch.tensor([int(target)], device=image.device)
        target = target.view(-1).to(image.device)
        return compute_attribution(self.method, self.wrapper, image, target, self.cfg)

    def get_part_importance(self, image, part_map, target, colors_to_part, with_bg=False):
        attribution = self.explain(image, target)
        return part_importance_from_map(attribution[0], part_map, colors_to_part)

    def get_important_parts(self, image, part_map, target, colors_to_part,
                            thresholds, with_bg=False):
        importance = self.get_part_importance(image, part_map, target, colors_to_part, with_bg)
        return important_parts_at_thresholds(importance, thresholds)
