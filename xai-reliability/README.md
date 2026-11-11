# Reliability of explainability methods for deep image classifiers

Code for the MSc project *Investigating the Reliability of Explainability
Methods for Deep Image Classifiers*: five attribution methods, four
architectures, two debiased faithfulness protocols, and the rank-correlation
analysis that answers RQ1-RQ4.

This repository covers the technical work scheduled for Weeks 3-7 of the
12-week plan, plus the analysis entry point used in Weeks 9-10.

---

## 1. The dataset

**ImageNetV2 matched-frequency.** One download, 1.26 GB, no account, no terms
gate, no approval of any kind. 10 images for each of the 1,000 ImageNet classes,
which splits into everything this project needs:

| Split | Size | Used for |
|---|---|---|
| evaluation subset | 1 image per class = **1,000 images** | every attribution map, every ROAD and IDSDS score |
| fine-tune split | 9 per class = **9,000 images** | the IDSDS patch-drop adaptation, disjoint from the above |

It is the same 1,000 ImageNet classes, so every pretrained model used here —
ResNet-50, EfficientNet-B0, ViT-B/16 and the B-cos network — applies unchanged.
It is also a published, citable dataset (Recht et al., 2019), not an improvised
subset.

No ethics approval is involved. This project uses public images and public
pretrained weights: no human participants, no personal data, no sensitive data.
Section 1.11 of the report covers this correctly as it stands.

### Download

```powershell
pip install -U "huggingface_hub[cli]"
hf download vaishaal/ImageNetV2 imagenetv2-matched-frequency.tar.gz ^
    --repo-type dataset --local-dir D:\data\imagenetv2
cd D:\data\imagenetv2
tar -xzf imagenetv2-matched-frequency.tar.gz
dir
```

Source: https://huggingface.co/datasets/vaishaal/ImageNetV2 — repository code is
MIT; the images carry the same Flickr terms as ImageNet itself and need no
access request. Cite Recht et al. (2019).

### Build the two manifests

Check the extracted folder names first. Digits (`0`, `1`, ... `999`) are already
ImageNet class indices, so use `--source indexfolder`. WordNet ids
(`n01440764`...) need `--source imagenet_val` on the same folder, which derives
the index by sorting the ids.

```powershell
set ROOT=D:\data\imagenetv2\imagenetv2-matched-frequency-format-val

:: evaluation subset: 1 image per class
python scripts\w3_build_subset.py --source indexfolder --root %ROOT%

:: fine-tune split: the other 9 per class, guaranteed disjoint
python scripts\w3_build_subset.py --source indexfolder --root %ROOT% ^
    --per-class 9 --exclude data\subset_manifest.csv ^
    --out data\finetune_manifest.csv
```

Commit both CSVs. They are the definition of "the dataset" for this project, and
the reproducibility claim in Section 1.10 rests on them.

### Verify before going further

```powershell
python scripts\w4_baseline_accuracy.py
```

Expect roughly 63-70% top-1 for ResNet-50, with the other three in a similar
band — about eleven points below each model's published ImageNet validation
figure. That gap is the known ImageNetV2 distribution shift, not a bug. A result
near 0.1% means the class-index mapping is wrong. Do not proceed past this
check: every later number inherits the error.

### The explanation target

Because the models sit near 65% accuracy on this set, the explanation target
matters. `configs/default.yaml` sets `attribution.target_mode: predicted`, so
each map explains the model's own top-1 class and ROAD measures whether deletion
changes *that* prediction. Explaining the ground-truth label instead would mean
a third of the maps describe evidence for a class the model never chose, which
turns a faithfulness measurement into an accuracy measurement. The targets are
saved beside each map so the scoring stage cannot drift from the attribution
stage.

### What to declare in the report

Two sentences, in Section 2.2 and again in the limitations:

> The evaluation subset comprises 1,000 images, one per ImageNet class, drawn
> from ImageNetV2 matched-frequency (Recht et al., 2019), a replication of the
> ILSVRC2012 validation protocol collected in 2019. Absolute top-1 accuracy is
> accordingly around eleven points below published ImageNet validation figures;
> since every comparison in this study is made between methods and between
> metrics on one fixed image set, this shifts the baselines without affecting
> the research questions.

And on the fine-tune: Hesse et al. (2024) adapt on the ImageNet *training*
split, whereas the patch-drop fine-tune here uses 9,000 held-out images from the
same pool. For a short low-learning-rate adaptation to patch removal that is
sufficient, but it is a deviation from their protocol and belongs in the
write-up rather than left for a reader to assume.

### Optional upgrade: the official ILSVRC2012 validation set

Not required, and not needed for any of the four research questions. The one
thing it buys is direct comparability with the published ROAD and IDSDS numbers,
which is a useful cross-check on this implementation of the noisy linear
imputation. It needs a click-through research-use licence (self-serve on Hugging
Face, hand-vetted on image-net.org) — a licence, not an ethics review.

If you ever want it, `scripts/w3_prepare_imagenet_val.py` converts any of the
three download formats into the same `<root>/<class_idx>/` layout, after which
nothing else changes:

```powershell
:: Hugging Face via the datasets library, after accepting the terms
python scripts\w3_prepare_imagenet_val.py --mode hf --repo imagenet-1k --out D:\data\in1k-val
:: already-downloaded webdataset shards
python scripts\w3_prepare_imagenet_val.py --mode wds --shard-dir <dir> --out D:\data\in1k-val
:: the raw tar plus the devkit
python scripts\w3_prepare_imagenet_val.py --mode devkit --val-tar <tar> ^
    --meta-mat <devkit>\data\meta.mat ^
    --ground-truth <devkit>\data\ILSVRC2012_validation_ground_truth.txt ^
    --out D:\data\in1k-val
```

That script exists mainly to stop one silent failure: the devkit ground truth is
1-based in ILSVRC2012_ID order, while pretrained models use alphabetical WNID
order. These are different permutations, so using the devkit number as the class
index mislabels nearly every image, nothing crashes, accuracy comes out near
0.1%, and every score afterwards is noise.

## 1b. Strengthening the study: FunnyBirds part-level ground truth

Optional, and the single best addition for the marks available. Scheduled for
Week 8, which the 12-week plan holds as slack for error correction.

**The argument.** Section 1.1 states the core problem: there is no ground truth
for the correct explanation of a prediction. ROAD and IDSDS work around it by
measuring deletion sensitivity, which is a proxy -- so RQ1 can show the two
proxies disagree, but cannot say which is closer to right. FunnyBirds (Hesse et
al., 2023) renders its images synthetically, so the framework knows which bird
parts actually determine the class and can score a map against that. Adding it
turns the finding from *"two proxies disagree"* into *"two proxies disagree, and
here is which one the ground-truth test sides with"*. That is a contribution
rather than an observation, and it is the kind of thing an examiner rewards.

**Why it is cheap.** The authors publish the dataset (1.6 GB, Apache-2.0, no
access request) and the evaluation framework with trained checkpoints for
ResNet-50, VGG16 and **ViT-B/16** -- two of which are already in this study, so
RQ2's CNN-versus-transformer comparison carries over. Their framework runs the
whole protocol; all it wants from you is an explainer object. `src/funnybirds.py`
supplies one that calls `compute_attribution`, so the maps scored here are
produced by the same code as the maps scored by ROAD and IDSDS. If those
diverged the comparison would be void, which is why it is an adapter rather than
a reimplementation.

**Setup.**

```bash
git clone https://github.com/visinf/funnybirds-framework
cd funnybirds-framework
wget download.visinf.tu-darmstadt.de/data/funnybirds/FunnyBirds.zip
unzip FunnyBirds.zip && rm FunnyBirds.zip
# download the ResNet-50 and ViT-B/16 checkpoints linked in their README
```

```powershell
python scripts\w8_funnybirds.py --framework D:\code\funnybirds-framework ^
    --checkpoint-dir D:\data\funnybirds_ckpt
```

That writes the bridging file into their `explainers/` folder and prints the
exact commands, including the one-line branch to paste into their
`evaluate_explainability.py`. Their metrics include a **single-deletion** score
from the same authors as IDSDS, which is the most directly comparable of the
set. Copy the headline numbers into `results/scores_funnybirds.csv` using the
columns this project already uses (`model, method, metric, seed, score`), and
`scripts/w9_analysis.py` picks them up as a third metric with no code change.

**What to claim, and what not to.** FunnyBirds is synthetic, so a method that
scores well there has been shown faithful on rendered birds, not on natural
images -- Hesse et al. say so themselves, and your own Table 2.1 already records
that limitation. Write it up as a controlled check that complements the ImageNet
results. Overclaiming here would undo the credit the addition earns.

**Verify the interface.** The three explainer methods follow the documented
interface, but their `explainers/explainer_wrapper.py` is the authority on the
exact signatures and on the threshold convention in `get_important_parts`. Check
it against the version you clone before trusting the numbers; the smoke test
covers the part-scoring maths but cannot cover their call signatures.

## 2. Install and check

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/smoke_test.py                          # random weights, CPU, ~2 minutes
```

The smoke test runs every stage on four synthetic images with randomly
initialised models, so it needs no dataset and no checkpoint download. LRP on
the ViT reports "not applicable" — that is the intended behaviour, not a bug
(see Section 5 below).

---

## 3. Weeks 3-7, command by command

```bash
# Week 3   dataset, manifests and EDA (see Section 1 for the download)
python scripts/w3_build_subset.py --source indexfolder --root <root>
python scripts/w3_build_subset.py --source indexfolder --root <root> \
    --per-class 9 --exclude data/subset_manifest.csv --out data/finetune_manifest.csv
python scripts/w3_eda.py

# Week 4   load the four models, confirm top-1 accuracy
python scripts/w4_baseline_accuracy.py

# Week 5   attribution maps: 20 method-model cells (+ the B-cos built-in)
python scripts/w5_run_attributions.py --models resnet50 --methods gradcam --limit 50   # pilot
python scripts/w5_run_attributions.py                                                   # full

# Week 6   ROAD
python scripts/w6_run_road.py --models resnet50 --methods gradcam --limit 200 --jobs 8

# Week 7   IDSDS: fine-tune, then score
python scripts/w7_finetune_patchdrop.py --models resnet50 bcos_resnet50
python scripts/w7_run_idsds.py --models resnet50 bcos_resnet50 --limit 200

# Week 8   repeat the whole thing for seeds 1 and 2
python scripts/w5_run_attributions.py --seed 1 && python scripts/w6_run_road.py --seed 1

# Week 8   optional third protocol with part-level ground truth (Section 1b)
python scripts/w8_funnybirds.py --framework <clone> --checkpoint-dir <ckpts>

# Weeks 9-10   rankings, Kendall's tau, figures
python scripts/w9_analysis.py
```

The Week 4 accuracy table is the gate for everything after it. Published ImageNet
validation figures: ResNet-50 (torchvision V2 weights) ~80.9%, EfficientNet-B0
~77.7%, ViT-B/16 ~81.1%, B-cos ResNet-50 ~75.9%. Subtract about eleven points
for ImageNetV2 and allow a few more for sampling noise on 1,000 images.
Anything further off means the
preprocessing or the class-index mapping is wrong, and no faithfulness score
computed on top of it would mean anything.

---

## 4. How the pieces fit

```
configs/default.yaml      every setting that can change a number
data/subset_manifest.csv  the fixed 1,000 images (committed)
scripts/w3_prepare_...py  optional: official ImageNet val -> class folders (W3)
src/data.py               manifest building and loading            (W3)
src/models.py             the four classifiers behind one interface (W4)
src/attributions.py       Grad-CAM, IG, LRP, RISE, GradientSHAP     (W5)
src/road.py               noisy linear imputation, MoRF/LeRF        (W6)
src/idsds.py              patch-drop fine-tune, single deletion     (W7)
src/funnybirds.py         optional part-level ground-truth protocol  (W8)
src/analysis.py           Kendall's tau, summary tables             (W9-10)
results/scores_*.csv      long-format rows: model, method, metric, seed, score
```

**One design decision shapes the rest.** Every perturbation happens in image
space: a float tensor in [0, 1], three channels. Each model applies its own
normalisation inside `forward`. Without this, "delete 30% of the pixels" would
mean something different for a ResNet (ImageNet mean/std) than for the B-cos
network (six-channel [x, 1-x] encoding), and the cross-architecture comparison
in RQ2 would be confounded by preprocessing rather than measuring architecture.

Both metrics are oriented so higher is better, and both write the same row
shape, so the analysis reads them with one function:

- **ROAD** = AUC(LeRF) − AUC(MoRF). A good map makes accuracy fall steeply when
  its top pixels go and stay flat when its bottom pixels go.
- **IDSDS** = Spearman correlation between per-patch attribution and the
  per-patch drop in the target logit, on the patch-drop fine-tuned model.

---

## 5. Known limitations, stated rather than hidden

**LRP on the ViT raises `NotImplementedError`.** There is no agreed relevance
rule for attention, and applying the CNN composite anyway would produce a
number that looks like LRP but is not. The cell is recorded as `not_applicable`
in `results/attribution_runs.csv`. Options for the report: leave it out and say
why, or substitute a transformer-specific relevance method (Chefer et al., 2021)
and label it as a different method. Do not quietly fill the gap.

**LRP on the B-cos network is not applicable either**, for a similar reason.
B-cos layers are dynamic-linear (their effective weights depend on the input)
and use bias-free normalisation, so the CNN relevance rules and zennit's ResNet
canonizer do not apply. The B-cos built-in explanation is the model's own exact
contribution map and takes LRP's place for this architecture, so the B-cos row
has gradcam, IG, RISE, GradientSHAP and `bcos_builtin`.

**LRP on EfficientNet-B0 is not applicable** in this implementation. zennit
has no canonizer for it: BatchNorm stays unfused and the squeeze-excitation
blocks multiply activations together, which the CNN relevance rules do not
handle. A run without one broke relevance conservation (every pixel positive,
the map summing to about a million times the logit, nearly uniform), so it was
not LRP. LRP therefore has a valid score on ResNet-50 only; the other three
architectures each have a stated reason for the gap.

**ROAD is the compute bottleneck, not IDSDS.** One sparse solve costs about
0.5-0.7 s per 224×224 image per deletion fraction on one core, so a full
1,000-image run over 9 fractions and 2 orders is roughly 3 hours of CPU per
method-model cell, or 60 hours for all 20. Mitigations, in order: `--jobs` to
match your core count (near-linear speedup), `--limit 200` for the pilot
numbers the IPR targets in Week 5-6, and the Section 3.5 contingency of
dropping to 5 deletion fractions. Budget this as CPU time, not GPU time.

**IDSDS needs a fine-tune per model.** Run it on ResNet-50 and B-cos first;
those two already answer RQ1 and RQ3. `--no-checkpoint` scores the
un-fine-tuned model, which is the out-of-distribution condition IDSDS exists to
fix — useful as an ablation, not as a result.

**GradientSHAP uses a cheap baseline distribution** (black, white, batch mean)
rather than sampled training images. Note it, and upgrade it if compute allows:
Gevaert et al. (2024) show the baseline choice moves the scores.

**The B-cos entry point should be confirmed on first run.** The model is loaded
via `torch.hub.load("B-cos/B-cos-v2", "resnet50", pretrained=True)` and the
six-channel encoding is probed at load time, so a mismatch fails immediately
rather than silently. Check the printed top-1 against ~75.9% before trusting
anything downstream.

---

## 6. Reproducibility

Fixed seeds (`set_seed` in `src/utils.py`), deterministic cuDNN, a committed
image manifest, and one YAML file holding every setting that can change a
number. Attribution maps are cached as float16 `.npy` (about 100 MB per
method-model cell at 1,000 images), so ROAD, IDSDS and the qualitative figures
all score the same maps rather than recomputing them.
