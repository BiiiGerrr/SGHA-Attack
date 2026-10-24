# Experiment settings

The three single-CLIP programs implement the feature hooks, anchor ranking and
weights, projections, cosine objectives, layer averages, PGD update and
preprocessing used in the experiments. The profile coefficients match Table VI.

For ablations with direct image guidance, pass `--no_anchor` and
`--tgt_data_path DIRECTORY`. Setting a
loss coefficient to zero does not implicitly remove the final image-guidance term.

HVSA and CLSS average selected-layer losses. Replacing the averages with sums
requires rescaling coefficients. Patch alignment uses mean-pooled patch descriptors.
CLSS uses the encoder projections and cosine normalization without an additional
final LayerNorm.

`legacy/attack_SGHA.py` preserves the previously public augmentation variant.
Use the new profile entry points for the experiment snapshots documented here.

## Seeds and anchor generation

`configs/anchor_seeds.json` contains the selected 19 anchor seeds in their original
order. Single-CLIP attacks use seed 2023. First-trial caption generation uses 1234
for UniDiffuser and 2023 for the other five native victims. The launcher passes
the selected value through `SGHA_SEED`; Python, NumPy and Torch are seeded by the
original backend routines. GPT judging sends no API seed parameter.

Anchor generation retains native PLMS, batch size 10, 50 steps, guidance 7.5,
512x512, full-EMA weights, safety processing and watermarking.

## Native victim inference

| Victim | Attack images | Batch | Generation |
|---|---|---:|---|
| UniDiffuser-v1 | ViT-B/32 | 100 | Native DPM-Solver, 50 steps, guidance 7, data_type=1 |
| BLIP-2 / Flan-T5-XXL | EVA01-g/14 | 5 | LAVIS nucleus sampling |
| InstructBLIP / Vicuna-13B-v1.1 | EVA01-g/14 | 2 | LAVIS nucleus sampling |
| MiniGPT-4 / Vicuna-v0 | EVA01-g/14 | 20 | do_sample=False, 1 beam, 100 new tokens |
| LLaVA-v1.5-13B | ViT-L/14-336 | 1 | vicuna_v1, temperature 0, 1 beam |
| LLaVA-NeXT-72B | ViT-L/14-336 | 1 | qwen_1_5, 4-bit, original attention configuration |

Model weights are retrieved using public HF IDs. LLaVA-v1.5 and NeXT use separate environments because both
provide the `llava` import name.

Observed dependencies: attacks used Torch 2.6.0 / TorchVision 0.21.0,
Transformers 4.57.3, OpenCLIP 3.2.0 and timm 1.0.21. LAVIS used Torch 2.4.1 /
TorchVision 0.19.1 and Transformers 4.30.2. LLaVA used Torch 2.1.2 /
TorchVision 0.16.2 and Transformers 4.37.2. Installation commands are in `INSTALLATION.md`.
The fetch script pins fresh LAVIS, LLaVA-v1.5 and MiniGPT-4 clones to the commits
in `configs/backend_refs.json` and records all fetched revisions. `BACKEND_REF`
overrides a revision. The MiniGPT-4 snapshot enables its previously commented
`stopping_criteria=stopping_criteria` generation argument; `scripts/patch_backend.py`
applies that same change. LAVIS also disables an unused 3D-library import, as in
the caption snapshot. Path edits are restored to public model identifiers.
The fetch helper also updates Stable Diffusion's relocated Lightning rank-zero
utility import for Lightning 2.x; it does not change the sampler or losses.

## Metrics

CLIP is normalized text/text cosine between victim responses and fixed target
descriptions, averaged over pairs and then over RN50, RN101, ViT-B/16, ViT-B/32
and ViT-L/14. Generated descriptions use truncating tokenization; short COCO
targets retain the original non-truncating tokenization.

GPT uses the same three-level prompt: failure=0, fooled but not targeted=0.5,
targeted success=1. ASR_fool counts score>=0.5 and ASR_target counts score=1.
Both are percentages. Every victim uses the SAME MiniGPT-4 clean descriptions.

The evaluator caches valid responses and retries missing, malformed or truncated
judgments. ASR is summarized after every pair has a valid score.
