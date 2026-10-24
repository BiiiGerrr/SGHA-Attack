# SGHA-Attack: Semantic-Guided Hierarchical Alignment for Transferable Targeted Attacks on Vision-Language Models

Official implementation of SGHA-Attack, including anchor generation, three CLIP
surrogate attacks, six victim caption backends, and CLIP / GPT evaluation.

[中文执行说明](README_zh.md) · [Native environments](docs/INSTALLATION.md) · [Method settings](docs/REPRODUCIBILITY.md)

## Pipeline

1. Obtain the selected clean images and preserve their ordered pairing with target captions.
2. Generate candidate anchor pools with the native Stable Diffusion PLMS sampler.
3. Generate adversarial images using the surrogate assigned to each victim.
4. Generate victim descriptions from those saved images.
5. Evaluate descriptions using five CLIP text encoders and GPT judging.

The repository includes the selected image filenames, target captions, anchor
seeds and experiment configurations. The following sections describe data
preparation, model setup and the complete attack and evaluation workflow.

## 1. Installation

Use Linux and Python 3.10 for the attack / CLIP environment:

```bash
git clone https://github.com/BiiiGerrr/SGHA-Attack.git
cd SGHA-Attack
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/core.txt
python -m pip install -e . --no-deps
bash scripts/check.sh
```

Select CUDA wheels suitable for your GPU and driver. Keep caption environments
separate from this environment, especially LLaVA-v1.5 and LLaVA-NeXT, which both
provide the `llava` package. Follow [the native environment commands](docs/INSTALLATION.md)
before running each caption backend. Download dependencies and weights directly
on the machine that will execute them. HF access is subject to each model's
terms; if an asset requires authentication, set your own `HF_TOKEN` in the shell.

## 2. Data preparation

`data/metadata/targets_1000.txt` contains the selected target descriptions;
`data/metadata/clean_images_1000.txt` specifies the sorted relative image filenames.
Obtain those ImageNet images separately:

```text
data/clean/
  images/
    ILSVRC2012_val_00000001.JPEG
    ...
```

**Pairing is positional:** target row i, sorted clean image i, and image i in every
anchor pool must correspond. Do not reorder individual files. For another dataset,
supply your own equally ordered target file and `--expected-list`.

## 3. Generate anchors

Install the anchor backend and its dependencies:

```bash
bash scripts/fetch_backends.sh anchors
python -m pip install -r requirements/anchors.txt
python -m pip install -e vendor/stable-diffusion --no-deps
GPU=0 bash scripts/generate_anchors.sh
```

`configs/anchor_seeds.json` contains the 19 selected experiment seeds in their
original order. The script generates one pool per seed. Override with
`ANCHOR_SEEDS_FILE=your_seeds.local.json` to use another seed list.
At least five pools are required by Top-K=5. Generated seed records
and inference outputs are ignored by Git.

Generation retains the native PLMS sampler, 50 steps, guidance 7.5, 512x512,
batch size 10, safety processing and watermarking. Weights resolve to
`CompVis/stable-diffusion-v-1-4-original` / `sd-v1-4-full-ema.ckpt`.

```text
outputs/anchors/seed_record.txt
outputs/anchors/generation.json
outputs/anchors/seed-853997548/samples/00000.png
...
```

Each pool contains `00000.png` through `00999.png`. Reuse pools across attacks.
Verify image ordering and hashes before attacking:

```bash
python -m sgha_attack.prepare_data --clean-root data/clean \
  --targets data/metadata/targets_1000.txt \
  --anchor-record outputs/anchors/seed_record.txt
```

## 4. Generate adversarial images

| Surrogate | Resolution | Layers (zero-based) | Batch | lambda_feature | lambda_cls | lambda_spatial | lambda_text_mid | Victims |
|---|---:|---|---:|---:|---:|---:|---:|---|
| ViT-B/32 | 224 | 7, 9, 11 | 250 | 1.5 | 1.0 | 0.7 | 2.5 | UniDiffuser |
| ViT-L/14-336 | 336 | 14, 18, 22 | 10 | 8.0 | 0.5 | 1.5 | 1.0 | LLaVA, LLaVA-NeXT |
| EVA01-g/14 | 224 | 23, 30, 37 | 5 | 6.0 | 1.0 | 0.8 | 2.5 | BLIP-2, InstructBLIP, MiniGPT-4 |

All three use attack seed 2023, 100 steps, alpha 1 pixel, epsilon 8 pixels (=8/255), Top-K=5,
anchor temperature 5, lambda_text=1, lambda_anchor=1 and feature noise=0.
`configs/attack_profiles.json` is forwarded explicitly to the native attack CLI.

```bash
# Run the three surrogates sequentially with the paper attack seed.
GPU=0 SEED=2023 bash scripts/attack_all.sh

# Or run one surrogate:
CUDA_VISIBLE_DEVICES=0 python -m sgha_attack.attack \
  --surrogate vit_l --clean-root data/clean \
  --anchor-record outputs/anchors/seed_record.txt \
  --output outputs/attacks/vit_l/seed_2023 --seed 2023
```

Original image stems and class folders are preserved as PNG. ViT-B uses OpenAI
CLIP; ViT-L resolves `openai/clip-vit-large-patch14-336`; EVA resolves
`QuanSun/EVA-CLIP` / `EVA01_CLIP_g_14_psz14_s11B.pt`. Override with `--model-id`
(an HF ID for ViT-L or checkpoint filename for EVA).

`--dry-run` prints the plan without importing Torch or downloading weights.
For a smaller run set both `--num-samples` and a divisible `--batch-size`.
Pass native ablation flags after standard arguments, for example
`--lambda_text 0 --lambda_text_mid 0`. Use `--no_anchor` to disable the anchor pool;
add `--tgt_data_path DIRECTORY` for direct target-image guidance.
Use a new output directory for every attack.

## 5. Generate victim captions

Activate each backend's environment from [INSTALLATION.md](docs/INSTALLATION.md).
The following commands retain the original model routing and first-trial seeds:

```bash
GPU=0 bash scripts/caption.sh unidiffuser outputs/attacks/vit_b/seed_2023 outputs/captions/unidiffuser.txt 1234
GPU=0 bash scripts/caption.sh blip2 outputs/attacks/vit_g/seed_2023 outputs/captions/blip2.txt 2023
GPU=0 bash scripts/caption.sh instructblip outputs/attacks/vit_g/seed_2023 outputs/captions/instructblip.txt 2023
GPU=0 bash scripts/caption.sh minigpt4 outputs/attacks/vit_g/seed_2023 outputs/captions/minigpt4.txt 2023
GPU=0 bash scripts/caption.sh llava outputs/attacks/vit_l/seed_2023 outputs/captions/llava.txt 2023
GPU=0 bash scripts/caption.sh llava_next outputs/attacks/vit_l/seed_2023 outputs/captions/llava_next.txt 2023
```

For multi-GPU NeXT inference, set e.g. `GPU=0,1`; use sufficient available memory.
The fourth script argument supplies the caption seed. UniDiffuser uses 1234 for
the paper first trial; the other five use 2023. Default batch
sizes are 100 / 5 / 2 / 20 / 1 / 1 in the listed order. Native sampling, prompts,
decoding, and MiniGPT-4 stopping criteria are preserved. Each command produces one
line per image and a `.run.json` manifest containing input hashes and settings.
Choose a new output filename for each caption run.

Weights use public IDs such as `thu-ml/unidiffuser-v1`,
`liuhaotian/llava-v1.5-13b`, `lmms-lab/llava-next-72b`,
`Vision-CAIR/MiniGPT-4` and `Vision-CAIR/vicuna`. Native LAVIS resolves its official
BLIP-2 / InstructBLIP checkpoints, Flan-T5-XXL and Vicuna-v1.1. MiniGPT-4 uses
Vicuna-v0.

Generate one shared clean-reference description file using MiniGPT-4:

```bash
GPU=0 bash scripts/caption.sh minigpt4 data/clean outputs/captions/clean_minigpt4.txt 2023
```

Use this shared clean-reference file for GPT judging of every victim.

## 6. CLIP and GPT evaluation

Use the attack environment for CLIP. Install `requirements/api.txt` in a Python
3.10 environment and set your own `OPENAI_API_KEY` for GPT judging. `.env.example`
is only a template; the program does not load `.env` automatically. If the judge
interpreter differs, set `API_PYTHON` to its executable.

```bash
ORIGINAL_CAPTIONS=outputs/captions/clean_minigpt4.txt \
  GPU=0 bash scripts/evaluate.sh outputs/captions/llava.txt outputs/evaluation/llava

# Individual stages:
python -m sgha_attack.eval_clip --captions outputs/captions/llava.txt \
  --targets data/metadata/targets_1000.txt --output outputs/evaluation/llava_clip.json
python -m sgha_attack.eval_gpt --captions outputs/captions/llava.txt \
  --targets data/metadata/targets_1000.txt --original outputs/captions/clean_minigpt4.txt \
  --model gpt-4.1-mini --output outputs/evaluation/llava_gpt.json
```

- CLIP: paired normalized **text/text** cosine, averaged over RN50, RN101,
  ViT-B/16, ViT-B/32 and ViT-L/14.
- ASR_fool (%): valid judge score >= 0.5; ASR_target (%): score == 1.
- GPT: unchanged three-level prompt, temperature 0, 1024 output tokens, official
  API endpoint, and no API seed parameter.
- Invalid/missing responses are not zero scores. The same command resumes only
  missing judgments; no final ASR summary is emitted until every row is valid.

Use equally sized target, clean-reference and generated-caption files.

## Checks and acknowledgements

```bash
bash scripts/check.sh
```

The checks cover attack parameters, caption interfaces, output manifests,
ablations and judge failure/resume behavior.

[THIRD_PARTY.md](THIRD_PARTY.md) lists upstream sources and licenses. This work
builds on AttackVLM; we thank its authors.
