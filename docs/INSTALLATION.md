# Native environments

Run these commands from the cloned repository on Linux. Choose PyTorch CUDA
wheels compatible with your hardware. Create separate environments rather than
installing every backend into the attack environment. Download code and weights
on the execution machine.

## Attack, CLIP evaluation and anchor generation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/core.txt
python -m pip install -e . --no-deps

# Add the native anchor generator in this environment:
bash scripts/fetch_backends.sh anchors
python -m pip install -r requirements/anchors.txt
python -m pip install -e vendor/stable-diffusion --no-deps
```

The anchor fetch helper updates a relocated Lightning utility import for the
modern dependency environment. Anchor-generation settings are recorded locally.

## UniDiffuser

```bash
python3.10 -m venv .venv-unidiffuser
source .venv-unidiffuser/bin/activate
python -m pip install -r requirements/core.txt -r requirements/unidiffuser.txt
python -m pip install -e . --no-deps
bash scripts/fetch_backends.sh unidiffuser
```

The launcher adds the native repository to the import path and resolves the
UVit, autoencoder and caption-decoder weights from `thu-ml/unidiffuser-v1`.
Native helper modules retrieve public CLIP / GPT-2 assets through their upstream
mechanisms. Captioning retains the native DPM-Solver and decoding.

## BLIP-2 and InstructBLIP

```bash
python3.10 -m venv .venv-lavis
source .venv-lavis/bin/activate
bash scripts/fetch_backends.sh lavis
python -m pip install -r requirements/lavis.txt
python -m pip install -e vendor/LAVIS --no-deps
python -m pip install -e . --no-deps
python -c "from lavis.models import load_model_and_preprocess; print('LAVIS imports OK')"
```

Use `--no-deps` for the native editable installation to retain the supplied
dependency versions: Torch 2.4.1, TorchVision 0.19.1 and Transformers 4.30.2.
The helper configures LAVIS for caption generation. BLIP-2 uses Flan-T5-XXL;
InstructBLIP uses Vicuna-13B-v1.1.

## MiniGPT-4

```bash
python3.9 -m venv .venv-minigpt4
source .venv-minigpt4/bin/activate
bash scripts/fetch_backends.sh minigpt4
python -m pip install -r requirements/minigpt4.txt
python -m pip install -e . --no-deps
```

Torch, TorchVision and TorchAudio use matching versions. The fetch helper enables the stopping-criteria
argument. The launcher overrides native weight paths with the public Vicuna-v0
identifier and the original projection checkpoint. BitsAndBytes needs a supported
CUDA setup for low-resource inference.

## LLaVA-v1.5

```bash
python3.10 -m venv .venv-llava
source .venv-llava/bin/activate
bash scripts/fetch_backends.sh llava
python -m pip install -e vendor/LLaVA
python -m pip install -e . --no-deps
```

The backend is pinned to its experiment commit in `configs/backend_refs.json`.
Native installation uses its own dependencies in this environment.

## LLaVA-NeXT

```bash
python3.10 -m venv .venv-next
source .venv-next/bin/activate
bash scripts/fetch_backends.sh llava_next
python -m pip install -e vendor/LLaVA-NeXT
python -m pip install bitsandbytes ninja packaging
python -m pip install flash-attn --no-build-isolation
python -m pip install -e . --no-deps
```

FlashAttention installation needs a compatible compiler/CUDA toolkit. The native
launcher requests 4-bit loading, `device_map=auto`, FlashAttention 2 and the
`qwen_1_5` conversation template for the 72B checkpoint. Select enough available
GPU memory; multiple visible GPUs may be supplied through `GPU`.

LLaVA and NeXT export the same `llava` import name and use separate environments.
The fetch helper records the NeXT revision under `vendor/`. Set `BACKEND_REF`
to select an upstream commit.

## GPT judging

```bash
python3.10 -m venv .venv-judge
source .venv-judge/bin/activate
python -m pip install -r requirements/api.txt
python -m pip install -e . --no-deps
```

Set your own `OPENAI_API_KEY` in the shell. Use the original `gpt-4.1-mini`
identifier. Set `API_PYTHON` to this environment's Python when using
`scripts/evaluate.sh` from the CLIP environment.

## Checks

Run `bash scripts/check.sh` to check the source and command interfaces. For a small
inference run, set `--num-samples` and a compatible `--batch-size`; use the
documented batch size for the full experiment.
