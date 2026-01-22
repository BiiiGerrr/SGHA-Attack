# SGHA-Attack: Semantic-Guided Hierarchical Alignment for Transferable Targeted Attacks on Vision-Language Models

This repository contains the official implementation of the paper **"SGHA-Attack: Semantic-Guided Hierarchical Alignment for Transferable Targeted Attacks on Vision-Language Models"**.

## 1. Data Preparation

To replicate the experiments, please prepare the datasets as follows:

- **Clean Images**: Sourced from [ImageNet-1K](https://www.image-net.org/).
- **Target Text**: Sourced from [MS-COCO](https://cocodataset.org/).
- **Target Images**: Please refer to [AttackVLM](https://github.com/yunqing-me/AttackVLM) for the target image generation process.

## 2. Attack Generation

To perform the attack, run the `attack_SGHA.py` script.

### Quick Start

```bash
python attack_SGHA.py \
  --cle_data_path "/path/to/clean_images" \
  --tgt_text_path "/path/to/target_text.txt" \
  --anchor_list_file "./seed_record.txt" \
  --output "./results" \
  --batch_size 250 \
  --epsilon 8 \
  --clip_encoder "ViT-B/32" \
  --hook_layers 7 9 11
```

## 3. Evaluation

For evaluating the transferability and effectiveness of the generated adversarial examples, please refer to the evaluation protocols in:

- **AttackVLM**: [https://github.com/yunqing-me/AttackVLM](https://github.com/yunqing-me/AttackVLM)
- **COA (Chain of Attack)**: [https://github.com/Shelton1013/Chain_of_Attacke](https://github.com/Shelton1013/Chain_of_Attacke)

## Acknowledgement

Our code is based on [AttackVLM](https://github.com/yunqing-me/AttackVLM). Thanks for their contributions to the community.
