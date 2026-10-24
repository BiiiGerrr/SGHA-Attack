# SGHA-Attack

SGHA-Attack的官方实现，包含锚点生成、三套CLIP替代模型攻击、六个受害模型的
描述生成，以及CLIP和GPT评估。仓库提供实验配置、19个锚点种子及各阶段的执行脚本。

完整命令见 [README](README.md)，各后端独立环境见 [INSTALLATION](docs/INSTALLATION.md)。
流程为：**准备数据 → 生成锚点 → 攻击 → 生成描述 → CLIP/GPT评估**。

## 1. 安装与准备数据

在Linux的Python 3.10环境运行，依赖和权重直接在执行机器上下载：

```bash
git clone https://github.com/BiiiGerrr/SGHA-Attack.git
cd SGHA-Attack
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/core.txt
python -m pip install -e . --no-deps
bash scripts/check.sh
```

自行获得 `data/metadata/clean_images_1000.txt` 列出的ImageNet图像，保留相对路径，
放在 `data/clean/images/`。目标文本为 `data/metadata/targets_1000.txt`。
第i行目标、第i张按ImageFolder排序的图像和每个锚点池第i张图像必须一一对应。

## 2. 生成锚点

```bash
bash scripts/fetch_backends.sh anchors
python -m pip install -r requirements/anchors.txt
python -m pip install -e vendor/stable-diffusion --no-deps
GPU=0 bash scripts/generate_anchors.sh
python -m sgha_attack.prepare_data --clean-root data/clean \
  --anchor-record outputs/anchors/seed_record.txt
```

`configs/anchor_seeds.json` 为原实验选定的19个种子，脚本按原顺序执行。
可通过ANCHOR_SEEDS_FILE指定其他列表；Top-K=5至少需要5个池。
生成的seed_record与推理结果由Git忽略。
保留原PLMS、50步、guidance 7.5、batch 10、512分辨率、安全处理和水印。
需要HF认证的权重请在自己的终端设置HF_TOKEN并接受对应使用条款。

## 3. 攻击与描述生成

```bash
GPU=0 SEED=2023 bash scripts/attack_all.sh
```

默认100步、epsilon=8/255、alpha=1像素、Top-K=5、anchor_temp=5。
三个配置均位于 `configs/attack_profiles.json`，入口会显式传入原生攻击程序。
攻击默认种子为2023；默认输出目录以seed_2023结尾。

| 受害模型 | 替代模型 | 原生后端 | batch |
|---|---|---|---:|
| UniDiffuser | ViT-B/32 | unidiffuser | 100 |
| BLIP-2 | EVA01-g/14 | LAVIS | 5 |
| InstructBLIP | EVA01-g/14 | LAVIS | 2 |
| MiniGPT-4 | EVA01-g/14 | MiniGPT-4 | 20 |
| LLaVA | ViT-L/14-336 | LLaVA | 1 |
| LLaVA-NeXT | ViT-L/14-336 | LLaVA-NeXT | 1 |

先按INSTALLATION安装对应独立环境，再运行，例如：

```bash
GPU=0 bash scripts/caption.sh llava outputs/attacks/vit_l/seed_2023 outputs/captions/llava.txt 2023
GPU=0 bash scripts/caption.sh minigpt4 data/clean outputs/captions/clean_minigpt4.txt 2023
```

首次描述生成种子：UniDiffuser为1234，其余本地模型为2023。
所有模型的GPT评估统一使用同一份MiniGPT-4干净描述。
LLaVA和LLaVA-NeXT使用独立环境；NeXT多卡可使用GPU=0,1。
保留原提示、采样/解码和停止条件，输出每图一行描述及哈希清单。
每次运行使用新的输出文件名。

## 4. CLIP与GPT评分

在CLIP环境运行；GPT环境另安装 `requirements/api.txt` 并在终端设置自己的
OPENAI_API_KEY。需要不同解释器时设置API_PYTHON。

```bash
ORIGINAL_CAPTIONS=outputs/captions/clean_minigpt4.txt \
  GPU=0 bash scripts/evaluate.sh outputs/captions/llava.txt outputs/evaluation/llava
```

CLIP为五个编码器上的描述与目标文本余弦分数均值。
GPT使用gpt-4.1-mini、温度0和原三档评分提示：ASR_fool统计score>=0.5，
ASR_target统计score=1，均为百分比。
失败、截断或无效评分不计为0；重跑只补缺失记录，全部有效后才生成最终ASR。

```bash
bash scripts/check.sh
```

检查覆盖攻击参数、六个描述后端入口、数据与输出校验，以及评分断点续评。
