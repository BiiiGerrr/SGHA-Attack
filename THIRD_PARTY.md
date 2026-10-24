# Upstream code and weights

SGHA-Attack builds on [AttackVLM](https://github.com/yunqing-me/AttackVLM).
Anchor generation retains CompVis sampling code and depends on
[Stable Diffusion](https://github.com/CompVis/stable-diffusion).
See its source license and the checkpoint's CreativeML Open RAIL-M terms.

Native integrations use [OpenAI CLIP](https://github.com/openai/CLIP),
[OpenCLIP](https://github.com/mlfoundations/open_clip),
[EVA-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP),
[UniDiffuser](https://github.com/thu-ml/unidiffuser) (AGPL-3.0),
[LAVIS](https://github.com/salesforce/LAVIS) (BSD-3-Clause),
[MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4),
[LLaVA](https://github.com/haotian-liu/LLaVA), and
[LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT).

External repositories are fetched into ignored `vendor/` with their licenses
intact. Base-LLM and checkpoint licenses remain applicable. Model weights and
ImageNet images are not included. Metadata contains selected image filenames and
target descriptions only.
