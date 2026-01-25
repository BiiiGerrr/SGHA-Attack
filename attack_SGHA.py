import argparse
import os
import random
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
# seed for everything
DEFAULT_RANDOM_SEED = 2023
device = "cuda" if torch.cuda.is_available() else "cpu"

def diverse_input(image, resize_rate=1.1, diversity_prob=0.5):
    """
    Diverse Input (DI) transformation for adversarial attacks
    Official arguments: resize_rate=1.1, diversity_prob=0.5
    """
 
    if torch.rand(1).item() > diversity_prob:
        return image

    # 获取输入图像的尺寸
    img_size = image.shape[-1]  # 假设是正方形图像 [C, H, W]

    # 计算放大后的尺寸
    img_resize = int(img_size * resize_rate)

    # 在原始尺寸和放大尺寸之间随机选择一个中间尺寸
    rnd = torch.randint(
        low=min(img_size, img_resize),
        high=max(img_size, img_resize),
        size=(1,),
        dtype=torch.int32
    ).item()

    # 1. 将输入图像缩放到随机选择的中间尺寸
    rescaled = F.interpolate(
        image,
        size=[rnd, rnd],
        mode='bilinear',
        align_corners=False
    )

    # 2. 随机添加padding到放大尺寸(img_resize)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(0, h_rem + 1, (1,), dtype=torch.int32).item()
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem + 1, (1,), dtype=torch.int32).item()
    pad_right = w_rem - pad_left

    # 应用padding（填充值为0）
    padded = F.pad(
        rescaled,
        [pad_left, pad_right, pad_top, pad_bottom],
        value=0
    )

    # 3. 将图像缩放回原始尺寸
    return F.interpolate(
        padded,
        size=[img_size, img_size],
        mode='bilinear',
        align_corners=False
    )

def seedEverything(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)


class FeatureExtractor:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.features = {layer: [] for layer in layers}
        self.hooks = []

        for layer_idx in layers:
            # CLIP ViT visual structure
            hook = model.visual.transformer.resblocks[layer_idx].register_forward_hook(
                self.generate_hook_fn(layer_idx)
            )
            self.hooks.append(hook)

    def generate_hook_fn(self, layer_idx):
        def hook_fn(module, input, output):
            # CLIP visual output: [Seq_Len, Batch, Dim] -> Permute to [Batch, Seq, Dim]
            out = output.permute(1, 0, 2)
            self.features[layer_idx] = out

        return hook_fn

    def clear(self):
        for layer in self.layers:
            self.features[layer] = None

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


# ==== NEW: 文本中间层 hook ====
class TextFeatureExtractor:
    """
    Hook CLIP 文本 transformer 的中间层输出:
    CLIP text transformer: output shape [Seq, Batch, Dim]，这里统一转为 [Batch, Seq, Dim]
    """
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.features = {layer: None for layer in layers}
        self.hooks = []

        for layer_idx in layers:
            hook = self.model.transformer.resblocks[layer_idx].register_forward_hook(
                self.generate_hook_fn(layer_idx)
            )
            self.hooks.append(hook)

    def generate_hook_fn(self, layer_idx):
        def hook_fn(module, input, output):
            out = output.permute(1, 0, 2)  # [Seq, B, D] -> [B, Seq, D]
            self.features[layer_idx] = out
        return hook_fn

    def clear(self):
        for layer in self.layers:
            self.features[layer] = None

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


def load_anchor_roots(list_file):
    roots = []
    if not os.path.isfile(list_file):
        raise FileNotFoundError(f"anchor_list_file not found: {list_file}")
    with open(list_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "->" not in line:
                continue
            path = line.split("->")[-1].strip()
            if os.path.isdir(path):
                roots.append(path)
    if len(roots) == 0:
        raise RuntimeError(f"No valid anchor root dirs found")
    print(f"Loaded {len(roots)} anchor roots.")
    return roots


if __name__ == "__main__":
    seedEverything()
    parser = argparse.ArgumentParser()

    # --- 基础设置 ---
    parser.add_argument("--batch_size", default=250, type=int)
    parser.add_argument("--num_samples", default=1000, type=int)
    parser.add_argument("--input_res", default=224, type=int)
    parser.add_argument("--clip_encoder", default="ViT-B/32", type=str)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--output", default="", type=str)

    # --- 路径设置 ---
    parser.add_argument("--cle_data_path", default='', type=str)
    parser.add_argument("--tgt_text_path", default='', type=str)
    parser.add_argument("--anchor_list_file", default="/seed_record.txt", type=str)
    # seed_record.txt
    # 01: 853997548 -> / path/ seed - 853997548
    # 02: 2365236890 -> / path / seed - 2365236890
    # .......
    # 19: 1128190746 -> / path / seed - 1128190746
    # 20: 4225157191 -> / path / seed - 4225157191

    # --- Anchor 基础参数 ---
    parser.add_argument("--num_anchors", default=5, type=int, help="Top-K anchors used per sample")
    parser.add_argument("--anchor_temp", default=5.0, type=float)
    parser.add_argument("--lambda_text", default=1.0, type=float)
    parser.add_argument("--lambda_anchor", default=1.0, type=float)

    # parser.add_argument("--tgt_data_path",default="/data/luowq_group/VLM-dataset/sd_coco_captions_1000_image",type=str,help="(Ablation) 目标图像的路径，结构与 cle_data_path 一致，一一对应 clean 图像")
    parser.add_argument("--tgt_data_path",default="",type=str,help="(Ablation) 目标图像的路径，结构与 cle_data_path 一致，一一对应 clean 图像")
    parser.add_argument("--no_anchor",default=False, help="如果设为 True，则不使用 seed anchors，而是使用 tgt_data_path 作为直接图像目标"
    )
    # --- [关键参数化] Hook 与 Loss 系数 ---
    parser.add_argument("--lambda_feature", default=1.5, type=float,
                        help="Total weight for intermediate feature loss")

    # nargs='+' 允许输入列表，例如: --hook_layers 9 11
    parser.add_argument("--hook_layers", nargs='+', type=int, default=[7, 9, 11],
                        help="Which layers to hook (e.g., 9 11 for ViT-B/32)")

    parser.add_argument("--lambda_cls", default=1.0, type=float,
                        help="Weight for [CLS] token directional alignment (Global Semantic)")
    parser.add_argument("--lambda_spatial", default=0.7, type=float,
                        help="Weight for Spatial tokens directional alignment (Local Texture)")

    parser.add_argument("--feature_noise_scale", default=0.0, type=float, help="Noise added to features")

    # ==== NEW: 文本中间层 cross-modal 对齐权重 ====
    parser.add_argument("--lambda_text_mid", default=2.5, type=float,
                        help="Weight for mid-layer image-text CLS/EOS alignment loss (0 to disable)")

    args = parser.parse_args()
    print(f"Running Ablation: Layers={args.hook_layers}, w_CLS={args.lambda_cls}, "
          f"w_Spatial={args.lambda_spatial}, lambda_text_mid={args.lambda_text_mid}")

    use_anchor = not args.no_anchor
    use_tgt_images = (args.tgt_data_path is not None and args.tgt_data_path != "")

    if use_anchor and use_tgt_images:
        raise ValueError(
            "不能同时使用 anchor_roots 和 tgt_data_path。"
            "如果要做无锚点消融，请加 --no_anchor 并指定 --tgt_data_path。"
        )

    if (not use_anchor) and (not use_tgt_images):
        print("[Warning] no_anchor=True 且未提供 tgt_data_path，将只使用文本相关的损失进行攻击（无图像指导）。")

    # Load Model
    clip_model, preprocess = clip.load(args.clip_encoder, device=device)

    # 检查并拿到 CLIP 自带的 projection 矩阵  ==== NEW ====
    visual_proj = getattr(clip_model.visual, "proj", None)
    if visual_proj is None:
        raise ValueError("clip_model.visual.proj is None; 当前代码的中间层投影只支持 ViT 结构的 CLIP 模型。")

    text_proj = getattr(clip_model, "text_projection", None)
    # text_proj 可能为 None（比如有些实现 text hidden dim == embed dim），下面会做分支判断

    # 初始化 Feature Extractor (根据参数选择层数)
    extractor = FeatureExtractor(clip_model, layers=args.hook_layers)

    # 文本中间层 extractor  ==== NEW ====
    text_extractor = TextFeatureExtractor(clip_model, layers=args.hook_layers)

    # Preprocessing
    transform_fn = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.input_res, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        torchvision.transforms.CenterCrop(args.input_res),
        torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
        torchvision.transforms.Lambda(lambda img: to_tensor(img)),
    ])
    clip_preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(clip_model.visual.input_resolution,
                                      interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
        torchvision.transforms.Lambda(lambda img: img / 255.0),
        torchvision.transforms.CenterCrop(clip_model.visual.input_resolution),
        torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    # Data Loaders
    clean_data = ImageFolderWithPaths(args.cle_data_path, transform=transform_fn)
    data_loader_imagenet = torch.utils.data.DataLoader(
        clean_data, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    anchor_loaders = []
    if use_anchor:
        anchor_roots = load_anchor_roots(args.anchor_list_file)
        anchor_datasets = [ImageFolderWithPaths(root, transform=transform_fn) for root in anchor_roots]
        anchor_loaders = [
            torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
            for ds in anchor_datasets
        ]

    # NEW: 目标图像 DataLoader，用于无 anchor 消融
    data_loader_tgt = None
    if use_tgt_images:
        tgt_data = ImageFolderWithPaths(args.tgt_data_path, transform=transform_fn)
        data_loader_tgt = torch.utils.data.DataLoader(
            tgt_data, batch_size=args.batch_size, shuffle=False, num_workers=0
        )

    with open(args.tgt_text_path, 'r') as f:
        tgt_text = f.readlines()[:args.num_samples]

    # --- Attack Loop ---
    if use_anchor:
        attack_iter = zip(data_loader_imagenet, *anchor_loaders)
    elif use_tgt_images:
        # 无 anchor，使用 clean + target 图像一一对应
        attack_iter = zip(data_loader_imagenet, data_loader_tgt)
    else:
        # 完全无图像指导，只用 clean 图像本身 + 文本
        attack_iter = data_loader_imagenet


    # for i, batch_all in enumerate(zip(data_loader_imagenet, *anchor_loaders)):
    for i, batch in enumerate(attack_iter):
        if args.batch_size * (i + 1) > args.num_samples:
            break

        if use_anchor:
            batch_all = batch
            (image_org, _, path) = batch_all[0]
            anchor_batches = [b[0] for b in batch_all[1:]]  # list[Tensor]
            tgt_batch_img = None
        elif use_tgt_images:
            (image_org, _, path), (tgt_batch_img, _, tgt_paths) = batch
            anchor_batches = []
        else:
            (image_org, _, path) = batch
            anchor_batches = []
            tgt_batch_img = None

        bs = image_org.size(0)

        # Text Features
        batch_texts = tgt_text[i * args.batch_size: (i + 1) * args.batch_size]
        text_tokens = clip.tokenize(batch_texts, truncate=True).to(device)

        # ==== NEW: 提取文本中间层并用 text_projection 投到公共空间 ====
        text_extractor.clear()
        with torch.no_grad():
            tgt_text_features = clip_model.encode_text(text_tokens)
            tgt_text_features = tgt_text_features / tgt_text_features.norm(dim=1, keepdim=True)

            # eos index（CLIP: 用 argmax 找到文本中最后一个非 padding token）
            eos_indices = text_tokens.argmax(dim=-1)  # [B]

            # 预先把各层文本中间特征投到公共空间，避免 PGD 内重复投影
            text_mid_proj = {}
            for layer_idx in args.hook_layers:
                layer_feat = text_extractor.features[layer_idx]  # [B, Seq_txt, D_t]
                if layer_feat is None:
                    raise RuntimeError(f"Text features for layer {layer_idx} not captured. "
                                       f"Check text_hook_layers and CLIP text transformer depth.")
                B_txt, S_txt, D_t = layer_feat.shape
                layer_feat_flat = layer_feat.view(B_txt * S_txt, D_t)

                if text_proj is not None:
                    proj_flat = layer_feat_flat @ text_proj  # [B*S_txt, D_c]
                else:
                    proj_flat = layer_feat_flat  # D_t == D_c 的情况

                proj = proj_flat.view(B_txt, S_txt, -1)  # [B, S_txt, D_c]
                text_mid_proj[layer_idx] = proj.detach()  # detach: 文本不参与梯度

        # image_org = image_org.to(device)
        image_org = image_org.to(device)
        if use_anchor and len(anchor_batches) > 0:
            anchor_batches = [ab.to(device) for ab in anchor_batches]
        if use_tgt_images and tgt_batch_img is not None:
            tgt_batch_img = tgt_batch_img.to(device)
        # anchor_batches = [ab.to(device) for ab in anchor_batches]

        batch_anchor_features = None
        batch_anchor_weights = None
        batch_topk_anchor_imgs = None
        tgt_img_features_norm = None
        target_inter_features = None  # dict[layer] -> [B, Seq_img, D_v]

        if use_anchor and len(anchor_batches) > 0:
            # 1. Anchor Selection & Top-K Collection
            batch_anchor_features = []
            batch_anchor_weights = []
            batch_topk_anchor_imgs = []

            for b in range(bs):
                anchor_imgs = [ab[b] for ab in anchor_batches]
                anchor_batch = torch.stack(anchor_imgs, dim=0)

                with torch.no_grad():
                    extractor.clear()
                    anchor_clip = clip_preprocess(anchor_batch)
                    anchor_feats = clip_model.encode_image(anchor_clip)
                    anchor_feats = anchor_feats / anchor_feats.norm(dim=1, keepdim=True)

                text_feat_b = tgt_text_features[b:b + 1]
                sims = (anchor_feats @ text_feat_b.t()).squeeze(1)
                M = anchor_feats.size(0)
                K = min(args.num_anchors, M)

                top_vals, top_idx = torch.topk(sims, k=K, dim=0)
                selected_feats = anchor_feats[top_idx]
                weights = torch.softmax(args.anchor_temp * top_vals, dim=0)

                batch_anchor_features.append(selected_feats.detach())
                batch_anchor_weights.append(weights.detach())
                batch_topk_anchor_imgs.append(anchor_batch[top_idx])

            # 2. Compute Weighted Target Features
            if args.lambda_feature > 0:
                tmp_target = {layer: [] for layer in args.hook_layers}
                with torch.no_grad():
                    for b in range(bs):
                        k_anchors = batch_topk_anchor_imgs[b]  # [K_b, C, H, W]
                        k_weights = batch_anchor_weights[b]  # [K_b]
                        K_b = k_anchors.size(0)
                        k_weights = (k_weights / k_weights.sum()).view(K_b, 1, 1)

                        extractor.clear()
                        _ = clip_model.encode_image(clip_preprocess(k_anchors))
                        for layer in args.hook_layers:
                            k_feats = extractor.features[layer]  # [K_b, Seq_img, D_v]
                            weighted_feat = torch.sum(k_feats * k_weights, dim=0)  # [Seq_img, D_v]
                            tmp_target[layer].append(weighted_feat)

                target_inter_features = {}
                for layer in args.hook_layers:
                    target_inter_features[layer] = torch.stack(tmp_target[layer], dim=0).detach()

        elif use_tgt_images and tgt_batch_img is not None:
            # NEW: 无 anchor，直接用一一对应的目标图像
            with torch.no_grad():
                extractor.clear()
                tgt_clip_in = clip_preprocess(tgt_batch_img)
                tgt_feats = clip_model.encode_image(tgt_clip_in)
                tgt_img_features_norm = tgt_feats / tgt_feats.norm(dim=1, keepdim=True)

                if args.lambda_feature > 0:
                    target_inter_features = {}
                    for layer in args.hook_layers:
                        # [B, Seq_img, D_v]
                        target_inter_features[layer] = extractor.features[layer].detach()
        else:
            # 既没有 anchor 也没有目标图像：target_inter_features 保持 None，只用文本相关的 loss
            pass

        # 3. PGD Optimization
        delta = torch.zeros_like(image_org, requires_grad=True)

        for j in range(args.steps):
            adv_pixels = image_org + delta
            # adv_pixels = diverse_input(adv_pixels)
            adv_pixels = torch.clamp(adv_pixels, 0.0, 255.0)
            adv_input = clip_preprocess(adv_pixels)

            extractor.clear()
            adv_image_features = clip_model.encode_image(adv_input)
            adv_image_features_norm = adv_image_features / adv_image_features.norm(dim=1, keepdim=True)

            # Feature Space Augmentation
            feat_for_loss = adv_image_features_norm
            if args.feature_noise_scale > 0:
                feat_for_loss = feat_for_loss + torch.randn_like(feat_for_loss) * args.feature_noise_scale

            # --- Loss Calculation ---
            text_sim = torch.sum(feat_for_loss * tgt_text_features, dim=1)

            # Feature Loss (Parametrized)
            loss_feature_batch = torch.tensor(0.0, device=device)
            if args.lambda_feature > 0 and target_inter_features is not None:
                layer_losses = []
                for layer_idx in args.hook_layers:
                    adv_layer = extractor.features[layer_idx]  # [B, Seq_img, D_v]
                    tgt_layer = target_inter_features[layer_idx]  # [B, Seq_img, D_v]

                    # A. CLS Token Directional Loss (Global Semantic)
                    cls_sim = F.cosine_similarity(adv_layer[:, 0, :], tgt_layer[:, 0, :], dim=-1)
                    l_cls = 1.0 - cls_sim.mean()

                    # B. Spatial Token Directional Loss (Texture/Style)
                    adv_spatial = torch.mean(adv_layer[:, 1:, :], dim=1)  # [B, D_v]
                    tgt_spatial = torch.mean(tgt_layer[:, 1:, :], dim=1)  # [B, D_v]
                    spatial_sim = F.cosine_similarity(adv_spatial, tgt_spatial, dim=-1)
                    l_spatial = 1.0 - spatial_sim.mean()

                    l_layer = args.lambda_cls * l_cls + args.lambda_spatial * l_spatial
                    layer_losses.append(l_layer)

                loss_feature_batch = torch.mean(torch.stack(layer_losses))

            # ==== NEW: 图像 CLS 中间层 vs 文本 EOS 中间层，在公共空间 D_c 上对齐 ====
            loss_text_mid = torch.tensor(0.0, device=device)
            if args.lambda_text_mid > 0:
                mid_losses = []
                for layer_idx in args.hook_layers:
                    adv_layer = extractor.features[layer_idx]  # [B, Seq_img, D_v]
                    B_img, S_img, D_v = adv_layer.shape
                    adv_flat = adv_layer.view(B_img * S_img, D_v)
                    # 映射到公共空间 D_c
                    adv_proj_flat = adv_flat @ visual_proj      # [B*S_img, D_c]
                    adv_proj = adv_proj_flat.view(B_img, S_img, -1)  # [B, S_img, D_c]

                    adv_cls_c = adv_proj[:, 0, :]  # [B, D_c]  CLS token in common space

                    txt_proj = text_mid_proj[layer_idx]  # [B, S_txt, D_c]
                    txt_eos_c = txt_proj[torch.arange(bs, device=device), eos_indices, :]  # [B, D_c]

                    txt_sim = F.cosine_similarity(adv_cls_c, txt_eos_c, dim=-1)
                    mid_losses.append(1.0 - txt_sim.mean())

                loss_text_mid = torch.mean(torch.stack(mid_losses))

            per_sample_loss = []
            for b in range(bs):
                l_total = args.lambda_text * text_sim[b]

                if use_anchor and batch_anchor_features is not None:
                    feats_b = batch_anchor_features[b]  # [K_b, D]
                    weights_b = batch_anchor_weights[b]  # [K_b]
                    sims_b = torch.sum(feats_b * adv_image_features_norm[b].unsqueeze(0), dim=1)
                    l_total += args.lambda_anchor * torch.sum(weights_b * sims_b)
                elif (not use_anchor) and (tgt_img_features_norm is not None):
                    # NEW: 无 anchor 时，adv 图像直接向 tgt 图像靠拢
                    img_sim_b = torch.sum(
                        adv_image_features_norm[b] * tgt_img_features_norm[b]
                    )
                    l_total += args.lambda_anchor * img_sim_b

                per_sample_loss.append(l_total)

            sim_loss = torch.mean(torch.stack(per_sample_loss))

            # Optimization Objective:
            # maximize sim_loss, minimize feature & text_mid loss
            total_loss = sim_loss
            if args.lambda_feature > 0:
                total_loss = total_loss - args.lambda_feature * loss_feature_batch
            if args.lambda_text_mid > 0:
                total_loss = total_loss - args.lambda_text_mid * loss_text_mid

            total_loss.backward()

            grad = delta.grad.detach()
            d = torch.clamp(delta + args.alpha * torch.sign(grad),
                            min=-args.epsilon, max=args.epsilon)
            delta.data = d
            delta.grad.zero_()

            if j % 20 == 0:
                print(
                    f"iter {i} step {j}: "
                    f"Obj={total_loss.item():.4f}, "
                    f"Text={torch.mean(text_sim).item():.4f}, "
                    f"FeatLoss={loss_feature_batch.item():.4f}, "
                    f"TxtMid={loss_text_mid.item():.4f}"
                )

        # Save images...
        adv_image = image_org + delta
        adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

        for path_idx in range(len(path)):
            folder, name = path[path_idx].split("/")[-2], path[path_idx].split("/")[-1]
            folder_to_save = os.path.join(args.output, folder)
            if not os.path.exists(folder_to_save):
                os.makedirs(folder_to_save, exist_ok=True)
            root, ext = os.path.splitext(name)
            save_path = os.path.join(folder_to_save, root + ".png")
            torchvision.utils.save_image(adv_image[path_idx], save_path)
