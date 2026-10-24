# Portable reference implementation; see docs/REPRODUCIBILITY.md.
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
import open_clip
from pathlib import Path
from huggingface_hub import hf_hub_download
open_clip.add_model_config(Path(__file__).resolve().parents[2] / 'configs/EVA01-g-14.json')
DEFAULT_RANDOM_SEED = int(os.environ.get('SGHA_SEED', '2023'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seedEverything(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_tensor(pic):
    mode_to_nptype = {'I': np.int32, 'I;16': np.int16, 'F': np.float32}
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
            block = model.visual.trunk.blocks[layer_idx]
            hook = block.register_forward_hook(self.generate_hook_fn(layer_idx))
            self.hooks.append(hook)

    def generate_hook_fn(self, layer_idx):

        def hook_fn(module, input, output):
            self.features[layer_idx] = output
        return hook_fn

    def clear(self):
        for layer in self.layers:
            self.features[layer] = None

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

class TextFeatureExtractor:

    def __init__(self, model, layers):
        self.model = model
        self.visual_keys = layers
        self.features = {layer: None for layer in layers}
        self.hooks = []
        self.text_blocks = None
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'resblocks'):
            self.text_blocks = model.transformer.resblocks
        elif hasattr(model, 'text') and hasattr(model.text, 'transformer'):
            if hasattr(model.text.transformer, 'resblocks'):
                self.text_blocks = model.text.transformer.resblocks
        elif hasattr(model, 'text') and hasattr(model.text, 'blocks'):
            self.text_blocks = model.text.blocks
        if self.text_blocks is None:
            raise ValueError('无法找到文本 Transformer blocks。')
        self.text_total_layers = len(self.text_blocks)
        self.visual_total_layers = 12
        if hasattr(model.visual, 'trunk') and hasattr(model.visual.trunk, 'blocks'):
            self.visual_total_layers = len(model.visual.trunk.blocks)
        elif hasattr(model.visual, 'transformer') and hasattr(model.visual.transformer, 'resblocks'):
            self.visual_total_layers = len(model.visual.transformer.resblocks)
        print(f'Debug: Detected Visual Depth={self.visual_total_layers}, Text Depth={self.text_total_layers}')
        for v_idx in self.visual_keys:
            ratio = v_idx / float(self.visual_total_layers)
            t_idx = int(round(ratio * self.text_total_layers))
            t_idx = min(max(t_idx, 0), self.text_total_layers - 1)
            print(f'Mapping Hook: Visual Layer {v_idx} -> Text Layer {t_idx} (Ratio: {ratio:.2f})')
            hook = self.text_blocks[t_idx].register_forward_hook(self.generate_hook_fn(v_idx))
            self.hooks.append(hook)

    def generate_hook_fn(self, layer_key):

        def hook_fn(module, input, output):
            if output.shape[0] == 77:
                out = output.permute(1, 0, 2)
            else:
                out = output
            self.features[layer_key] = out
        return hook_fn

    def clear(self):
        for layer in self.visual_keys:
            self.features[layer] = None

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()

def load_anchor_roots(list_file):
    roots = []
    if not os.path.isfile(list_file):
        raise FileNotFoundError(f'anchor_list_file not found: {list_file}')
    with open(list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '->' not in line:
                continue
            path = line.split('->')[-1].strip()
            if os.path.isdir(path):
                roots.append(path)
    if len(roots) == 0:
        raise RuntimeError(f'No valid anchor root dirs found')
    print(f'Loaded {len(roots)} anchor roots.')
    return roots
if __name__ == '__main__':
    seedEverything()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--num_samples', default=1000, type=int)
    parser.add_argument('--input_res', default=224, type=int)
    parser.add_argument('--steps', default=100, type=int)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--output', default='outputs/images', type=str)
    parser.add_argument('--local_model_path', default='', type=str, help='local path to EVA-CLIP model weights')
    parser.add_argument('--cle_data_path', default='data/clean', type=str)
    parser.add_argument('--tgt_text_path', default='data/metadata/targets_1000.txt', type=str)
    parser.add_argument('--anchor_list_file', default='outputs/anchors/seed_record.txt', type=str)
    parser.add_argument('--num_anchors', default=5, type=int, help='Top-K anchors used per sample')
    parser.add_argument('--anchor_temp', default=5.0, type=float)
    parser.add_argument('--lambda_text', default=1.0, type=float)
    parser.add_argument('--lambda_anchor', default=1.0, type=float)
    parser.add_argument('--tgt_data_path', default=None, type=str)
    parser.add_argument('--no_anchor', help='如果设为 True，则不使用 seed anchors', action='store_true')
    parser.add_argument('--lambda_feature', default=6.0, type=float)
    parser.add_argument('--hook_layers', nargs='+', type=int, default=[23, 30, 37], help='Which layers to hook')
    parser.add_argument('--lambda_cls', default=1.0, type=float)
    parser.add_argument('--lambda_spatial', default=0.8, type=float)
    parser.add_argument('--feature_noise_scale', default=0.0, type=float)
    parser.add_argument('--lambda_text_mid', default=2.5, type=float)
    parser.add_argument('--seed', type=int, default=DEFAULT_RANDOM_SEED)
    args = parser.parse_args()
    seedEverything(args.seed)
    if args.num_samples <= 0 or args.batch_size <= 0 or args.num_samples % args.batch_size:
        raise ValueError('num_samples must be positive and divisible by batch_size for the reference attack.')
    print(f'Running Ablation with EVA-CLIP: Layers={args.hook_layers}')
    use_anchor = not args.no_anchor
    use_tgt_images = args.tgt_data_path is not None and args.tgt_data_path != ''
    if use_anchor and use_tgt_images:
        raise ValueError('不能同时使用 anchor_roots 和 tgt_data_path。')
    print(f'Loading EVA-CLIP from local weights: {args.local_model_path}')
    clip_model, _, preprocess_val = open_clip.create_model_and_transforms('EVA01-g-14', pretrained=None, device=device)
    if not args.local_model_path:
        args.local_model_path = hf_hub_download('QuanSun/EVA-CLIP', 'EVA01_CLIP_g_14_psz14_s11B.pt')
    checkpoint = torch.load(args.local_model_path, map_location=device)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    fixed_state_dict = {}
    for k, v in state_dict.items():
        if k == 'text.logit_scale':
            new_k = 'logit_scale'
            fixed_state_dict[new_k] = v
        elif k.startswith('visual.'):
            new_k = 'visual.trunk.' + k[len('visual.'):]
            fixed_state_dict[new_k] = v
        else:
            fixed_state_dict[k] = v
    missing, unexpected = clip_model.load_state_dict(fixed_state_dict, strict=False)
    print('missing:', missing)
    print('unexpected:', unexpected)
    clip_model.eval()
    print('EVA-CLIP model loaded successfully!')
    tokenizer = open_clip.get_tokenizer('EVA01-g-14')
    visual_proj = None
    candidates = []
    if hasattr(clip_model.visual, 'trunk') and hasattr(clip_model.visual.trunk, 'head'):
        candidates.append(clip_model.visual.trunk.head)
    if hasattr(clip_model.visual, 'head'):
        candidates.append(clip_model.visual.head)
    if hasattr(clip_model.visual, 'proj'):
        p = clip_model.visual.proj
        if p is not None:
            visual_proj = p.detach()
            print('Found Visual Projection in: visual.proj')
    if visual_proj is None:
        for module in candidates:
            if module is None:
                continue
            if isinstance(module, torch.nn.Linear):
                visual_proj = module.weight.t().detach()
                print(f'Found Visual Projection in Linear layer: {module}')
                break
            elif isinstance(module, torch.nn.Sequential):
                found_linear = False
                for m in module:
                    if isinstance(m, torch.nn.Linear):
                        visual_proj = m.weight.t().detach()
                        print(f'Found Visual Projection in Sequential: {m}')
                        found_linear = True
                        break
                if found_linear:
                    break
    if visual_proj is None:
        print('Debug: clip_model.visual keys:', dir(clip_model.visual))
        if hasattr(clip_model.visual, 'trunk'):
            print('Debug: clip_model.visual.trunk keys:', dir(clip_model.visual.trunk))
        raise ValueError('无法在 EVA-CLIP 模型中找到 visual projection 矩阵。请查看上方 Debug 信息。')
    print(f'Visual Projection loaded. Shape: {visual_proj.shape}')
    text_proj = None
    if hasattr(clip_model, 'text') and hasattr(clip_model.text, 'text_projection'):
        text_proj = clip_model.text.text_projection
    elif hasattr(clip_model, 'text_projection'):
        text_proj = clip_model.text_projection
    elif hasattr(clip_model, 'text') and hasattr(clip_model.text, 'head'):
        pass
    if text_proj is not None:
        if isinstance(text_proj, torch.nn.Linear):
            text_proj = text_proj.weight.t().detach()
        else:
            text_proj = text_proj.detach()
        print(f'Text Projection loaded. Shape: {text_proj.shape}')
    else:
        print('Debug: clip_model.text keys:', dir(clip_model.text))
        raise ValueError('无法找到 text projection 矩阵，导致无法进行跨模态维度对齐。')
    extractor = FeatureExtractor(clip_model, layers=args.hook_layers)
    text_extractor = TextFeatureExtractor(clip_model, layers=args.hook_layers)
    transform_fn = torchvision.transforms.Compose([torchvision.transforms.Resize(args.input_res, interpolation=torchvision.transforms.InterpolationMode.BICUBIC), torchvision.transforms.CenterCrop(args.input_res), torchvision.transforms.Lambda(lambda img: img.convert('RGB')), torchvision.transforms.Lambda(lambda img: to_tensor(img))])
    clip_preprocess = torchvision.transforms.Compose([torchvision.transforms.Resize(args.input_res, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True), torchvision.transforms.Lambda(lambda img: img / 255.0), torchvision.transforms.CenterCrop(args.input_res), torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    clean_data = ImageFolderWithPaths(args.cle_data_path, transform=transform_fn)
    data_loader_imagenet = torch.utils.data.DataLoader(clean_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    anchor_loaders = []
    if use_anchor:
        anchor_roots = load_anchor_roots(args.anchor_list_file)
        anchor_datasets = [ImageFolderWithPaths(root, transform=transform_fn) for root in anchor_roots]
        anchor_loaders = [torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4) for ds in anchor_datasets]
    data_loader_tgt = None
    if use_tgt_images:
        tgt_data = ImageFolderWithPaths(args.tgt_data_path, transform=transform_fn)
        data_loader_tgt = torch.utils.data.DataLoader(tgt_data, batch_size=args.batch_size, shuffle=False, num_workers=4)
    with open(args.tgt_text_path, 'r') as f:
        tgt_text = f.readlines()[:args.num_samples]
    if use_anchor:
        attack_iter = zip(data_loader_imagenet, *anchor_loaders)
    elif use_tgt_images:
        attack_iter = zip(data_loader_imagenet, data_loader_tgt)
    else:
        attack_iter = data_loader_imagenet
    for i, batch in enumerate(attack_iter):
        if args.batch_size * (i + 1) > args.num_samples:
            break
        if use_anchor:
            batch_all = batch
            image_org, _, path = batch_all[0]
            anchor_batches = [b[0] for b in batch_all[1:]]
            tgt_batch_img = None
        elif use_tgt_images:
            (image_org, _, path), (tgt_batch_img, _, tgt_paths) = batch
            anchor_batches = []
        else:
            image_org, _, path = batch
            anchor_batches = []
            tgt_batch_img = None
        bs = image_org.size(0)
        batch_texts = tgt_text[i * args.batch_size:(i + 1) * args.batch_size]
        text_tokens = tokenizer(batch_texts).to(device)
        text_extractor.clear()
        with torch.no_grad():
            tgt_text_features = clip_model.encode_text(text_tokens)
            tgt_text_features = tgt_text_features / tgt_text_features.norm(dim=1, keepdim=True)
            eos_indices = text_tokens.argmax(dim=-1)
            text_mid_proj = {}
            for layer_idx in args.hook_layers:
                layer_feat = text_extractor.features[layer_idx]
                if layer_feat is None:
                    raise RuntimeError(f'Text features for layer {layer_idx} empty.')
                B_txt, S_txt, D_t = layer_feat.shape
                layer_feat_flat = layer_feat.reshape(B_txt * S_txt, D_t)
                if text_proj is not None:
                    proj_flat = layer_feat_flat @ text_proj
                else:
                    proj_flat = layer_feat_flat
                proj = proj_flat.view(B_txt, S_txt, -1)
                text_mid_proj[layer_idx] = proj.detach()
        image_org = image_org.to(device)
        if use_anchor and len(anchor_batches) > 0:
            anchor_batches = [ab.to(device) for ab in anchor_batches]
        if use_tgt_images and tgt_batch_img is not None:
            tgt_batch_img = tgt_batch_img.to(device)
        batch_anchor_features = None
        batch_anchor_weights = None
        batch_topk_anchor_imgs = None
        tgt_img_features_norm = None
        target_inter_features = None
        if use_anchor and len(anchor_batches) > 0:
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
            if args.lambda_feature > 0:
                tmp_target = {layer: [] for layer in args.hook_layers}
                with torch.no_grad():
                    for b in range(bs):
                        k_anchors = batch_topk_anchor_imgs[b]
                        k_weights = batch_anchor_weights[b]
                        K_b = k_anchors.size(0)
                        k_weights = (k_weights / k_weights.sum()).view(K_b, 1, 1)
                        extractor.clear()
                        _ = clip_model.encode_image(clip_preprocess(k_anchors))
                        for layer in args.hook_layers:
                            k_feats = extractor.features[layer]
                            weighted_feat = torch.sum(k_feats * k_weights, dim=0)
                            tmp_target[layer].append(weighted_feat)
                target_inter_features = {}
                for layer in args.hook_layers:
                    target_inter_features[layer] = torch.stack(tmp_target[layer], dim=0).detach()
        elif use_tgt_images and tgt_batch_img is not None:
            with torch.no_grad():
                extractor.clear()
                tgt_clip_in = clip_preprocess(tgt_batch_img)
                tgt_feats = clip_model.encode_image(tgt_clip_in)
                tgt_img_features_norm = tgt_feats / tgt_feats.norm(dim=1, keepdim=True)
                if args.lambda_feature > 0:
                    target_inter_features = {}
                    for layer in args.hook_layers:
                        target_inter_features[layer] = extractor.features[layer].detach()
        delta = torch.zeros_like(image_org, requires_grad=True)
        for j in range(args.steps):
            adv_pixels = image_org + delta
            adv_pixels = torch.clamp(adv_pixels, 0.0, 255.0)
            adv_input = clip_preprocess(adv_pixels)
            extractor.clear()
            adv_image_features = clip_model.encode_image(adv_input)
            adv_image_features_norm = adv_image_features / adv_image_features.norm(dim=1, keepdim=True)
            feat_for_loss = adv_image_features_norm
            if args.feature_noise_scale > 0:
                feat_for_loss = feat_for_loss + torch.randn_like(feat_for_loss) * args.feature_noise_scale
            text_sim = torch.sum(feat_for_loss * tgt_text_features, dim=1)
            loss_feature_batch = torch.tensor(0.0, device=device)
            if args.lambda_feature > 0 and target_inter_features is not None:
                layer_losses = []
                for layer_idx in args.hook_layers:
                    adv_layer = extractor.features[layer_idx]
                    tgt_layer = target_inter_features[layer_idx]
                    cls_sim = F.cosine_similarity(adv_layer[:, 0, :], tgt_layer[:, 0, :], dim=-1)
                    l_cls = 1.0 - cls_sim.mean()
                    adv_spatial = torch.mean(adv_layer[:, 1:, :], dim=1)
                    tgt_spatial = torch.mean(tgt_layer[:, 1:, :], dim=1)
                    spatial_sim = F.cosine_similarity(adv_spatial, tgt_spatial, dim=-1)
                    l_spatial = 1.0 - spatial_sim.mean()
                    l_layer = args.lambda_cls * l_cls + args.lambda_spatial * l_spatial
                    layer_losses.append(l_layer)
                loss_feature_batch = torch.mean(torch.stack(layer_losses))
            loss_text_mid = torch.tensor(0.0, device=device)
            if args.lambda_text_mid > 0:
                mid_losses = []
                for layer_idx in args.hook_layers:
                    adv_layer = extractor.features[layer_idx]
                    B_img, S_img, D_v = adv_layer.shape
                    adv_flat = adv_layer.reshape(B_img * S_img, D_v)
                    adv_proj_flat = adv_flat @ visual_proj
                    adv_proj = adv_proj_flat.view(B_img, S_img, -1)
                    adv_cls_c = adv_proj[:, 0, :]
                    txt_proj = text_mid_proj[layer_idx]
                    txt_eos_c = txt_proj[torch.arange(bs, device=device), eos_indices, :]
                    txt_sim = F.cosine_similarity(adv_cls_c, txt_eos_c, dim=-1)
                    mid_losses.append(1.0 - txt_sim.mean())
                loss_text_mid = torch.mean(torch.stack(mid_losses))
            per_sample_loss = []
            for b in range(bs):
                l_total = args.lambda_text * text_sim[b]
                if use_anchor and batch_anchor_features is not None:
                    feats_b = batch_anchor_features[b]
                    weights_b = batch_anchor_weights[b]
                    sims_b = torch.sum(feats_b * adv_image_features_norm[b].unsqueeze(0), dim=1)
                    l_total += args.lambda_anchor * torch.sum(weights_b * sims_b)
                elif not use_anchor and tgt_img_features_norm is not None:
                    img_sim_b = torch.sum(adv_image_features_norm[b] * tgt_img_features_norm[b])
                    l_total += args.lambda_anchor * img_sim_b
                per_sample_loss.append(l_total)
            sim_loss = torch.mean(torch.stack(per_sample_loss))
            total_loss = sim_loss
            if args.lambda_feature > 0:
                total_loss = total_loss - args.lambda_feature * loss_feature_batch
            if args.lambda_text_mid > 0:
                total_loss = total_loss - args.lambda_text_mid * loss_text_mid
            total_loss.backward()
            grad = delta.grad.detach()
            d = torch.clamp(delta + args.alpha * torch.sign(grad), min=-args.epsilon, max=args.epsilon)
            delta.data = d
            delta.grad.zero_()
            if j % 20 == 0:
                print(f'iter {i} step {j}: Obj={total_loss.item():.4f}, Text={torch.mean(text_sim).item():.4f}, FeatLoss={loss_feature_batch.item():.4f}, TxtMid={loss_text_mid.item():.4f}')
        adv_image = image_org + delta
        adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)
        for path_idx in range(len(path)):
            folder, name = (path[path_idx].split('/')[-2], path[path_idx].split('/')[-1])
            folder_to_save = os.path.join(args.output, folder)
            if not os.path.exists(folder_to_save):
                os.makedirs(folder_to_save, exist_ok=True)
            root, ext = os.path.splitext(name)
            save_path = os.path.join(folder_to_save, root + '.png')
            torchvision.utils.save_image(adv_image[path_idx], save_path)
