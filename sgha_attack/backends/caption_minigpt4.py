# Portable reference implementation; see docs/REPRODUCIBILITY.md.
import os
import argparse
import random
from PIL import Image
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import re
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.datasets.builders import *
from minigpt4.runners import *
from minigpt4.tasks import *
DEFAULT_RANDOM_SEED = int(os.environ.get('SGHA_SEED', '2023'))

def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)

class ImagePathDataset(Dataset):

    def __init__(self, root_dir, vis_processor, valid_exts=('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        self.vis_processor = vis_processor
        self.samples = []
        subfolders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        subfolders = sorted(subfolders)
        for folder in subfolders:
            class_folder = os.path.join(root_dir, folder)
            entries = sorted(os.listdir(class_folder))
            for fname in entries:
                full_path = os.path.join(class_folder, fname)
                if not os.path.isfile(full_path):
                    continue
                if not fname.lower().endswith(valid_exts):
                    continue
                self.samples.append(full_path)
        if len(self.samples) == 0:
            entries = sorted(os.listdir(root_dir))
            for fname in entries:
                full_path = os.path.join(root_dir, fname)
                if os.path.isfile(full_path) and fname.lower().endswith(valid_exts):
                    self.samples.append(full_path)
        print(f'[Dataset] Collected {len(self.samples)} images.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.vis_processor(img)
        return (img_tensor, img_path)

def build_prompt(query_text: str):
    return f'###Human: <Img><ImageHere></Img> {query_text} ###Assistant:'

def clean_answer(ans_raw: str):
    txt = ans_raw.replace('\u200b', '').strip()
    stop_markers = ['###Human:', '### Human:', '###Assistant:', '### Assistant:', '</s>']
    cut_idx = len(txt)
    for m in stop_markers:
        if m in txt:
            cut_idx = min(cut_idx, txt.index(m))
    txt = txt[:cut_idx].strip()
    txt = re.sub('\\s+', ' ', txt)
    return txt
if __name__ == '__main__':
    seedEverything()
    parser = argparse.ArgumentParser(description='MiniGPT-4 batch caption generation (no Chat)')
    parser.add_argument('--cfg-path', default='configs/minigpt4_eval.yaml', help='path to configuration file.')
    parser.add_argument('--gpu-id', type=int, default=0, help='specify the gpu to load the model.')
    parser.add_argument('--options', nargs='+', help='override some settings in the used config, key-value pair in xxx=yyy format will be merged into config file.')
    parser.add_argument('--query', default='what is the content of this image?', type=str)
    parser.add_argument('--dataset_path', default='outputs/images', type=str)
    parser.add_argument('--save_path', default='outputs/captions/captions.txt', type=str)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    args = parser.parse_args()
    print('Loading MiniGPT-4 model...')
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')
    model.eval()
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    print('Done loading model & processor.')
    dataset = ImagePathDataset(root_dir=args.dataset_path, vis_processor=vis_processor)

    def collate_fn(batch):
        imgs = [item[0] for item in batch]
        paths = [item[1] for item in batch]
        imgs = torch.stack(imgs, dim=0)
        return (imgs, paths)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    print('Start generating captions ...')
    global_index = 0
    for images, paths in dataloader:
        images = images.to(f'cuda:{args.gpu_id}', non_blocking=True)
        batch_prompts = [build_prompt(args.query) for _ in range(images.size(0))]
        answers = model.generate(images=images, texts=batch_prompts, num_beams=args.num_beams, max_new_tokens=args.max_new_tokens, min_length=1, top_p=args.top_p, repetition_penalty=args.repetition_penalty, length_penalty=1, temperature=args.temperature, do_sample=False, stop_words_ids=[2])
        for ans_raw, p in zip(answers, paths):
            print('ans_raw:', ans_raw)
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            clean_msg = clean_answer(ans_raw)
            print('clean_msg:', clean_msg)
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print(f'Num: {global_index} [DEBUG]', repr(ans_raw))
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            with open(args.save_path, 'a', encoding='utf-8') as f:
                f.write(clean_msg + '\n')
            global_index += 1
    print('Caption generated successfully.')
