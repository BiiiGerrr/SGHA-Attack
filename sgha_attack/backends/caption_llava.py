# Portable reference implementation; see docs/REPRODUCIBILITY.md.
import argparse
import os
import random
import torch
import torchvision
import einops
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
DEFAULT_RANDOM_SEED = int(os.environ.get('SGHA_SEED', '2023'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)

class ImageFolderForLavis(torchvision.datasets.ImageFolder):

    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        return (path, path)

def main():
    seedEverything()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_samples', default=1000, type=int)
    parser.add_argument('--img_path', default='outputs/images', type=str)
    parser.add_argument('--output_path', default='outputs/captions/captions.txt', type=str)
    parser.add_argument('--model_path', type=str, default='liuhaotian/llava-v1.5-13b')
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument('--conv_mode', type=str, default='vicuna_v1')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    args = parser.parse_args()
    print(f'Loading LLaVA model: {args.model_path}')
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    print('Model loaded.')
    dataset = ImageFolderForLavis(args.img_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    with open(args.output_path, 'w') as f_out:
        for i, (image_path_tensor, path) in enumerate(tqdm(dataloader)):
            if i >= args.num_samples // args.batch_size:
                break
            start = time.perf_counter()
            img_path = path[0]
            image = Image.open(img_path).convert('RGB')
            image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).half().to(device)
            question = 'what is the content of this image?'
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + question
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
            with torch.inference_mode():
                output_ids = model.generate(input_ids, images=image_tensor, image_sizes=[image.size], do_sample=True if args.temperature > 0 else False, temperature=args.temperature, top_p=args.top_p, num_beams=args.num_beams, max_new_tokens=512, use_cache=True)
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            with open(args.output_path, 'a') as f_out:
                outputs = outputs.replace('\u200b', '').replace('\n', '').replace('\r', '')
                f_out.write(f'{outputs.strip()}\n')
                f_out.flush()
            end = time.perf_counter()
            print(f'Batch {i}: {end - start:.2f}s')
    print('All captions saved to:', args.output_path)
if __name__ == '__main__':
    main()
