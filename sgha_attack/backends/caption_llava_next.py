# Portable reference implementation; see docs/REPRODUCIBILITY.md.
import argparse
import os
import random
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import torch
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
DEFAULT_RANDOM_SEED = int(os.environ.get('SGHA_SEED', '2023'))

def seedEverything(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ImageFolderForLavis(torchvision.datasets.ImageFolder):

    def __getitem__(self, index: int):
        path, _ = self.samples[index]
        return (path, path)

def main():
    seedEverything()
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_samples', default=1000, type=int)
    parser.add_argument('--model_path', type=str, default='lmms-lab/llava-next-72b')
    parser.add_argument('--model_base', type=str, default=None)
    parser.add_argument('--conv_mode', type=str, default='qwen_1_5')
    parser.add_argument('--img_path', default='outputs/images', type=str)
    parser.add_argument('--output_path', default='outputs/captions/captions.txt', type=str)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--load_4bit', default=True, help='Use 4-bit quantization')
    parser.add_argument('--load_8bit', default=False, help='Use 8-bit quantization')
    args = parser.parse_args()
    print(f'Loading LLaVA-NeXT model: {args.model_path}')
    disable_torch_init()
    model_name = 'llava-next-72b-qwen'
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, load_4bit=args.load_4bit, load_8bit=args.load_8bit, device_map='auto', attn_implementation='flash_attention_2')
    print('Model loaded.')
    dataset = ImageFolderForLavis(args.img_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    output_dir = os.path.dirname(args.output_path)
    if output_dir and (not os.path.exists(output_dir)):
        os.makedirs(output_dir)
    print(f'Start inference with template: {args.conv_mode}')
    with open(args.output_path, 'w', encoding='utf-8') as f_out:
        for i, (image_path_tensor, path) in enumerate(tqdm(dataloader)):
            if i >= args.num_samples // args.batch_size:
                break
            try:
                img_path = path[0]
                image = Image.open(img_path).convert('RGB')
                image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).to(model.device, dtype=torch.float16)
                question = 'what is the content of this image?'
                if args.conv_mode == 'qwen_1_5':
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + question
                elif model.config.mm_use_im_start_end:
                    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
                else:
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + question
                conv = conv_templates[args.conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
                with torch.inference_mode():
                    output_ids = model.generate(input_ids, images=image_tensor, image_sizes=[image.size], do_sample=True if args.temperature > 0 else False, temperature=args.temperature, top_p=args.top_p, num_beams=args.num_beams, max_new_tokens=512, use_cache=True)
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                clean_output = outputs.replace('\u200b', '').replace('\n', ' ').replace('\r', '')
                f_out.write(f'{clean_output.strip()}\n')
                f_out.flush()
            except Exception as e:
                print(f'Error processing {path[0]}: {e}')
                continue
    print('All captions saved to:', args.output_path)
if __name__ == '__main__':
    main()
