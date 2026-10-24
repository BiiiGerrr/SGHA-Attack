"""Launch one of the six native caption backends with explicit paths and seeds."""
import argparse
import json
import os
import runpy
import sys
from pathlib import Path
from .common import ROOT,atomic_json,ensure_new_output,image_paths,read_lines,sha256

BACKENDS={
 'unidiffuser':('unidiffuser',100),
 'blip2':('LAVIS',5),
 'instructblip':('LAVIS',2),
 'minigpt4':('MiniGPT-4',20),
 'llava':('LLaVA',1),
 'llava_next':('LLaVA-NeXT',1),
}
MODEL_IDS={'llava':'liuhaotian/llava-v1.5-13b','llava_next':'lmms-lab/llava-next-72b','minigpt4':'Vision-CAIR/vicuna'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=list(BACKENDS),required=True)
    p.add_argument('--images',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,help='defaults to the first-trial seed for the selected victim')
    p.add_argument('--batch-size',type=int)
    p.add_argument('--num-samples',type=int,default=1000)
    p.add_argument('--model-id',help='Override the public LLM or LLaVA HF identifier')
    p.add_argument('--checkpoint',help='Override the MiniGPT-4 projection checkpoint')
    p.add_argument('--vendor-root',default=str(ROOT/'vendor'))
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.seed is None:a.seed=1234 if a.model=='unidiffuser' else 2023
    vendor_name,default_batch=BACKENDS[a.model];bs=default_batch if a.batch_size is None else a.batch_size
    if a.num_samples<=0 or bs<=0 or a.num_samples%bs:raise ValueError('num-samples and batch-size must be positive, with num-samples divisible by batch-size')
    if a.model in ('llava','llava_next') and bs!=1:raise ValueError('The native LLaVA caption programs use batch-size=1')
    script=ROOT/'sgha_attack/backends'/('caption_'+a.model+'.py')
    images=str(Path(a.images).resolve());output=Path(a.output).resolve()
    vendor=Path(a.vendor_root).resolve()/vendor_name
    identifier=a.model_id or MODEL_IDS.get(a.model)
    args=['--batch_size',str(bs)]
    if a.model=='unidiffuser':
        args+=['--config',str(ROOT/'configs/unidiffuser.py'),'--nnet_path','HF_UNIDIFFUSER_WEIGHT','--n_samples',str(a.num_samples),'--img_path',images,'--output',str(output.with_suffix('')),'--mode','i2t']
    elif a.model=='minigpt4':
        args+=['--cfg-path',str(ROOT/'configs/minigpt4_eval.yaml'),'--dataset_path',images,'--save_path',str(output),'--query','what is the content of this image?','--num_beams','1','--max_new_tokens','100','--options','model.llama_model='+identifier,'model.ckpt=HF_MINIGPT4_WEIGHT']
    else:
        args+=['--img_path',images,'--output_path',str(output),'--num_samples',str(a.num_samples)]
        if a.model in ('llava','llava_next'):args+=['--model_path',identifier]
    if a.dry_run:
        print(json.dumps(dict(model=a.model,vendor=str(vendor),seed=a.seed,batch_size=bs,arguments=args),indent=2));return
    ensure_new_output(output)
    files=image_paths(images,a.num_samples)
    if not vendor.is_dir():raise FileNotFoundError('Run scripts/fetch_backends.sh %s first, then install its dependencies.'%a.model)
    if a.model=='minigpt4' and len(image_paths(images))!=a.num_samples:raise ValueError('MiniGPT-4 native script processes the entire directory; supply exactly num-samples images.')
    from huggingface_hub import hf_hub_download
    if a.model=='unidiffuser':
        if output.suffix!='.txt':raise ValueError('UniDiffuser output must end in .txt')
        args[args.index('HF_UNIDIFFUSER_WEIGHT')]=hf_hub_download('thu-ml/unidiffuser-v1','uvit_v1.pth')
    if a.model=='minigpt4':
        ckpt=a.checkpoint or hf_hub_download('Vision-CAIR/MiniGPT-4','pretrained_minigpt4.pth')
        args[-1]='model.ckpt='+str(Path(ckpt).resolve())
    os.environ['SGHA_SEED']=str(a.seed)
    previous_argv=sys.argv[:];previous_path=sys.path[:]
    sys.path.insert(0,str(vendor))
    sys.argv=[str(script)]+args
    # Native UniDiffuser helpers resolve their packaged config/assets from cwd.
    previous=Path.cwd()
    try:
        os.chdir(vendor)
        try:runpy.run_path(str(script),run_name='__main__')
        except SystemExit as exc:
            if exc.code not in (None,0):raise
    finally:
        os.chdir(previous)
        sys.argv=previous_argv
        sys.path[:]=previous_path
    captions=read_lines(output)
    if len(captions)!=a.num_samples:raise RuntimeError('Incomplete caption output: expected %d, found %d'%(a.num_samples,len(captions)))
    atomic_json(output.with_suffix('.run.json'),dict(model=a.model,model_id=identifier,seed=a.seed,n=len(captions),batch_size=bs,images=[dict(file=str(f.relative_to(images)),sha256=sha256(f)) for f in files],caption_sha256=sha256(output)))
    print('Saved %d captions to %s'%(len(captions),output))

if __name__=='__main__':main()
