"""Generate the paper's fixed-seed anchor pools with the native PLMS sampler."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .common import ROOT,read_lines

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--targets',default=str(ROOT/'data/metadata/targets_1000.txt'))
    p.add_argument('--output-root',default='outputs/anchors')
    p.add_argument('--seeds',default=str(ROOT/'configs/anchor_seeds.json'),help='JSON list of distinct integer seeds')
    p.add_argument('--vendor-root',default=str(ROOT/'vendor/stable-diffusion'))
    p.add_argument('--checkpoint',help='optional original Stable Diffusion full-EMA checkpoint')
    p.add_argument('--batch-size',type=int,default=10)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();seeds=json.loads(Path(a.seeds).read_text())
    targets=read_lines(a.targets)
    if a.batch_size<=0 or len(targets)%a.batch_size:raise ValueError('Target count must be divisible by positive batch-size.')
    if not isinstance(seeds,list) or len(seeds)<5 or any(type(s) is not int or not 0<=s<2**32 for s in seeds):raise ValueError('Supply at least five integer seeds in [0, 2**32) for the default Top-K=5 attack.')
    if len(set(seeds))!=len(seeds):raise ValueError('Duplicate anchor seeds')
    output=Path(a.output_root).resolve();vendor=Path(a.vendor_root).resolve()
    plan=dict(seeds=seeds,n=len(targets),batch_size=a.batch_size,sampler='PLMS',steps=50,guidance=7.5,checkpoint=a.checkpoint or 'hf://CompVis/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt',output=str(output))
    if a.dry_run:print(json.dumps(plan,indent=2));return
    if output.exists():raise FileExistsError('Use a new anchor output-root; partially generated pools must not be mixed.')
    if not (vendor/'ldm').is_dir():raise FileNotFoundError('Run scripts/fetch_backends.sh anchors first.')
    from huggingface_hub import hf_hub_download
    ckpt=a.checkpoint or hf_hub_download('CompVis/stable-diffusion-v-1-4-original','sd-v1-4-full-ema.ckpt')
    output.mkdir(parents=True)
    record=output/'seed_record.txt'
    env={**os.environ,'PYTHONPATH':str(vendor)+os.pathsep+os.environ.get('PYTHONPATH','')}
    for i,seed in enumerate(seeds):
        pool=output/('seed-%d'%seed)
        args=[sys.executable,str(ROOT/'sgha_attack/backends/generate_anchors_reference.py'),'--from-file',str(Path(a.targets).resolve()),'--outdir',str(pool),'--config',str(ROOT/'configs/sd_v1_inference.yaml'),'--ckpt',str(Path(ckpt).resolve()),'--seed',str(seed),'--n_samples',str(a.batch_size),'--plms','--skip_grid']
        pool.mkdir()
        with (pool/'run.log').open('w') as log:subprocess.run(args,cwd=vendor,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if len(list((pool/'samples').glob('*.png')))!=len(targets):raise RuntimeError('Incomplete anchor pool: %s'%pool)
        with record.open('a') as f:f.write('%02d: %d -> %s\n'%(i+1,seed,pool))
    (output/'generation.json').write_text(json.dumps(plan,indent=2)+'\n')
    print('Completed %d anchor pools: %s'%(len(seeds),record))

if __name__=='__main__':main()
