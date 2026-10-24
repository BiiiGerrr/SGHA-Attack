"""Run a paper single-surrogate profile without rewriting its attack kernel."""
import argparse
import json
import os
import runpy
import sys
from pathlib import Path
from .common import ROOT,anchor_roots,atomic_json,image_paths,read_lines

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--surrogate',choices=['vit_b','vit_l','vit_g'],required=True)
    p.add_argument('--clean-root','--cle_data_path',dest='clean_root',required=True)
    p.add_argument('--targets','--tgt_text_path',dest='targets',default=str(ROOT/'data/metadata/targets_1000.txt'))
    p.add_argument('--anchor-record','--anchor_list_file',dest='anchor_record',default=str(ROOT/'outputs/anchors/seed_record.txt'))
    p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=2023)
    p.add_argument('--num-samples','--num_samples',dest='num_samples',type=int,default=1000)
    p.add_argument('--batch-size','--batch_size',dest='batch_size',type=int)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--epsilon',type=int,default=8,help='pixel units: 8 means 8/255')
    p.add_argument('--alpha',type=float,default=1.0,help='pixel units')
    p.add_argument('--model-id',help='HF CLIP ID for ViT-L, or an EVA checkpoint file override')
    p.add_argument('--dry-run',action='store_true')
    a,extra=p.parse_known_args(argv)
    profile=json.loads((ROOT/'configs/attack_profiles.json').read_text())[a.surrogate]
    bs=profile['batch_size'] if a.batch_size is None else a.batch_size
    if a.num_samples<=0 or bs<=0 or a.num_samples%bs:raise ValueError('num-samples must be divisible by batch-size')
    script=ROOT/'sgha_attack/attacks'/(''+a.surrogate+'.py')
    args=['--cle_data_path',str(Path(a.clean_root).resolve()),'--tgt_text_path',str(Path(a.targets).resolve()),'--anchor_list_file',str(Path(a.anchor_record).resolve()),'--output',str(Path(a.output).resolve()),'--num_samples',str(a.num_samples),'--batch_size',str(bs),'--steps',str(a.steps),'--epsilon',str(a.epsilon),'--alpha',str(a.alpha),'--seed',str(a.seed)]
    for key,value in profile.items():
        if key=='batch_size':continue
        args+=['--'+key]+([str(v) for v in value] if isinstance(value,list) else [str(value)])
    args+=['--num_anchors','5','--anchor_temp','5.0','--lambda_text','1.0','--lambda_anchor','1.0','--feature_noise_scale','0.0']
    if a.model_id:
        if a.surrogate=='vit_b':args+=['--clip_encoder',a.model_id]
        elif a.surrogate=='vit_l':args+=['--clip_model_path',a.model_id]
        else:args+=['--local_model_path',a.model_id]
    args+=extra
    if a.dry_run:
        print(json.dumps(dict(script=str(script.relative_to(ROOT)),arguments=args,profile=profile),indent=2));return
    out=Path(a.output)
    if out.exists():raise FileExistsError('Choose a new output directory to avoid mixing attack runs.')
    clean=image_paths(a.clean_root,a.num_samples);read_lines(a.targets,a.num_samples)
    if any(f.parent==Path(a.clean_root) for f in clean):raise ValueError('Use clean/class_name/image files for ImageFolder')
    if '--no_anchor' not in extra:
        for root in anchor_roots(a.anchor_record):image_paths(root,a.num_samples)
    os.environ['SGHA_SEED']=str(a.seed)
    previous_argv=sys.argv[:]
    sys.argv=[str(script)]+args
    try:runpy.run_path(str(script),run_name='__main__')
    finally:sys.argv=previous_argv
    generated=image_paths(out)
    expected=[p.relative_to(a.clean_root).with_suffix('.png').as_posix() for p in clean]
    if [p.relative_to(out).as_posix() for p in generated]!=expected:raise RuntimeError('Attack output count/order differs from the requested input images.')
    atomic_json(out/'run.json',dict(surrogate=a.surrogate,seed=a.seed,steps=a.steps,epsilon_pixels=a.epsilon,alpha_pixels=a.alpha,profile=profile,argv=args))

if __name__=='__main__':main()
