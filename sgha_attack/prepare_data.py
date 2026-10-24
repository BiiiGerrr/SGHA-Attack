"""Freeze the positional pairing used by the reference attack programs."""
import argparse
from pathlib import Path
from .common import ROOT,anchor_roots,atomic_json,image_paths,read_lines,sha256

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clean-root',required=True)
    p.add_argument('--targets',default=str(ROOT/'data/metadata/targets_1000.txt'))
    p.add_argument('--anchor-record',required=True)
    p.add_argument('--num-samples',type=int,default=1000)
    p.add_argument('--expected-list',default=str(ROOT/'data/metadata/clean_images_1000.txt'))
    p.add_argument('--output',default='outputs/data_manifest.json')
    a=p.parse_args()
    root=Path(a.clean_root).resolve();images=image_paths(root,a.num_samples)
    if any(i.parent==root for i in images):raise ValueError('Single-CLIP attacks require ImageFolder layout: clean/class_name/image.png')
    targets=read_lines(a.targets,a.num_samples)
    expected=read_lines(a.expected_list,a.num_samples)
    relative=[f.relative_to(root).as_posix() for f in images]
    if relative!=expected:raise ValueError('Clean image order differs from the paper list. Provide --expected-list for a new dataset and keep its targets aligned.')
    roots=anchor_roots(a.anchor_record)
    pools=[image_paths(r,a.num_samples) for r in roots]
    samples=[dict(index=i,clean=relative[i],clean_sha256=sha256(images[i]),target=targets[i],anchors=[dict(file=str(pool[i].resolve()),sha256=sha256(pool[i])) for pool in pools]) for i in range(a.num_samples)]
    atomic_json(a.output,dict(n=a.num_samples,clean_root=str(root),targets_sha256=sha256(a.targets),anchor_record_sha256=sha256(a.anchor_record),samples=samples))
    print('Verified %d aligned pairs and %d anchor pools.'%(a.num_samples,len(roots)))

if __name__=='__main__':main()
