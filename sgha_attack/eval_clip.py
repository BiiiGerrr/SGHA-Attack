"""Five-encoder normalized text/text CLIP similarity, as in the paper."""
import argparse
import json
from .common import atomic_json,read_lines,same_length,sha256

ENCODERS=['RN50','RN101','ViT-B/16','ViT-B/32','ViT-L/14']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--captions',required=True)
    p.add_argument('--targets',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--batch-size',type=int,default=50)
    p.add_argument('--device',default=None)
    a=p.parse_args()
    predicted=read_lines(a.captions);targets=read_lines(a.targets);same_length(predicted,targets)
    if a.batch_size<=0:raise ValueError('batch-size must be positive')
    import clip
    import torch
    device=a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    results={};pair_scores={}
    for encoder in ENCODERS:
        model,_=clip.load(encoder,device=device,jit=False);model.eval()
        def features(rows,truncate):
            parts=[]
            with torch.no_grad():
                tokens=clip.tokenize(rows,truncate=truncate).to(device)
                for start in range(0,len(rows),a.batch_size):parts.append(model.encode_text(tokens[start:start+a.batch_size]))
                values=torch.cat(parts,dim=0)
                return values/values.norm(dim=1,keepdim=True)
        with torch.no_grad():
            predicted_features=features(predicted,True)
            target_features=features(targets,False)
            similarities=(predicted_features*target_features).sum(dim=1)
            results[encoder]=similarities.mean().item()
            pair_scores[encoder]=similarities.cpu().tolist()
        del model,predicted_features,target_features,similarities
    report=dict(n=len(predicted),clip_ensemble=sum(results.values())/len(results),encoders=results,per_pair=pair_scores,captions_sha256=sha256(a.captions),targets_sha256=sha256(a.targets),metric='mean paired normalized text-text cosine')
    atomic_json(a.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='per_pair'},indent=2))

if __name__=='__main__':main()
