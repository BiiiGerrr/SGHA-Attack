"""Evaluate saved captions with resumable, strictly validated GPT judgments."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import time
from .common import atomic_json,digest,read_lines,same_length
from .metrics import JUDGE_PROMPT,asr,parse_score

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',required=True,help='the same saved MiniGPT-4 clean descriptions for every victim')
    p.add_argument('--targets',required=True)
    p.add_argument('--captions',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--model',default='gpt-4.1-mini')
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--retries',type=int,default=6)
    a=p.parse_args()
    original,targets,captions=[read_lines(path) for path in (a.original,a.targets,a.captions)]
    same_length(original,targets,captions)
    output=Path(a.output).resolve();records=output.with_suffix('')/'records'
    spec=dict(model=a.model,temperature=0,max_tokens=1024,prompt=JUDGE_PROMPT,original=original,targets=targets,captions=captions)
    fingerprint=digest(spec)
    meta=records.parent/'input.json'
    if meta.exists() and json.loads(meta.read_text())['input_hash']!=fingerprint:raise ValueError('Existing output belongs to different inputs/settings; use a new output path.')
    atomic_json(meta,dict(input_hash=fingerprint,model=a.model,n=len(targets),temperature=0,api_seed_parameter_sent=False))
    if not os.environ.get('OPENAI_API_KEY'):raise ValueError('Set OPENAI_API_KEY in your own shell.')
    from openai import OpenAI
    client=OpenAI(api_key=os.environ['OPENAI_API_KEY'],base_url='https://api.openai.com/v1')

    def work(i):
        path=records/('%05d.json'%i)
        if path.exists():
            old=json.loads(path.read_text())
            if old.get('input_hash')!=fingerprint or old.get('index')!=i or old.get('score') not in (0.,.5,1.):raise ValueError('Invalid cached judgment at index %d'%i)
            return old['score']
        prompt=JUDGE_PROMPT.format(original_text=original[i],target_text=targets[i],generated_text=captions[i])
        error=None
        for retry in range(a.retries):
            try:
                reply=client.chat.completions.create(model=a.model,messages=[dict(role='user',content=prompt)],temperature=0,max_tokens=1024)
                choice=reply.choices[0]
                if choice.finish_reason=='length' or not choice.message.content:raise ValueError('Incomplete judge response')
                raw=choice.message.content.strip();score=parse_score(raw)
                atomic_json(path,dict(index=i,input_hash=fingerprint,score=score,analysis=raw,requested_model=a.model,returned_model=reply.model))
                old_error=records.parent/'errors'/('%05d.json'%i)
                if old_error.exists():atomic_json(old_error,dict(index=i,status='resolved'))
                return score
            except Exception as exc:
                error=type(exc).__name__
                if retry+1<a.retries:time.sleep(min(2**retry,30))
        atomic_json(records.parent/'errors'/('%05d.json'%i),dict(index=i,error_type=error,status='unresolved'))
        raise RuntimeError('Judge failed at index %d (%s); resume the same command to retry missing records.'%(i,error))

    failed=[];scores=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(work,i):i for i in range(len(targets))}
        for future in as_completed(futures):
            try:scores.append(future.result())
            except Exception:failed.append(futures[future])
    if failed:
        if output.exists():output.unlink()
        raise RuntimeError('%d judgments missing; no ASR summary written. Resume to fill missing indices.'%len(failed))
    summary={**asr(scores),'judge_model':a.model,'temperature':0,'input_hash':fingerprint}
    atomic_json(output,summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
