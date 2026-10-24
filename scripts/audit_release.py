"""Audit source or reachable Git history; never print matched secret values."""
import argparse
import ast
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[1]
SKIP={'.git','__pycache__','vendor','models','outputs','.venv','.pytest_cache'}
HOSTS={'github.com','huggingface.co','api.openai.com','storage.googleapis.com','opensource.org','cocodataset.org','www.image-net.org'}
CREDENTIAL=re.compile(r'(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AIza[A-Za-z0-9_-]{30,}|'+'-----'+'BEGIN '+r'.*PRIVATE KEY-----)')
PRIVATE=re.compile(r'(?:/(?:data|home|mnt|root|scratch|workspace)/[A-Za-z0-9_.-]+/|[A-Za-z]:[\\/](?:Users|Phd|models)[\\/])')
SECRET_FIELD=re.compile(r'^(?:(?:.*[._-])?(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password)|hf_token|github_token)$',re.I)

def inspect_text(text,name):
    found=[]
    for i,row in enumerate(text.splitlines(),1):
        if CREDENTIAL.search(row):found.append((name,i,'credential-like literal'))
        if PRIVATE.search(row):found.append((name,i,'machine-specific path'))
        for raw in re.findall(r'https?://[^\s<>"\x27`]+',row):
            url=urlparse(raw.rstrip(').,;'))
            if url.hostname not in HOSTS:found.append((name,i,'unlisted network host'))
            if url.username or url.password:found.append((name,i,'credentials in URL'))
        if not name.endswith('.py'):
            match=re.match(r'\s*[\x27\x22]?([A-Za-z_.-]+)[\x27\x22]?\s*[:=]\s*(.*)',row)
            if match and SECRET_FIELD.match(match[1]) and match[2].strip().strip('\x27\x22'):
                found.append((name,i,'nonempty credential setting'))
    if name.endswith('.py'):
        tree=ast.parse(text)
        for node in ast.walk(tree):
            key=None;value=None
            if isinstance(node,ast.keyword):key=node.arg;value=node.value
            elif isinstance(node,(ast.Assign,ast.AnnAssign)):
                target=node.targets[0] if isinstance(node,ast.Assign) else node.target
                if isinstance(target,ast.Name):key=target.id
                value=node.value
            if key and SECRET_FIELD.match(key) and isinstance(value,ast.Constant) and isinstance(value.value,str) and value.value:
                found.append((name,node.lineno,'hardcoded credential'))
    return found

def git(*args):
    return subprocess.check_output(['git','-c','safe.directory='+ROOT.as_posix(),'-C',str(ROOT)]+list(args))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history',action='store_true',help='also scan unique blobs reachable from every local Git ref')
    args=parser.parse_args();findings=[];count=0
    for directory,dirs,files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in SKIP and not d.startswith('.venv') and not d.endswith('.egg-info')]
        for filename in files:
            path=Path(directory)/filename
            if path.suffix in {'.pyc','.bundle'}:continue
            try:text=path.read_text(encoding='utf8')
            except UnicodeError:continue
            count+=1;findings+=inspect_text(text,path.relative_to(ROOT).as_posix())
    history_count=0
    if args.history:
        for row in git('rev-list','--objects','--all').decode().splitlines():
            parts=row.split(' ',1)
            if len(parts)!=2:continue
            oid,name=parts
            if git('cat-file','-t',oid).strip()!=b'blob':continue
            try:text=git('cat-file','blob',oid).decode('utf8')
            except UnicodeError:continue
            history_count+=1
            findings+=inspect_text(text,'history/'+oid[:12]+'/'+name)
    for name,line,kind in findings:print('%s:%d: %s'%(name,line,kind))
    print('Audited %d source files and %d historical blobs; %d findings.'%(count,history_count,len(findings)))
    return bool(findings)

if __name__=='__main__':sys.exit(main())
