"""Shared I/O and data-alignment checks. Importing this module needs no GPU."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS={'.png','.jpg','.jpeg','.bmp','.webp','.tif','.tiff'}

def read_lines(path, limit=None):
    lines=Path(path).read_text(encoding='utf-8').splitlines()
    if limit is not None:
        if len(lines)<limit:
            raise ValueError('%s has %d rows; expected at least %d'%(path,len(lines),limit))
        lines=lines[:limit]
    if not lines or any(not row.strip() for row in lines):
        raise ValueError('Empty input or blank row: %s'%path)
    return [row.strip() for row in lines]

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    os.replace(temporary,path)

def image_paths(root,limit=None):
    root=Path(root)
    files=sorted((p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),key=lambda p:p.relative_to(root).as_posix())
    if limit is not None:
        if len(files)<limit:raise ValueError('Not enough images in %s'%root)
        files=files[:limit]
    if not files:raise ValueError('No images in %s'%root)
    return files

def anchor_roots(record_file):
    roots=[]
    for row in read_lines(record_file):
        if '->' not in row:raise ValueError('Expected seed record: NN: SEED -> DIRECTORY')
        path=Path(row.rsplit('->',1)[1].strip()).expanduser().resolve()
        if not path.is_dir():raise FileNotFoundError(path)
        roots.append(path)
    return roots

def same_length(*sequences):
    sizes=[len(s) for s in sequences]
    if not sizes or len(set(sizes))!=1:raise ValueError('Unaligned row counts: %s'%sizes)

def ensure_new_output(path):
    path=Path(path).expanduser().resolve()
    if path.exists():raise FileExistsError('Use a new output path; existing results are never appended or overwritten: %s'%path)
    path.parent.mkdir(parents=True,exist_ok=True)
    return path
