"""Apply documented native-backend and dependency compatibility patches."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',required=True)
    p.add_argument('--directory',required=True)
    a=p.parse_args()
    if a.backend=='stable-diffusion':
        path=Path(a.directory)/'ldm/models/diffusion/ddpm.py'
        text=path.read_text(encoding='utf8')
        old='from pytorch_lightning.utilities.distributed import rank_zero_only'
        new='from pytorch_lightning.utilities.rank_zero import rank_zero_only'
        if old in text:
            path.write_text(text.replace(old,new),encoding='utf8')
            print('Updated the Lightning rank-zero import; sampling computations are unchanged.')
        elif new not in text:raise RuntimeError('Unexpected Stable Diffusion revision; inspect the Lightning import.')
        return
    if a.backend=='LAVIS':
        path=Path(a.directory)/'lavis/models/ulip_models/utils/io.py'
        text=path.read_text(encoding='utf8')
        if '\nimport open3d\n' in text:
            path.write_text(text.replace('\nimport open3d\n','\n# import open3d  # Disabled for caption-only integration, as in the experiment.\n'),encoding='utf8')
            print('Disabled the unused 3D point-cloud import for the caption integration.')
        return
    if a.backend!='MiniGPT-4':return
    path=Path(a.directory)/'minigpt4/models/minigpt_base.py'
    text=path.read_text(encoding='utf8')
    old='# stopping_criteria=stopping_criteria,'
    new='stopping_criteria=stopping_criteria,'
    if old in text:
        if text.count(old)!=1:raise RuntimeError('Unexpected MiniGPT-4 revision; inspect stopping criteria before patching.')
        path.write_text(text.replace(old,new),encoding='utf8')
        print('Enabled the same MiniGPT-4 stopping criteria as the experiment snapshot.')
    elif new not in text:raise RuntimeError('MiniGPT-4 source does not match the documented backend; no patch applied.')

if __name__=='__main__':main()
