"""Exercise dispatch and saved-output validation without GPU/model downloads."""
import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch
from sgha_attack import attack,caption
from sgha_attack.common import ROOT


class LauncherTests(unittest.TestCase):
    def test_all_attack_profiles_forward_the_paper_parameters(self):
        expected={
            'vit_b':(250,224,[7,9,11],1.5,1.,.7,2.5),
            'vit_l':(10,336,[14,18,22],8.,.5,1.5,1.),
            'vit_g':(5,224,[23,30,37],6.,1.,.8,2.5),
        }
        for name,(batch,res,layers,feature,cls,spatial,mid) in expected.items():
            out=io.StringIO()
            with redirect_stdout(out):attack.main(['--surrogate',name,'--clean-root','data/clean','--output','outputs/dry','--dry-run'])
            args=json.loads(out.getvalue())['arguments']
            for key,value in {'batch_size':batch,'input_res':res,'steps':100,'alpha':1.,'epsilon':8,'seed':2023,'num_anchors':5,'anchor_temp':5.,'lambda_feature':feature,'lambda_cls':cls,'lambda_spatial':spatial,'lambda_text_mid':mid,'lambda_text':1.,'lambda_anchor':1.,'feature_noise_scale':0.}.items():
                self.assertEqual(float(args[args.index('--'+key)+1]),value,(name,key))
            start=args.index('--hook_layers')+1
            self.assertEqual([int(v) for v in args[start:start+len(layers)]],layers)

    def test_no_anchor_ablation_does_not_require_anchor_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);clean=root/'clean/images';clean.mkdir(parents=True)
            (clean/'one.png').write_bytes(b'offline-image')
            targets=root/'targets.txt';targets.write_text('target\n')
            output=root/'attack'
            def run(*args,**kwargs):
                self.assertIn('--no_anchor',sys.argv)
                (output/'images').mkdir(parents=True)
                (output/'images/one.png').write_bytes(b'offline-adversarial-image')
            argv=sys.argv[:]
            with patch.object(attack.runpy,'run_path',side_effect=run):
                attack.main(['--surrogate','vit_b','--clean-root',str(clean.parent),'--targets',str(targets),'--output',str(output),'--num-samples','1','--batch-size','1','--no_anchor'])
            self.assertEqual(sys.argv,argv)
            self.assertTrue((output/'run.json').exists())

    def test_six_caption_launchers_match_native_flags_and_save_manifests(self):
        fake=ModuleType('huggingface_hub')
        downloads=[]
        def hf(repo,filename):downloads.append((repo,filename));return '/cache/'+filename
        fake.hf_hub_download=hf
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);images=root/'input/images';images.mkdir(parents=True)
            for i in range(100):(images/('%05d.png'%i)).write_bytes(b'offline-image')
            for name,(vendor_name,batch) in caption.BACKENDS.items():
                (root/'vendor'/vendor_name).mkdir(parents=True,exist_ok=True)
                output=root/'captions'/(name+'.txt')
                argv=['caption','--model',name,'--images',str(images.parent),'--output',str(output),'--vendor-root',str(root/'vendor'),'--num-samples','100','--seed','7']
                before_cwd=Path.cwd();before_path=sys.path[:]
                def run(script,run_name):
                    self.assertEqual(Path.cwd(),root/'vendor'/vendor_name)
                    args=sys.argv[1:]
                    self.assertEqual(int(args[args.index('--batch_size')+1]),batch)
                    self.assertEqual(os.environ['SGHA_SEED'],'7')
                    tree=ast.parse(Path(script).read_text())
                    native_flags={n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant)}
                    if name=='unidiffuser':
                        native_flags={'--'+n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr.startswith('DEFINE_') and n.args and isinstance(n.args[0],ast.Constant)}
                        self.assertEqual(args[args.index('--mode')+1],'i2t')
                        self.assertEqual(args[args.index('--output')+1],str(output.with_suffix('')))
                    for flag in (v for v in args if v.startswith('--')):self.assertIn(flag,native_flags,(name,flag))
                    if name=='minigpt4':
                        self.assertIn('model.llama_model=Vision-CAIR/vicuna',args)
                        self.assertIn('model.ckpt='+str(Path('/cache/pretrained_minigpt4.pth').resolve()),args)
                    output.write_text(('caption\n')*100)
                with patch.dict(sys.modules,{'huggingface_hub':fake}),patch.object(sys,'argv',argv),patch.object(caption.runpy,'run_path',side_effect=run),redirect_stdout(io.StringIO()):caption.main()
                self.assertEqual(Path.cwd(),before_cwd)
                self.assertEqual(sys.path,before_path)
                manifest=json.loads(output.with_suffix('.run.json').read_text())
                self.assertEqual(manifest['n'],100)
                self.assertEqual(manifest['seed'],7)
                self.assertEqual(len(manifest['images']),100)
            self.assertIn(('thu-ml/unidiffuser-v1','uvit_v1.pth'),downloads)
            self.assertIn(('Vision-CAIR/MiniGPT-4','pretrained_minigpt4.pth'),downloads)

    def test_invalid_batches_fail_before_dispatch(self):
        with self.assertRaises(ValueError):attack.main(['--surrogate','vit_b','--clean-root','data/clean','--output','outputs/dry','--batch-size','0','--dry-run'])
        with patch.object(sys,'argv',['caption','--model','llava','--images','input','--output','output','--batch-size','0','--dry-run']):
            with self.assertRaises(ValueError):caption.main()

    def test_caption_default_seeds_match_the_original_first_trials(self):
        for name in caption.BACKENDS:
            output=io.StringIO()
            with patch.object(sys,'argv',['caption','--model',name,'--images','input','--output','output.txt','--dry-run']),redirect_stdout(output):caption.main()
            self.assertEqual(json.loads(output.getvalue())['seed'],1234 if name=='unidiffuser' else 2023)


if __name__=='__main__':unittest.main()
