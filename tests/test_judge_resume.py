import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch
from sgha_attack.eval_gpt import main

class JudgeResumeTests(unittest.TestCase):
    def test_api_failure_is_not_scored_and_resume_does_not_repeat_successes(self):
        requests=[]
        failing={1}
        def create(**kwargs):
            # Identify a pair from its deliberately unique generated text.
            prompt=kwargs['messages'][0]['content']
            index=next(i for i in range(3) if 'generated-%d'%i in prompt)
            requests.append(index)
            if index in failing:raise ConnectionError('offline simulated transport failure')
            return SimpleNamespace(model='offline-judge',choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content='Analysis.\nScore: 1'))])
        fake=ModuleType('openai')
        fake.OpenAI=lambda **kwargs:SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name in ['original','targets','captions']:
                prefix='generated' if name=='captions' else name
                (root/(name+'.txt')).write_text('\n'.join('%s-%d'%(prefix,i) for i in range(3))+'\n')
            output=root/'summary.json'
            argv=['judge','--original',str(root/'original.txt'),'--targets',str(root/'targets.txt'),'--captions',str(root/'captions.txt'),'--output',str(output),'--workers','1','--retries','1']
            fake_key='offline'
            with patch.dict(sys.modules,{'openai':fake}),patch.dict(os.environ,OPENAI_API_KEY=fake_key),patch.object(sys,'argv',argv):
                with self.assertRaises(RuntimeError):main()
                self.assertFalse(output.exists())
                self.assertEqual(len(list((root/'summary/records').glob('*.json'))),2)
                requests.clear();failing.clear();main()
                self.assertEqual(requests,[1])
                summary=json.loads(output.read_text())
                self.assertEqual(summary['asr_target'],100.)
                self.assertEqual(summary['n'],3)
                self.assertEqual(json.loads((root/'summary/errors/00001.json').read_text())['status'],'resolved')

if __name__=='__main__':unittest.main()
