import json
from pathlib import Path
import tempfile
import unittest
from sgha_attack.common import atomic_json,digest,read_lines,same_length
from sgha_attack.metrics import JUDGE_PROMPT,asr,parse_score

class DataMetricTests(unittest.TestCase):
    def test_missing_or_malformed_judgment_is_not_a_failure_score(self):
        for invalid in ['', 'Transport error', 'Score: 0.7', 'Score: -1', 'Score: 1 trailing text']:
            with self.assertRaises(ValueError):parse_score(invalid)

    def test_last_explicit_score_wins(self):
        self.assertEqual(parse_score('A tentative Score: 0.5\nFinal analysis.\nScore: 1'),1.)

    def test_metric_thresholds_are_distinct(self):
        result=asr([0.,.5,1.,1.])
        self.assertEqual(result['asr_fool'],75.)
        self.assertEqual(result['asr_target'],50.)
        with self.assertRaises(ValueError):asr([])

    def test_alignment_rejects_missing_and_blank_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'rows.txt';p.write_text('first\n\nthird\n')
            with self.assertRaises(ValueError):read_lines(p)
            p.write_text('first\nsecond\n')
            with self.assertRaises(ValueError):read_lines(p,3)
            with self.assertRaises(ValueError):same_length([1],[1,2])

    def test_changed_inputs_cannot_share_a_cache_fingerprint(self):
        a=dict(model='judge',caption='cat',target='dog')
        b={**a,'caption':'dog'}
        self.assertNotEqual(digest(a),digest(b))

    def test_prompt_formats_literal_braces_and_original_criteria(self):
        prompt=JUDGE_PROMPT.format(original_text='original',target_text='target',generated_text='generated')
        self.assertIn('{ original }',prompt)
        self.assertIn('assign a score of 0.5',prompt)
        self.assertIn('main objects are consistent',prompt)

    def test_atomic_records(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'records/0.json';atomic_json(p,{'score':1.})
            self.assertEqual(json.loads(p.read_text()),{'score':1.})
            self.assertFalse(p.with_suffix('.json.tmp').exists())

if __name__=='__main__':unittest.main()
