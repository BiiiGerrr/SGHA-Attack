"""Meaningful redaction checks using constructed, non-secret fixtures."""
import unittest
from scripts.audit_release import inspect_text

class ReleaseAuditTests(unittest.TestCase):
    def test_assignment_style_credentials_are_flagged_without_echoing_value(self):
        value='constructed-fixture-value'
        findings=inspect_text('OPENAI_API_KEY='+repr(value),'config.py')
        self.assertTrue(any(kind=='hardcoded credential' for _,_,kind in findings))
        self.assertNotIn(value,str(findings))

    def test_environment_credentials_and_empty_template_are_allowed(self):
        self.assertEqual(inspect_text('import os\napi_key=os.environ["OPENAI_API_KEY"]','config.py'),[])
        self.assertEqual(inspect_text('OPENAI_API_KEY=\nHF_TOKEN=\n','.env.example'),[])

    def test_machine_paths_and_unlisted_hosts_are_detected(self):
        path='/'+'/'.join(['home','example-user','weights','model'])
        url='https'+':/'+'/relay.invalid/v1'
        self.assertTrue(any(kind=='machine-specific path' for _,_,kind in inspect_text(path,'settings.txt')))
        self.assertTrue(any(kind=='unlisted network host' for _,_,kind in inspect_text(url,'settings.txt')))

    def test_official_endpoint_is_allowed(self):
        self.assertEqual(inspect_text('https://api.openai.com/v1','settings.txt'),[])

if __name__=='__main__':unittest.main()
