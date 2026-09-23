import unittest
from unittest import mock
from src.score_mbppplus_docker import docker_command,sanitize,syntax_valid,run_capped


class DockerScoringContractTest(unittest.TestCase):
    def test_isolated_digest_only_and_no_host_mounts(self):
        cmd=docker_command('ganler/evalplus@sha256:'+'a'*64,'rvl_candidate_test','print(1)')
        for option in ['--network','none','--read-only','--cap-drop','ALL',
                       '--security-opt','no-new-privileges','--pull','never']:
            self.assertIn(option,cmd)
        self.assertNotIn('-v',cmd);self.assertNotIn('--volume',cmd)
        self.assertNotIn('--mount',cmd)
        self.assertIn('linux/arm64',cmd)
        self.assertEqual(docker_command('sha256:'+'b'*64,'test','pass')[0],'docker')
        with self.assertRaises(ValueError):docker_command('ganler/evalplus:latest','x','')

    def test_deterministic_fence_policy_and_syntax_only(self):
        self.assertEqual(sanitize('```python\ndef f(x): return x\n```'),
                         'def f(x): return x')
        self.assertTrue(syntax_valid('def f(x): return x'))
        self.assertFalse(syntax_valid('def f(:'))
        self.assertEqual(sanitize('Explanation:\ndef f(x): return x'),
                         'Explanation:\ndef f(x): return x')

    def test_absent_docker_fails_before_candidate_execution(self):
        with mock.patch('src.score_mbppplus_docker.subprocess.Popen',
                        side_effect=FileNotFoundError):
            with self.assertRaisesRegex(RuntimeError,'Docker is unavailable'):
                run_capped(['docker','run'],{'source':'def f(): pass'},'test')


if __name__=='__main__':unittest.main()
