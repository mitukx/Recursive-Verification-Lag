import unittest
from src.score_mbppplus_docker import docker_command,sanitize,syntax_valid


class DockerScoringContractTest(unittest.TestCase):
    def test_isolated_digest_only_and_no_host_mounts(self):
        cmd=docker_command('ganler/evalplus@sha256:'+'a'*64,'rvl_candidate_test','print(1)')
        for option in ['--network','none','--read-only','--cap-drop','ALL',
                       '--security-opt','no-new-privileges','--pull','never']:
            self.assertIn(option,cmd)
        self.assertNotIn('-v',cmd);self.assertNotIn('--volume',cmd)
        self.assertNotIn('--mount',cmd)
        with self.assertRaises(ValueError):docker_command('ganler/evalplus:latest','x','')

    def test_deterministic_fence_policy_and_syntax_only(self):
        self.assertEqual(sanitize('```python\ndef f(x): return x\n```'),
                         'def f(x): return x')
        self.assertTrue(syntax_valid('def f(x): return x'))
        self.assertFalse(syntax_valid('def f(:'))
        self.assertEqual(sanitize('Explanation:\ndef f(x): return x'),
                         'Explanation:\ndef f(x): return x')


if __name__=='__main__':unittest.main()
