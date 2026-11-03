import pathlib, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
class RunnerTests(unittest.TestCase):
    def test_shell_syntax(self):
        import subprocess
        subprocess.run(['bash','-n',str(ROOT/'run_gpu_matrix.sh')],check=True)
    def test_runner_has_timeouts(self):
        s=(ROOT/'run_gpu_matrix.sh').read_text()
        self.assertIn('timeout 120s',s)
        self.assertIn('timeout 900s',s)
        self.assertIn('compute-sanitizer',s)
    def test_runner_does_not_modify_upstream(self):
        s=(ROOT/'integration/apply.py').read_text()
        self.assertIn("path.with_name('kvm_ready.cu')",s)
        self.assertNotIn('path.write_text(result)',s)
