import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CommandLineTests(unittest.TestCase):
    def test_all_commands_expose_help(self):
        for name in ("train.py", "retrieve.py", "run_sensitivity.py"):
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / name), "--help"],
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--config", result.stdout)


if __name__ == "__main__":
    unittest.main()
