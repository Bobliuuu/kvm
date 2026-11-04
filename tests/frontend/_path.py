"""Makes `import kvmfe` work under `python3 -I -m unittest discover -s tests/frontend` (isolated mode ignores cwd)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'frontend'))
