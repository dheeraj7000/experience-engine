"""Put the repo root on sys.path so flat top-level packages import in tests
without an editable install."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
