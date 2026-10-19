import sys
from pathlib import Path

# Let tests import `src` without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
