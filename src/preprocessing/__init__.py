import sys
from pathlib import Path

# Ensure the parent src/ directory is on sys.path so that sibling modules
# (dataset, utilities, etc.) are importable from within this sub-package.
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
