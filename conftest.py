import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PACKAGE = os.path.join(ROOT, "ctg_viz")
if PACKAGE not in sys.path:
    sys.path.insert(0, PACKAGE)
