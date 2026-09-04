import os
import sys

_possible_dassl_dirs = [
    "/kaggle/working/Dassl.pytorch",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Dassl.pytorch")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Dassl.pytorch")),
]
for _p in _possible_dassl_dirs:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
