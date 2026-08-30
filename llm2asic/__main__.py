# llm2asic/__main__.py
"""支持 ``python -m llm2asic ...`` 运行 CLI。"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
