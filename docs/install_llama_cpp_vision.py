"""CLI wrapper: install vision llama-cpp-python into the Python that runs this file.

Use ComfyUI's interpreter so the GGUF nodes can import llama_cpp:

  path\\to\\ComfyUI\\python_embeded\\python.exe docs\\install_llama_cpp_vision.py
"""

import sys
from pathlib import Path

print("===================================================")
print("1-Click Llama-CPP-Python (Vision) Wheel Installer")
print("===================================================")
print(f"Python: {sys.executable}")

root = Path(__file__).resolve().parent.parent
py_dir = root / "py"
if str(py_dir) not in sys.path:
    sys.path.insert(0, str(py_dir))

from llama_cpp_install import ensure_llama_cpp_vision

ok = ensure_llama_cpp_vision(force=True)
sys.exit(0 if ok else 1)
