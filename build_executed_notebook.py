"""
Legacy entry point kept for convenience.
The real builder is build_notebook.py; this wrapper simply runs it.
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("build_notebook.py")), run_name="__main__")
