"""Run current manuscript Figure 6."""
from pathlib import Path
import runpy
import sys

directory = Path(__file__).resolve().parent / "current"
sys.path.insert(0, str(directory))
runpy.run_path(str(directory / "figure_06.py"), run_name="__main__")
