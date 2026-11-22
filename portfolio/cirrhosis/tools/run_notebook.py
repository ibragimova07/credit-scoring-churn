"""Execute a notebook in place using nbclient (dev utility).

Usage: python tools/run_notebook.py 04_training.ipynb
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
name = sys.argv[1]
path = ROOT / "notebooks" / name
nb = nbformat.read(path, as_version=4)
client = NotebookClient(
    nb, timeout=1800, kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}}, allow_errors=False,
)
client.execute()
nbformat.write(nb, path)
print(f"executed {name}")
