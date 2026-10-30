"""Validate the generated notebook JSON and rebuild a flat execution script."""
import ast
import json
from pathlib import Path

NOTEBOOK = Path("Credit_Default_Classification.ipynb")

nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
assert nb["nbformat"] == 4 and nb["nbformat_minor"] == 5

mds = codes = 0
for i, cell in enumerate(nb["cells"]):
    assert cell["source"], f"empty cell {i}"
    assert "id" in cell, f"cell {i} missing id"
    if cell["cell_type"] == "markdown":
        mds += 1
    elif cell["cell_type"] == "code":
        codes += 1
        ast.parse("".join(cell["source"]))
    else:
        raise ValueError(f"bad cell type {cell['cell_type']}")

script = "\n\n# ===== CELL BREAK =====\n\n".join(
    "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
)
Path("_verify_nb.py").write_text(script, encoding="utf-8")
print(f"JSON OK | {mds} markdown + {codes} code | all parse | verify script rebuilt")
