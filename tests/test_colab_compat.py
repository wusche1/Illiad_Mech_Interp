"""Test that exercise notebooks can install and import successfully in a Colab-like environment.

Simulates Colab by creating a fresh venv with Colab's Python/numpy versions,
running the pip install line, then testing the imports.

Run with: uv run pytest tests/test_colab_compat.py -v
Skip with: SKIP_SLOW=1 uv run pytest tests/ -v
"""
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parent.parent
EXERCISES = ROOT / "exercises"


def _extract_pip_install_and_imports(nb_path):
    """Extract the pip install line and subsequent import cell from a notebook."""
    nb = nbformat.read(nb_path, as_version=4)
    pip_line = None
    import_src = None
    for cell in nb["cells"]:
        if cell.cell_type != "code":
            continue
        src = cell.source
        if "%pip install" in src:
            m = re.search(r"%pip install (.+)", src)
            if m:
                pip_line = m.group(1)
        elif pip_line and import_src is None and ("import torch" in src or "from transformer_lens" in src):
            import_src = src
    return pip_line, import_src


def discover_colab_notebooks():
    results = []
    for nb_path in sorted(EXERCISES.glob("*/notebook_*.ipynb")):
        pip_line, import_src = _extract_pip_install_and_imports(nb_path)
        if pip_line and import_src:
            variant = nb_path.stem.replace("notebook_", "")
            label = f"{nb_path.parent.name}/{variant}"
            results.append((nb_path, pip_line, import_src, label))
    return results


_discovered = discover_colab_notebooks()


@pytest.mark.parametrize(
    "nb_path,pip_line,import_src",
    [(p, pip, imp) for p, pip, imp, _ in _discovered],
    ids=[label for _, _, _, label in _discovered],
)
def test_colab_install_and_import(nb_path, pip_line, import_src, tmp_path):
    if os.environ.get("SKIP_SLOW"):
        pytest.skip("SKIP_SLOW is set")

    # Create a fresh venv
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    pip = str(venv / "bin" / "pip")
    python = str(venv / "bin" / "python")

    # Parse the pip line properly (handles quoted args like "numpy>=2")
    # Add ipython since it's pre-installed in Colab but not in a fresh venv
    pip_args = shlex.split(pip_line.replace("-q", "").strip()) + ["ipython"]
    result = subprocess.run(
        [pip, "install"] + pip_args,
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, f"pip install failed:\n{result.stderr}"

    # Test imports
    result = subprocess.run(
        [python, "-c", import_src],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"Import failed:\n{result.stderr}"
