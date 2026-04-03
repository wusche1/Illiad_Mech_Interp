"""Test that exercise notebooks install and import correctly in a Colab-like environment.

Fetches the exact versions of conflict-prone packages from Colab's published
pip freeze (https://github.com/googlecolab/backend-info), pre-installs them,
then runs the notebook's pip install line and tests imports.

Run with: uv run pytest tests/test_colab_compat.py -v
Skip with: SKIP_SLOW=1 uv run pytest tests/ -v
"""
import os
import re
import shlex
import subprocess
import urllib.request
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parent.parent
EXERCISES = ROOT / "exercises"

COLAB_FREEZE_URL = (
    "https://raw.githubusercontent.com/googlecolab/backend-info/main/pip-freeze.gpu.txt"
)

# Packages that are pre-installed in Colab and likely to conflict with our installs.
COLAB_RELEVANT_PACKAGES = {
    "numpy", "numba", "transformers", "torch", "einops",
    "plotly", "matplotlib", "ipython",
}


def _get_colab_versions():
    """Fetch Colab's freeze and extract versions for relevant packages."""
    response = urllib.request.urlopen(COLAB_FREEZE_URL, timeout=10)
    lines = response.read().decode().splitlines()
    versions = {}
    for line in lines:
        m = re.match(r"^([a-zA-Z0-9_-]+)==(.+)$", line.strip())
        if m and m.group(1).lower().replace("-", "_") in {p.replace("-", "_") for p in COLAB_RELEVANT_PACKAGES}:
            # Strip CUDA build tags (e.g. +cu128) — not available on PyPI for macOS
            version = re.sub(r"\+.*$", "", m.group(2))
            versions[m.group(1)] = version
    return versions


def _extract_pip_install_and_imports(nb_path):
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


def _find_python_312():
    for candidate in ["python3.12", "python3"]:
        try:
            result = subprocess.run(
                [candidate, "--version"], capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and "3.12" in result.stdout:
                return candidate
        except FileNotFoundError:
            continue
    try:
        result = subprocess.run(
            ["uv", "python", "find", "3.12"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        pass
    pytest.skip("Python 3.12 not available (needed to simulate Colab)")


@pytest.fixture(scope="session")
def colab_versions():
    """Fetch Colab's package versions once per test session."""
    return _get_colab_versions()


@pytest.mark.parametrize(
    "nb_path,pip_line,import_src",
    [(p, pip, imp) for p, pip, imp, _ in _discovered],
    ids=[label for _, _, _, label in _discovered],
)
def test_colab_install_and_import(nb_path, pip_line, import_src, colab_versions, tmp_path):
    if os.environ.get("SKIP_SLOW"):
        pytest.skip("SKIP_SLOW is set")

    python = _find_python_312()
    venv = tmp_path / "venv"
    subprocess.run([python, "-m", "venv", str(venv)], check=True, timeout=30)
    pip_bin = str(venv / "bin" / "pip")
    python_bin = str(venv / "bin" / "python")

    # Pre-install Colab's pinned versions of conflict-prone packages
    colab_pins = [f"{pkg}=={ver}" for pkg, ver in colab_versions.items()]
    result = subprocess.run(
        [pip_bin, "install", "-q"] + colab_pins,
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, f"Colab pre-install failed:\n{result.stderr}"

    # Run the notebook's pip install line
    pip_args = shlex.split(pip_line.replace("-q", "").strip())
    result = subprocess.run(
        [pip_bin, "install"] + pip_args,
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, f"pip install failed:\n{result.stderr}"

    # Test imports
    result = subprocess.run(
        [python_bin, "-c", import_src],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"Import failed:\n{result.stderr}"
