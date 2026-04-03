"""Execute every exercise notebook with solutions injected, verify all tests pass.

For each notebook, finds cells tagged with exercise_id and solution_id metadata.
Extracts the code from the solution markdown cell's ```python block, swaps it into
the exercise code cell, then executes the entire notebook.

Notebooks importing transformer_lens are marked slow (they need model downloads + GPU).
Skip them with: pytest tests/ -v -m "not slow"
"""
import os
import re
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parent.parent
LECTURES = ROOT / "lectures"

SLOW_TIMEOUT = 600
FAST_TIMEOUT = 120

SLOW_IMPORTS = {"transformer_lens", "circuitsvis"}


def discover_exercises():
    """Find exercise dirs that have a notebook.ipynb with exercise_id cells."""
    results = []
    for nb_path in sorted(LECTURES.glob("*/exercises/*/notebook.ipynb")):
        d = nb_path.parent
        label = f"{d.parent.parent.name}/{d.name}"
        results.append((d, label))
    return results


def extract_solutions(nb):
    """Extract solution code from markdown cells tagged with solution_id."""
    solutions = {}
    for cell in nb.cells:
        sid = cell.metadata.get("solution_id")
        if not sid:
            continue
        m = re.search(r"```python\n(.+?)```", cell.source, re.DOTALL)
        if m:
            solutions[sid] = m.group(1).strip()
    return solutions


def _is_slow(nb):
    """Check if any code cell imports a slow dependency."""
    for cell in nb.cells:
        if cell.cell_type == "code":
            for keyword in SLOW_IMPORTS:
                if keyword in cell.source:
                    return True
    return False


@pytest.mark.parametrize(
    "exercise_dir",
    [d for d, _ in discover_exercises()],
    ids=[n for _, n in discover_exercises()],
)
def test_notebook(exercise_dir):
    nb = nbformat.read(exercise_dir / "notebook.ipynb", as_version=4)
    solutions = extract_solutions(nb)

    exercise_ids = [
        cell.metadata["exercise_id"]
        for cell in nb.cells
        if cell.metadata.get("exercise_id")
    ]
    for eid in exercise_ids:
        assert eid in solutions, f"Exercise '{eid}' has no matching solution cell"

    for cell in nb.cells:
        eid = cell.metadata.get("exercise_id")
        if eid and eid in solutions:
            cell.source = solutions[eid]

    slow = _is_slow(nb)
    if slow and os.environ.get("SKIP_SLOW"):
        pytest.skip("Skipping slow notebook (SKIP_SLOW is set)")

    timeout = SLOW_TIMEOUT if slow else FAST_TIMEOUT
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    ep.preprocess(nb, resources={"metadata": {"path": str(exercise_dir)}})

    for i, cell in enumerate(nb.cells):
        for output in cell.get("outputs", []):
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            if "FAIL" in text:
                pytest.fail(f"Cell {i} output contains FAIL:\n{text}")
