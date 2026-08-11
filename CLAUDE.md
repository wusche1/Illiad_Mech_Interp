# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The exercise notebooks for the C.2 Mechanistic Interpretability day of the Iliad
Intensive. Students open them in Colab straight from `main`, so the URLs baked
into each notebook must always point at this repo.

**The slides are not here.** They were ported to
[`iliad-team/iliad-intensive`](https://github.com/iliad-team/iliad-intensive)
under `tex/mechanistic-interpretability/`, which compiles and publishes them.
Edit them there. This repo's `lectures/`, `lib/`, `bib/` and the Zotero/docling
authoring pipeline were removed for that reason — do not restore them from
upstream (`wusche1/Illiad_Mech_Interp`) without deciding which copy wins.

## Commands

```bash
uv sync                                     # install deps
uv run pytest tests/ -v                     # execute every notebook with solutions injected
SKIP_SLOW=1 uv run pytest tests/ -v         # skip notebooks needing model downloads/GPU
make update-links                           # rewrite Colab + raw URLs from the git remote
uv run python scripts/tools/test_colab.py   # drive a notebook through real Colab (playwright)
```

## Exercise System

### File structure

Each exercise lives in `exercises/YY_name/` with two files:
- `notebook.ipynb` (or `notebook_normal.ipynb` + `notebook_hard.ipynb`) — what students open, and the single source of truth
- `utils.py` — test/check functions (print PASS/FAIL). Notebooks fetch it from this repo's raw URL when running on Colab.

### Notebook cell pattern

Per exercise within a notebook, cells appear in this order:
1. **Markdown** — explanation + instructions
2. **Code cell** — skeleton with `# TODO`. Cell metadata must include `"exercise_id": "some_id"`
3. **Test cell** — calls check function from `utils.py` (e.g. `test_tangent(tangent)`)
4. **Hint/Solution markdown** — collapsible `<details>` blocks with the full standalone solution in a ` ```python ``` ` block. Cell metadata must include `"solution_id": "some_id"` matching the exercise

The solution code block must be a complete, standalone replacement for the exercise cell (full function/class definition, not just the body).

The notebook also needs at the top:
1. **Colab badge** as first markdown cell (REQUIRED)
2. **Setup cell** — fetches `utils.py` from GitHub raw URL on Colab, uses importlib reload

### Testing

`tests/test_notebooks.py` discovers notebooks under `exercises/*/`. For each one it:
1. Finds all cells with `exercise_id` metadata and all cells with `solution_id` metadata
2. Extracts the python code from each solution cell's ` ```python ``` ` block
3. Asserts every exercise has a matching solution
4. Swaps the solution code into the exercise cell
5. Executes the entire notebook
6. Fails if any cell errors or any output contains "FAIL"

No separate solutions file needed. The notebook is the single source of truth.

`tests/test_colab_compat.py` checks the Colab-specific scaffolding (badge, setup cell) rather than running anything.

### Adding a new exercise

1. Create `exercises/YY_name/`
2. Author `notebook.ipynb` directly in Jupyter/Colab following the cell pattern above
3. Write `utils.py` with test functions that print PASS/FAIL
4. Tag exercise code cells with `exercise_id` and solution markdown cells with `solution_id` in cell metadata
5. Run `make update-links` to fix Colab badge and raw GitHub URLs
6. Verify: `uv run pytest tests/ -v` should pick it up automatically

## Key Rules

1. **Exercises are the main value prop** — participants can read the slides on their own
2. **Notebooks must have the Colab badge** as the first markdown cell
3. **Every self-reference points at `iliad-team/iliad-intensive-C.2`**, not at the upstream repo — badges, raw `utils.py` URLs, README links
4. **Never break `main`** — students open notebooks from it live during the session
5. **Slide changes belong in `iliad-team/iliad-intensive`**, not here
