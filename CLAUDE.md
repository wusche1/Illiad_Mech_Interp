# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a course/lecture template using Claude Code + Beamer + Zotero with RAG-based literature search.
Each lecture is a standalone Beamer presentation with an accompanying Jupyter notebook.

## Build Commands

```bash
# Compile a single lecture (e.g., 01_intro)
make lecture-01_intro

# Compile all lectures
make all

# Clean build artifacts
make clean

# Run Zotero sync daemon (extracts papers, creates embeddings)
uv run python scripts/syncing/main.py

# Search literature with RAG
uv run python scripts/tools/rag.py "your query"
```

## Architecture

### Lecture Structure
- `lectures/XX_name/notes.md` - User's draft notes and source material
- `lectures/XX_name/slides.tex` - Standalone Beamer presentation (AI writes from notes)
- `lectures/XX_name/notebook.ipynb` - Jupyter notebook with Google Colab badge
- `lectures/XX_name/claude_notes.md` - AI's research notes (only when explicitly instructed)

### Shared LaTeX Config
- `lib/preamble.tex` - Shared preamble included by all lectures
- `lib/packages.tex` - Beamer package imports
- `lib/settings.tex` - Beamer theme and formatting
- `lib/metadata.tex` - Course metadata (`\coursetitle`, `\instructor`, `\institution`)

### Bibliography System
- `bib/refs.bib` - BibTeX file auto-synced from Zotero
- `bib/labels.bib` - Extracted LaTeX labels for autocomplete
- `bib/[citation_key]/` - Per-paper folders containing fulltext, chapters, metadata

### Sync Pipeline (`scripts/syncing/main.py`)
Runs continuously (30s interval) via nohup in the background. Check if it's running with `pgrep -f syncing/main.py`. If not running, start it with:
```bash
nohup uv run python scripts/syncing/main.py > /dev/null 2>&1 &
```

### Experiments
- `experiments/` - Cloned experiment repositories (git-ignored)

### Configuration
`scripts/config.yaml` controls Zotero sync, text extraction, chapter splitting, and embeddings settings.

## Key Rules

1. **Never modify `_fulltext.md` files** - Read-only source material
2. **Only write `slides.tex` when explicitly asked** - User writes notes, AI transforms to Beamer frames
3. **Put each sentence on a new line** in LaTeX files (improves diffs, invisible in PDF)
4. **Read existing slides before editing** to maintain context and consistency
5. **Verify citations** by reading the corresponding `bib/[key]/[key]_fulltext.md`
6. **Search literature** when answering questions about papers or verifying claims
7. **Notes are the source of truth** - slides should never contain information not in notes
8. **Never manually edit `bib/refs.bib`** - Auto-synced from Zotero. Use the Zotero skill to add papers.
9. **Avoid em-dashes (---)** - Use commas, parentheses, or separate sentences instead
10. **Use `[fragile]`** on frames with verbatim or code content
11. **Use `[allowframebreaks]`** on bibliography frames
12. **Keep frames concise** - Bullet points, not paragraphs
13. **New lectures**: Copy an existing lecture folder, update the title and Colab badge URL
14. **Notebooks must have the Colab badge** as the first markdown cell
15. **Shared metadata** lives in `lib/metadata.tex` - update there, not per-lecture
