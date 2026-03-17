# Claude Code + LaTeX + Zotero Course Template

A template for building a course as a set of Beamer presentations with accompanying Jupyter notebooks, powered by Claude Code and Zotero-based literature search (RAG).

## Features

- **Multiple standalone lectures** - Each lecture compiles independently as a Beamer PDF
- **Jupyter notebooks** - One per lecture, with Google Colab badges for easy sharing
- **Literature RAG** - Zotero papers are auto-indexed and searchable via embeddings
- **Shared config** - Course metadata, theme, and bibliography shared across all lectures

## Quick Start

1. **Use this template** to create a new repository
2. Run the setup script:
   ```bash
   chmod +x scripts/setup/*.sh
   ./scripts/setup/setup.sh
   ```
3. Edit `lib/metadata.tex` with your course title and name
4. Set up Zotero integration (optional):
   ```bash
   ./scripts/setup/setup_zotero.sh
   ```
5. Build the template lecture:
   ```bash
   make lecture-01_intro
   ```

## Project Structure

```
lectures/
  01_intro/
    notes.md          # Your draft notes (source of truth)
    slides.tex        # Beamer presentation
    notebook.ipynb    # Jupyter notebook with Colab link
lib/
  metadata.tex        # Course title, instructor name
  preamble.tex        # Shared preamble for all lectures
  packages.tex        # LaTeX packages
  settings.tex        # Beamer theme settings
bib/                  # Auto-synced bibliography from Zotero
scripts/              # Sync pipeline and RAG tools
Makefile              # Build system
```

## Building Lectures

```bash
make lecture-01_intro    # Build a single lecture
make all                 # Build all lectures
make clean               # Remove build artifacts
```

Output PDFs go to `lectures/XX_name/output/slides.pdf`.

## Creating a New Lecture

1. Copy an existing lecture folder: `cp -r lectures/01_intro lectures/02_topic`
2. Edit `slides.tex` - update title
3. Edit `notebook.ipynb` - update Colab badge URL path
4. Write your notes in `notes.md`

## Literature Search

```bash
# Start the Zotero sync daemon
nohup uv run python scripts/syncing/main.py > /dev/null 2>&1 &

# Search your papers
uv run python scripts/tools/rag.py "your query"
```

## Google Colab Integration

Each notebook includes an "Open in Colab" badge. Update the GitHub URL in the badge to match your repository:

```
https://colab.research.google.com/github/USER/REPO/blob/main/lectures/XX_name/notebook.ipynb
```

## Zotero Setup

### Better BibTeX Setup
1. Open Zotero > Tools > Add-ons > Install Add-on From File (select the downloaded .xpi)
2. Restart Zotero
3. Go to Zotero > Settings > Better BibTeX > Set Citation key format to `auth.lower + year`
4. Add some sources to your collection
5. Right-click on your collection > Export Collection...
6. Choose format: Better BibTeX, enable 'Keep Updated', do NOT enable 'Export Files'
7. Save to `bib/refs.bib` in your project

### Zotero API (Optional)
To let Claude Code manage your Zotero library directly:
1. Get an API key at: https://www.zotero.org/settings/keys
2. Set `ZOTERO_API_KEY` in your environment

## How It Works

1. **Write naturally**: Draft your ideas in `lectures/*/notes.md`
2. **Add references**: Import papers to Zotero; they auto-sync to your project
3. **Let AI help**: Ask the assistant to transform notes into polished Beamer frames
4. **Search your literature**: Both you and the AI can search across all your papers using the RAG system

AI assistant instructions are configured via `CLAUDE.md` and `.claude/` in the repo.

## Notes
- Supports macOS and Windows. For Linux, adapt the setup scripts manually.
- See `CLAUDE.md` for assistant behavior rules.
