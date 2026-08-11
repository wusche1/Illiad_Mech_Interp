update-links:
	uv run python scripts/tools/update_colab_links.py

test:
	uv run pytest tests/ -v

.PHONY: update-links test
