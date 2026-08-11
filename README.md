# Mechanistic Interpretability — Iliad Intensive

The exercise notebooks for C.2. Open them in Colab; nothing needs installing.

The slides are no longer in this repo. They are built from LaTeX in
[`iliad-team/iliad-intensive`](https://github.com/iliad-team/iliad-intensive/tree/main/tex/mechanistic-interpretability)
and published on the curriculum site:

- [C.2 — Mechanistic Interpretability](https://iliad-team.github.io/iliad-intensive/interpretability/mechanistic-interpretability)
- [Slides (PDF)](https://iliad-team.github.io/iliad-intensive/downloads/mechanistic-interpretability/mechanistic-interpretability-slides.pdf)

## Exercises and external links (in lecture order)

1. [Feature Visualization exercise](https://colab.research.google.com/github/iliad-team/iliad-intensive-C.2/blob/main/exercises/03_feature_viz/notebook.ipynb)
2. Logit Lens exercise — [normal](https://colab.research.google.com/github/iliad-team/iliad-intensive-C.2/blob/main/exercises/01_logit_lens/notebook_normal.ipynb) · [hard](https://colab.research.google.com/github/iliad-team/iliad-intensive-C.2/blob/main/exercises/01_logit_lens/notebook_hard.ipynb)
3. [Neuronpedia — SAE features (Gemma 3 27B)](https://www.neuronpedia.org/gemma-3-27b/31-gemmascope-2-res-16k)
4. [Sparse Autoencoders exercise](https://colab.research.google.com/github/iliad-team/iliad-intensive-C.2/blob/main/exercises/04_saes/notebook.ipynb)
5. [Neuronpedia — Attribution graphs (Gemma 2 2B)](https://www.neuronpedia.org/gemma-2-2b/graph)
6. Induction Heads exercise — [normal](https://colab.research.google.com/github/iliad-team/iliad-intensive-C.2/blob/main/exercises/02_induction_heads/notebook_normal.ipynb) · [hard](https://colab.research.google.com/github/iliad-team/iliad-intensive-C.2/blob/main/exercises/02_induction_heads/notebook_hard.ipynb)
7. [Neuronpedia — Natural Language Autoencoders (Llama 3.3 70B)](https://www.neuronpedia.org/llama3.3-70b-it/nla)

## Discussion reading

- Nanda et al., [A Pragmatic Vision for Interpretability](https://www.alignmentforum.org/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability), 2025
- Ségerie, [Against Almost Every Theory of Impact of Interpretability](https://www.lesswrong.com/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability-1), 2023
- Hendrycks, [The Misguided Quest for Mechanistic AI Interpretability](https://ai-frontiers.org/articles/the-misguided-quest-for-mechanistic-ai-interpretability), 2025
- Chughtai, [Activation Space Interpretability May Be Doomed](https://www.alignmentforum.org/posts/gYfpPbww3wQRaxAFD/activation-space-interpretability-may-be-doomed), 2025

## Running locally

```bash
uv sync
uv run pytest tests/ -v    # executes each notebook with its solutions injected
```
