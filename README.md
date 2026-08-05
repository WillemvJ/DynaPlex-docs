# DynaPlex documentation

DynaPlex is a Python library for solving Markov Decision Problems and similar
models (POMDP, HMM). MDP/POMDP models in DynaPlex are written in (a subset
of) Python, and are interpreted using a custom fast interpreter that supports
multi-threading and vectorization. DynaPlex focuses on solving problems
arising in Operations Management: Supply Chain, Transportation and Logistics,
Manufacturing, etc.

This repository holds the documentation source, built with
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).

## Working on the docs

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/mkdocs serve        # live preview at http://127.0.0.1:8000
```

`mkdocs build --strict` must pass (no warnings) before publishing.

- Content lives in `docs/` (Markdown).
- `docs/downloads/` holds the runnable example scripts included in the
  tutorial pages; they are copies of the tested originals in the code repo —
  update them from there, do not edit in place.
- Versioned deployment is done with [mike](https://github.com/jimporter/mike)
  (see the docs plan in the code repo, `roadmap/docs_community_strategy.md`).

## Legacy

The previous Sphinx/reStructuredText source is preserved unchanged under
`source/` until the migration is finalized, then deleted.
