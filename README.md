# DynaPlex

DynaPlex is a Python library for solving Markov Decision Problems and
similar models (POMDP, HMM), aimed at problems arising in operations management (OM), e.g. supply chain management,
logistics, transportation, manufacturing and maintenance optimization. Models are written in
**DynaML** — a subset of Python suitable for modelling OM problems — and executed by a fast multi-threaded
engine with a bundled LLVM JIT: you read and write canonical Python code,
and get auto-vectorized C++ speed. Just as importantly, the library bundles
algorithms such as deep controlled learning (DCL), specifically designed for highly stochastic problems
arising in OM.

**[Documentation & tutorials](https://dynaplex.github.io/DynaPlex/)** ·
**[PyPI](https://pypi.org/project/dynaplex/)** ·
**[Discussions](https://github.com/DynaPlex/DynaPlex/discussions)**

## A new DynaPlex

DynaPlex has been rewritten from the ground up around a purpose-built
engine that compiles and runs DynaML at native (C++) speed,
multi-threaded, with the LLVM JIT bundled in the wheel. No C++ and no
build step: `pip install dynaplex` is the whole setup.

With the rewrite, the engine's source is closed. DynaPlex ships as free
pre-built wheels on PyPI, and this repository is the library's public
home: the documentation source, the issue tracker and the discussions
live here. Documentation contributions are welcome — every docs page has
an edit button that leads back to this repo.

The original C++20 library is preserved in full — all branches, including
the research branches — in the archived
[DynaPlex/DynaPlex-legacy](https://github.com/DynaPlex/DynaPlex-legacy).

## Installation

```bash
pip install dynaplex
```

DynaPlex is distributed as pre-built, self-contained wheels — there is no
build-from-source step and no external LLVM to install.

| | |
|---|---|
| Python | 3.13 only |
| macOS (Apple Silicon) | arm64, macOS 14.0 or newer |
| macOS (Intel) | x86_64, macOS 15.0 or newer |
| Linux | x86_64, glibc 2.28 or newer (`manylinux_2_28`) |
| Windows | AMD64 (64-bit) |

The only required runtime dependency is NumPy. PyTorch is needed only for
the reinforcement-learning algorithms; install it separately via the
[PyTorch selector](https://pytorch.org/get-started/locally/).

Verify your install (including the JIT) with:

```bash
python -m dynaplex.selftest
```

## Citing

When using DynaPlex in your research, please cite:

```bibtex
@software{DynaPlex,
  author = {Akkerman, Fabian and Begnardi, Luca and {Lo Bianco}, Riccardo and Temizoz, Tarkan and Mes, Martijn and {van Jaarsveld}, Willem},
  title = {{DynaPlex} software library and documentation},
  url = {https://github.com/DynaPlex/DynaPlex},
  year = {2026}
}
```

## Getting help

- **Questions and discussion** —
  [GitHub Discussions](https://github.com/DynaPlex/DynaPlex/discussions).
- **Bug reports and DynaML feature requests** —
  [GitHub issues](https://github.com/DynaPlex/DynaPlex/issues); the issue
  forms tell you what to include.
- **Documentation fixes** — edit the page directly via its edit button, or
  open a PR against `docs/` in this repository.

## Working on the documentation

The docs are built with
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/mkdocs serve        # live preview at http://127.0.0.1:8000
```

`mkdocs build --strict` must pass (no warnings) before a change can be
merged. Content lives in `docs/` (Markdown). The runnable example scripts
in `docs/downloads/` are copies of tested originals maintained alongside
the engine — please point out problems in them via an issue rather than
editing them in place.
