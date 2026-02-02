# DynaPlex 2 Docs

DynaPlex 2 is a Python library for solving Markov Decision Problems and similar models (POMDP, HMM). MDP/POMDP Models in DynaPlex 2 are written in (a subset of) Python, and are interpreted using a custom fast interpreter that supports multi-threading and vectorization. DynaPlex 2 focuses on solving problems arising in Operations Management: Supply Chain, Transportation and Logistics, Manufacturing, etc.

## Requirements

- Python 3.12 only
- Windows: AMD64
- Linux: x86_64
- macOS: modern hardware (Apple Silicon) only; no legacy intel hardware supported.
- Algorithms require a compatible PyTorch install; use the PyTorch selector: https://pytorch.org/get-started/locally/

build from source presently not supported.  

## Installation

```bash
pip install dynaplex
```

or 

```bash
pip install dynaplex[complete]
```

## Documentation

Full docs and tutorials are available at https://dynaplex-docs.readthedocs.io/en/latest/  
[ReadTheDocs](https://dynaplex-docs.readthedocs.io/en/latest/)

## Support

For support, bug reports, questions about DynaPlex and the docs, please use the github issue tracker: https://github.com/WillemvJ/DynaPlex-docs/issues
