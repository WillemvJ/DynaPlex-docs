# Development Setup

## Prerequisites

1. Install Poetry (e.g. with `py`)
2. Navigate to the root of `DynaPlex-Docs`

## Build Documentation

### 1. Update lock file

```bash
py -m poetry lock
```

### 2. Install dependencies

```bash
py -m poetry install --with docs
```

### 3. Build docs

```bash
py -m poetry run sphinx-build -M html source build
```

### 4. View the result

```bash
start build\html\index.html
```
