install poetry (e.g. with py)
navigate to root of DynaPlex-Docs


# 1. Update lock file
py -m poetry lock

# 2. Install dependencies
py -m poetry install --with docs --no-root

# 3. Build docs
py -m poetry run sphinx-build -M html source build

# 4. View the result
start build\html\index.html