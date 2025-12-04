# DynaPlex Docs

Documentation website for DynaPlex 2.*.* (Python-only version).

This documentation was migrated from the DynaPlexPrivate repository's docs directory. The structure and configuration files have been set up for Python-only documentation.

## Setup

To build the documentation locally:

1. Install dependencies using Poetry:
   ```bash
   poetry install --with docs
   ```

2. Build the documentation:
   ```bash
   poetry run make html
   ```

See `build_docs.md` for more details.

## Note on Content

Many documentation files still contain references to C++ code and need to be updated for the Python-only version. Files that likely need review include:
- `getting_started/adding_model.rst` - Contains C++ code examples
- `getting_started/adding_executable.rst` - C++/CMake specific (removed from index)
- `tutorial/adding_mdp.rst` - Contains C++ code examples
- Various reference files in `reference/` directory

The configuration files (`conf.py`, `pyproject.toml`) have been updated to remove C++ dependencies (Breathe extension removed).

