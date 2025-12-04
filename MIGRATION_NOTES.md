# Migration Notes

This documentation was migrated from `DynaPlexPrivate/docs` to serve as the basis for DynaPlex 2.*.* Python-only documentation.

## Changes Made

### Configuration Files
- ✅ `pyproject.toml` - Removed `breathe` dependency (C++ documentation tool)
- ✅ `.readthedocs.yaml` - Updated for new repository structure
- ✅ `Makefile` - Copied as-is (no changes needed)
- ✅ `source/conf.py` - Removed Breathe extension and C++-related configurations:
  - Removed `breathe` from extensions
  - Removed `breathe_projects` configuration
  - Removed `.xml` from `source_suffix` (now only `.rst`)
  - Updated path to `pyproject.toml` (from `../../` to `../`)
  - Updated version to v2.0
  - Commented out `sys.path.insert` (may need adjustment based on project structure)

### Content Files
- ✅ `source/index.rst` - Updated to reflect Python-only focus:
  - Removed C++ references from description
  - Removed section about cloning with submodules (C++ specific)
  - Removed `getting_started/adding_executable` from table of contents (C++/CMake specific)

### Files That Need Review/Update

The following files still contain C++ code examples and references that should be updated for Python:

1. **Getting Started:**
   - `getting_started/adding_model.rst` - Contains C++ code examples, CMake references
   - `getting_started/testing.rst` - May contain C++ test references

2. **Tutorial:**
   - `tutorial/adding_mdp.rst` - Contains extensive C++ code examples
   - `tutorial/setup.rst` - May contain C++ setup instructions
   - `tutorial/policy.rst` - May contain C++ references

3. **Reference:**
   - `reference/*.rst` - All reference files likely contain C++ API documentation that needs to be replaced with Python API docs

4. **Other:**
   - `faq/faq.rst` - May contain C++-related Q&A
   - `legacy/legacy.rst` - May reference old C++ features

## Next Steps

1. Review and update all content files to replace C++ examples with Python equivalents
2. Update API reference documentation to reflect Python API
3. Remove or update any remaining C++-specific instructions
4. Test the documentation build process
5. Update repository URLs in `conf.py` if needed

