import datetime
import json
import urllib.request

# -- Project information
# Note: If you need to import Python modules from the main DynaPlex project,
# uncomment and adjust the path below:
# sys.path.insert(0, os.path.abspath('../../'))

now = datetime.date.today()


# -- Version: single public source of truth -------------------------------
# The DynaPlex code lives in a private repo, so its git tags are not visible to
# this (public) docs build. The one public thing both agree on is the PUBLISHED
# PyPI release, so we key the docs version off that: every time a release is
# published, the docs version tracks it automatically -- no manual bump, no
# cross-repo secret. Uses the PyPI JSON API (not `pip install`) so it is
# independent of this build's Python version and the wheel's ABI.
def _pypi_version(package: str, fallback: str = "0.0.0+unknown") -> str:
    try:
        url = f"https://pypi.org/pypi/{package}/json"
        with urllib.request.urlopen(url, timeout=10) as fh:
            return json.load(fh)["info"]["version"]
    except Exception:
        return fallback


release = _pypi_version("dynaplex")   # full package version, e.g. "1.10.1"
version = release                     # shown in the version dropdown / theme

# The site TITLE uses the bare name "DynaPlex" so it reads e.g. "DynaPlex 1.10.1
# documentation" (not "DynaPlex 2 1.10.1", which double-counts the generation).
# Descriptive prose elsewhere still refers to the product as "DynaPlex 2"; once a
# 2.0 package is published the version alone makes the generation clear.
project = "DynaPlex"
authors = "DynaPlex contributors"
copyright = f"2023 - {now.year}, {authors}"

# -- API documentation
autoclass_content = "class"
autodoc_member_order = "bysource"
autodoc_typehints = "signature"

# -- nbsphinx
nbsphinx_execute = "always"

# -- General configuration
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_immaterial",
    "nbsphinx",
]

exclude_patterns = []

source_suffix = ['.rst']

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

add_module_names = False
python_use_unqualified_type_names = True

# -- Options for HTML output
html_theme = "sphinx_immaterial"
html_logo = "assets/images/icon.png"
html_static_path = ['_static']
html_theme_options = {
    "site_url": "https://dynaplex.nl/",
    "repo_url": "https://github.com/dynaplex/dynaplex-docs/",
    "icon": {
        "repo": "fontawesome/brands/github",
        "edit": "material/file-edit-outline",
    },
    "features": [
        "navigation.top",
        "navigation.path",
        "navigation.prune",
        "toc.follow",
        "toc.integrate",
        "navigation.indexes"
    ],
    "palette": [
        {
            "media": "(prefers-color-scheme: light)",
            "primary": "DarkCyan",
            "accent": "CornflowerBlue",
            "scheme": "default",
            "toggle": {
                "icon": "material/lightbulb-outline",
                "name": "Switch to dark mode",
            },
        },
        {
            "media": "(prefers-color-scheme: dark)",
            "primary": "DarkCyan",
            "accent": "CornflowerBlue",
            "scheme": "slate",
            "toggle": {
                "icon": "material/lightbulb",
                "name": "Switch to light mode",
            },
        },
    ],
    "version_dropdown": True,
    "version_info": [
        {
            "version": "",
            "title": f"v{release}",
            "aliases": [],
        },
    ],
}

# -- Options for EPUB output
epub_show_urls = "footnote"
