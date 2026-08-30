"""Materialize docs/downloads/*.py from the dynaplex package at build time.

The tutorial download files are single-sourced in the main repo (shipped in
the wheel as dynaplex/examples/ plus dynaplex/models/binpacking.py); nothing
under docs/downloads/ is checked in. Resolution order mirrors the mkdocstrings
config: the staged package tree of a sibling DynaPlex checkout when present
(phase-1 local builds), otherwise the installed dynaplex package (docs CI
installs the wheel). Sources are located without importing dynaplex.
"""
from pathlib import Path
import importlib.util

REPO = Path(__file__).resolve().parent.parent
DOWNLOADS = REPO / "docs" / "downloads"
STAGED = REPO.parent / "DynaPlex" / "build" / "Release" / "python" / "dynaplex"

# download filename -> path inside the dynaplex package
SOURCES = {
    "airplane_mdp_example.py": "examples/airplane_mdp_example.py",
    "airplane_statistics_example.py": "examples/airplane_statistics_example.py",
    "binpacking.py": "models/binpacking.py",
    "binpacking_dcl.py": "examples/binpacking_dcl.py",
}


def _package_root() -> Path:
    if STAGED.is_dir():
        return STAGED
    spec = importlib.util.find_spec("dynaplex")
    if spec and spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations)))
    raise SystemExit(
        "materialize_downloads: dynaplex package not found — expected the "
        f"staged tree at {STAGED} or an installed dynaplex wheel."
    )


def on_pre_build(config):
    root = _package_root()
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    for name, rel in SOURCES.items():
        src = root / rel
        if not src.is_file():
            raise SystemExit(f"materialize_downloads: missing source {src}")
        dst = DOWNLOADS / name
        content = src.read_bytes()
        # Only write on real change: `mkdocs serve` watches docs/, so an
        # unconditional copy retriggers the build in an endless reload loop.
        if not (dst.is_file() and dst.read_bytes() == content):
            dst.write_bytes(content)
