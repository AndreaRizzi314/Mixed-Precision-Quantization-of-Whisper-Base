"""Locate repository inputs independently of the command's working directory."""
from pathlib import Path


def repository_root() -> Path:
    # The tools operate on this research checkout, installed with `pip install -e .`.
    root = Path(__file__).resolve().parents[2]
    if not (root / "configs" / "search.json").is_file():
        raise RuntimeError("Install this repository in editable mode: python -m pip install -e .")
    return root
