"""Utility helpers for parsing stored recipe metadata."""

import ast
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Set

import pandas as pd


def parse_list(x: Any) -> list:
    """Convert stringified lists into Python lists safely."""

    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return list(x)


def parse_set(x: Any) -> Set[str]:
    """Convert stringified iterables into Python sets safely."""

    if pd.isna(x):
        return set()
    if isinstance(x, str):
        try:
            val = ast.literal_eval(x)
        except Exception:
            return {x.strip()}
    else:
        val = x
    if isinstance(val, (list, tuple, set)):
        return set(val)
    return {val}


@lru_cache(maxsize=1)
def _load_parent_map() -> dict:
    """Load the parent ingredient mapping shipped with the repository."""

    base_dir = Path(__file__).resolve().parent
    mapping_path = base_dir / "outputs" / "new_map.json"
    if not mapping_path.exists():
        raise FileNotFoundError(
            "Ingredient parent map not found. Expected at "
            f"{mapping_path}."
        )
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _build_child_lookup() -> dict:
    parent_map = _load_parent_map()
    lookup = {}
    for parent, children in parent_map.items():
        for child in children:
            lookup[child.lower()] = parent.lower()
    return lookup


def get_parent(ingredient: str):
    """Return the parent category for an ingredient name if available."""

    if not ingredient:
        return None
    return _build_child_lookup().get(ingredient.strip().lower())
