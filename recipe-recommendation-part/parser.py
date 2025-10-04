import ast
import pandas as pd
import json

def parse_list(x):
    """Convert stringified lists into Python lists safely"""
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return list(x)

def parse_set(x):
    """Convert stringified lists into Python sets safely"""
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


with open(r"D:\docs\24679\project1\outputs\new_map.json", "r", encoding="utf-8") as f:
    PARENT_MAP = json.load(f)

CHILD2PARENT = {}
for parent, children in PARENT_MAP.items():
    for child in children:
        CHILD2PARENT[child.lower()] = parent.lower()

def get_parent(ingredient: str):
    return CHILD2PARENT.get(ingredient.strip().lower())