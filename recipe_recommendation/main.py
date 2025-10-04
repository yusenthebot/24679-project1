"""Utility entry-points for the recipe recommendation engine."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .candidate import ml_generate_candidates
from .highlight import print_candidates
from .parser import get_parent, parse_list, parse_set
from .train_model import cold_start_classifier, train_model_classifier

BASE_DIR = Path(__file__).resolve().parent
RECIPES_FILE = BASE_DIR / "outputs" / "recipes_cleaned_strict.csv"


def load_recipes(recipes_path: Optional[os.PathLike] = None) -> pd.DataFrame:
    """Load the curated recipe dataset shipped with the repository."""

    path = Path(recipes_path) if recipes_path else RECIPES_FILE
    if not path.exists():
        raise FileNotFoundError(
            "Recipe dataset not found. Expected to find it at "
            f"{path}."
        )

    df = pd.read_csv(path)

    # Parse ingredient-related columns
    for col in ["staple", "main", "seasoning", "other"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_list)

    for col in [
        "staple_parent",
        "main_parent",
        "seasoning_parent",
        "other_parent",
        "cuisine_attr",
        "region",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(parse_set)

    return df


def extract_user_parents(
    detection_payload: Dict[str, Any],
    high_conf_threshold: float = 0.8,
) -> Tuple[List[str], List[str], List[str]]:
    """Map detected ingredients to parent categories for recommendation."""

    user_parents: List[str] = []
    high_conf: List[str] = []
    low_conf: List[str] = []

    for ing in detection_payload.get("ingredients", []):
        name = ing.get("name", "").strip().lower().replace("_", " ")
        parent = get_parent(name)
        conf = ing.get("confidence", 0)
        if parent and conf >= high_conf_threshold:
            user_parents.append(parent)
            high_conf.append(f"{name} → {parent}")
        elif parent:
            low_conf.append(f"{name} → {parent}")
        else:
            low_conf.append(name or "unknown")

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for parent in user_parents:
        if parent not in seen:
            deduped.append(parent)
            seen.add(parent)

    return deduped, high_conf, low_conf


def ensure_user_assets(user_id: str = "user1") -> Tuple[Path, Path, Path]:
    """Ensure the feature store and model exist for the given user."""

    user_dir = BASE_DIR / "user-data" / user_id
    profile_file = user_dir / "user_profile.json"
    features_file = user_dir / "user_features.csv"
    model_file = user_dir / "ranker.pkl"

    if not profile_file.exists():
        raise FileNotFoundError(
            "Missing user profile. Create or configure a profile at "
            f"{profile_file}."
        )

    if not features_file.exists():
        print("No features found, running cold-start simulation...")
        cold_start_classifier(user_id, recipes_file=str(RECIPES_FILE))
        train_model_classifier(user_id, recipes_file=str(RECIPES_FILE))
    elif not model_file.exists():
        print("Features exist but no model found, training model...")
        train_model_classifier(user_id, recipes_file=str(RECIPES_FILE))

    return profile_file, features_file, model_file


def recommend_recipes(
    detection_payload: Dict[str, Any],
    user_profile: Dict[str, Any],
    user_id: str = "user1",
    recipes_df: Optional[pd.DataFrame] = None,
    topk: int = 5,
) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """Generate a ranked list of recipe candidates for the detected items."""

    if recipes_df is None:
        recipes_df = load_recipes()

    user_parents, high_conf, low_conf = extract_user_parents(detection_payload)

    _, _, model_file = ensure_user_assets(user_id)

    candidates = ml_generate_candidates(
        recipes_df,
        user_parents,
        user_profile=user_profile,
        model_path=str(model_file),
        topk=topk,
    )

    return candidates, user_parents, high_conf, low_conf


def cli_entrypoint(user_id: str = "user1") -> None:
    """Simple CLI entry-point mirroring the legacy behaviour."""

    profile_file, _, model_file = ensure_user_assets(user_id)
    with profile_file.open("r", encoding="utf-8") as f:
        user_profile = json.load(f)

    payload_path = BASE_DIR / "recipe_input.json"
    if not payload_path.exists():
        raise FileNotFoundError(
            "Detection payload not found. Run the detector first to generate "
            f"{payload_path}."
        )

    with payload_path.open("r", encoding="utf-8") as f:
        detection_payload = json.load(f)

    recipes_df = load_recipes()
    candidates, user_parents, high_conf, low_conf = recommend_recipes(
        detection_payload,
        user_profile,
        user_id=user_id,
        recipes_df=recipes_df,
    )

    print("\nHigh-confidence ingredients:")
    for item in high_conf:
        print(f"  - {item}")
    if low_conf:
        print("\nLow-confidence or unmatched ingredients:")
        for item in low_conf:
            print(f"  - {item}")

    print(f"\nFound {len(candidates)} candidate recipes:\n")
    print_candidates(candidates, user_parents, topk=len(candidates))


if __name__ == "__main__":
    cli_entrypoint("user1")
