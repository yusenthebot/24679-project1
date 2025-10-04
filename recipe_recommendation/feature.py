import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher

# _HASHTRICK_DIM = 512

def build_features(recipe, user_profile):
    """
    Build feature dictionary for rule-based scoring and ML ranker input.
    Both numeric and categorical features are included.
    """
    features = {}

    # Ingredient coverage
    # main = recipe.get("main", set()) or set()
    # staple = recipe.get("staple", set()) or set()
    # other = recipe.get("other", set()) or set()

    # matched_main = recipe.get("matched_main", 0)
    # matched_staple = recipe.get("matched_staple", 0)
    # matched_other = recipe.get("matched_other", 0)

    total_main = len(recipe["main"])
    total_other = len(recipe["other"])
    total_staple = len(recipe["staple"])

    features["main_match_ratio"] = recipe["matched_main"] / max(total_main, 1)
    features["other_match_ratio"] = recipe["matched_other"] / max(total_other, 1)
    features["staple_match_ratio"] = recipe["matched_staple"] / max(total_staple, 1)

    features["missing_main_count"] = total_main - recipe["matched_main"]
    features["missing_other_count"] = total_other - recipe["matched_other"]
    features["missing_staple_count"] = total_staple - recipe["matched_staple"]

    # Nutrition
    features["calories"] = recipe.get("calories", 0)
    features["protein"] = recipe.get("protein", 0)
    features["fat"] = recipe.get("fat", 0)
    features["protein_ratio"] = features["protein"] / max(features["calories"], 1)
    features["fat_ratio"] = features["fat"] / max(features["calories"], 1)

    # Region / cuisine preferences
    recipe_region = recipe.get("region", "")
    if isinstance(recipe_region, set):
        features["region_match"] = 1 if any(r in user_profile.get("preferred_regions", []) for r in recipe_region) else 0
    else:
        features["region_match"] = 1 if recipe_region in user_profile.get("preferred_regions", []) else 0

    # # Tag preferences
    # preferred_tags = set(user_profile.get("preferred_tags", []))
    # recipe_tags = set(recipe.get("tags", []))
    # features["tag_overlap"] = len(preferred_tags & recipe_tags)

    # Diet constraints
    veg_type = (
                user_profile.get("diet", {}).get("vegetarian_type", "")
                or ""
            ).lower()
    is_veg_user = veg_type in ["vegetarian", "flexible_vegetarian", "vegan"]
    features["is_vegetarian_safe"] = (0 if is_veg_user and recipe.get("contains_meat", False) else 1)
    
    features["low_calorie_penalty"] = (
        1 if recipe.get("calories", 0) <= user_profile.get("calorie_threshold", 9999) else 0
    )

    # Preferred / disliked mains
    preferred_main = set(user_profile.get("other_preferences", {}).get("preferred_main", []))
    disliked_main = set(user_profile.get("other_preferences", {}).get("disliked_main", []))
    recipe_main = set(recipe.get("main", []))

    features["preferred_main_overlap"] = len(recipe_main & preferred_main)
    features["disliked_main_overlap"] = len(recipe_main & disliked_main)

    # Cooking time
    cooking_time_max = user_profile.get("other_preferences", {}).get("cooking_time_max", None)
    if cooking_time_max:
        features["within_cooking_time"] = 1 if recipe.get("minutes", 9999) <= cooking_time_max else 0
    else:
        features["within_cooking_time"] = 1

    # # Parent hash-trick features
    # parent_tokens = []
    # for p in recipe.get("main_parent", set()) or []:
    #     parent_tokens.append(f"main={p}")
    # for p in recipe.get("staple_parent", set()) or []:
    #     parent_tokens.append(f"staple={p}")
    # for p in recipe.get("seasoning_parent", set()) or []:
    #     parent_tokens.append(f"seasoning={p}")
    # for p in recipe.get("other_parent", set()) or []:
    #     parent_tokens.append(f"other={p}")

    # if parent_tokens:
    #     hasher = FeatureHasher(n_features=_HASHTRICK_DIM, input_type="string", alternate_sign=False)
    #     hashed_vec = hasher.transform([parent_tokens]).toarray()[0]
    #     for i, val in enumerate(hashed_vec):
    #         features[f"parent_hash_{i}"] = float(val)
    # else:
    #     for i in range(_HASHTRICK_DIM):
    #         features[f"parent_hash_{i}"] = 0.0

    return features
