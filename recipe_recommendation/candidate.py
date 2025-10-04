import pandas as pd
import numpy as np
from .feature import build_features
import joblib

class Scorer:
    """Rule-based scoring system for recipes"""

    def __init__(self, weights=None):
        # Ingredient category weights
        self.weights = weights or {"main": 1.0, "staple": 0.6, "other": 0.3}

    def score_recipe_rule(self, recipe_parents, user_parents):
        """Compute base score based on ingredient overlaps"""
        score = 0.0
        total = 0.0

        for cat, w in self.weights.items():
            if cat in recipe_parents:
                overlap = len(recipe_parents[cat] & set(user_parents))
                total_cat = len(recipe_parents[cat])
                if total_cat > 0:
                    score += w * overlap / total_cat
                total += w

        return score / total if total > 0 else 0.0


def rule_generate_candidates(df, user_parents, user_profile):
    """Generate candidate recipes with rule-based scoring"""

    scorer = Scorer()

    def score(row):
        # Collect recipe ingredients by parent category
        recipe_parents = {
            "main": row["main_parent"],
            "staple": row["staple_parent"],
            "other": row["other_parent"],
        }
        base_score = scorer.score_recipe_rule(recipe_parents, user_parents)

        # Build feature dict
        recipe_dict = {
            "main": row["main_parent"],
            "staple": row["staple_parent"],
            "other": row["other_parent"],
            "seasoning": row.get("seasoning_parent", set()),
            "matched_main": len(row["main_parent"] & set(user_parents)),
            "matched_staple": len(row["staple_parent"] & set(user_parents)),
            "matched_other": len(row["other_parent"] & set(user_parents)),
            "calories": row.get("calories", 0),
            "protein": row.get("protein", 0),
            "fat": row.get("fat", 0),
            "region": row.get("region", ""),
            "cuisine_attr": row.get("cuisine_attr", []),
            "contains_meat": row.get("contains_meat", False),
            "minutes": row.get("minutes", None),
        }
        features = build_features(recipe_dict, user_profile)

        # --- Rule-based scoring ---

        score = base_score

        # Strict vegetarian filter
        if user_profile.get("vegetarian", False) and features["contains_meat"]:
            return None

        # Calories & macros
        # Calories & macros
        if user_profile.get("low_calorie", False):
            cal_th = user_profile.get("calorie_threshold", None)
            if cal_th is not None and features["calories"] > cal_th:
                score *= 0.7
        if user_profile.get("high_protein", False) and features["protein_ratio"] > 0.25:
            score *= 1.1
        if user_profile.get("low_fat", False) and features["fat_ratio"] > 0.35:
            score *= 0.8

        # Region preference
        if user_profile.get("preferred_regions"):
            region_val = recipe_dict["region"]
            if isinstance(region_val, list):
                if any(r in user_profile["preferred_regions"] for r in region_val):
                    score *= 1.05
            else:
                if region_val in user_profile["preferred_regions"]:
                    score *= 1.05

        # Cuisine preference
        if user_profile.get("preferred_cuisine"):
            if any(c in user_profile["preferred_cuisine"] for c in recipe_dict["cuisine_attr"]):
                score *= 1.2

        return min(score, 1.0)  # normalize

    # Apply scoring
    df = df.copy()
    df["match_score"] = df.apply(score, axis=1)
    df = df.dropna(subset=["match_score"])
    df = df[df["match_score"] > 0]
    df = df.sort_values("match_score", ascending=False)

    # --- Region hard constraint ---
    if user_profile.get("preferred_regions"):
        preferred = df[df["region"].apply(
                        lambda r: any(rr in user_profile["preferred_regions"] for rr in r) if isinstance(r, list) else r in user_profile["preferred_regions"]
                    )]
        if not preferred.empty:
            top_preferred = preferred.head(1)
            df = pd.concat([top_preferred, df]).drop_duplicates(subset=["name"]).reset_index(drop=True)


    return df


def ml_generate_candidates(df, user_parents, user_profile, model_path, topk=5, prefilter_k=5000):
    """
    Two-stage candidate generation:
    1. Rule-based prefiltering
    2. ML-based reranking
    """

    rule_candidates = rule_generate_candidates(
        df,
        user_parents,
        user_profile=user_profile,
    ).head(prefilter_k)

    if rule_candidates.empty:
        print("No candidates found by rule-based filter.")
        return rule_candidates

    # ML-based reranking
    import joblib
    model = joblib.load(model_path)

    feature_rows = []
    for _, row in rule_candidates.iterrows():
        recipe_dict = {
            "main": row["main_parent"],
            "staple": row["staple_parent"],
            "other": row["other_parent"],
            "matched_main": len(row["main_parent"] & set(user_parents)),
            "matched_staple": len(row["staple_parent"] & set(user_parents)),
            "matched_other": len(row["other_parent"] & set(user_parents)),
            "calories": row.get("calories", 0),
            "protein": row.get("protein", 0),
            "fat": row.get("fat", 0),
            "region": row.get("region", ""),
            "cuisine_attr": row.get("cuisine_attr", []),
            "contains_meat": row.get("contains_meat", False),
            "minutes": row.get("minutes", None),
        }
        features = build_features(recipe_dict, user_profile)
        feature_rows.append(features)

    feature_df = pd.DataFrame(feature_rows)
    rule_candidates["ml_score"] = model.predict_proba(feature_df)[:, 1]

    return rule_candidates.sort_values("ml_score", ascending=False).head(topk)


