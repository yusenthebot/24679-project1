import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from xgboost import XGBClassifier, callback

from recipe_recommendation.candidate import (
    ml_generate_candidates,
    rule_generate_candidates,
)
from recipe_recommendation.feature import build_features
from recipe_recommendation.parser import get_parent, parse_list, parse_set

BASE_DIR = Path(__file__).resolve().parent
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from xgboost import XGBClassifier, callback

from .candidate import ml_generate_candidates, rule_generate_candidates
from .feature import build_features
from .parser import get_parent, parse_list, parse_set
from candidate import ml_generate_candidates, rule_generate_candidates
from feature import build_features
from parser import get_parent, parse_list, parse_set

BASE_DIR = Path(__file__).resolve().parent


def sample_user_parents(parent_file,
                        user_profile=None,
                        prev_inventory=None,
                        min_items=3, max_items=10,
                        keep_ratio=0.6,
                        reset_interval=None,
                        round_idx=0):
    """
    Preference-aware sampler for a user's pantry (parent categories).

    - Likes get higher probability (weight=3)
    - Disliked/forbidden parents are excluded
    - A fraction (keep_ratio) of the previous pantry is kept (sticky pantry)
    - Every reset_interval rounds, the pantry is completely refreshed
    """
    import json, random

    # Load all parents
    with open(parent_file, "r", encoding="utf-8") as f:
        parent_map = json.load(f)
    all_parents = list(parent_map.keys())

    # Reset condition
    force_reset = reset_interval and (round_idx % reset_interval == 0)

    # User preferences
    oprefs = (user_profile or {}).get("other_preferences", {})
    liked    = set(oprefs.get("preferred_main", []))
    disliked = set(oprefs.get("disliked_main", []))
    forbidden = set((user_profile or {}).get("forbidden_parents", [])) | disliked

    # Build candidate pool with weights
    pool, weights = [], []
    for p in all_parents:
        if p in forbidden:
            continue
        w = 3.0 if p in liked else 1.0
        pool.append(p)
        weights.append(w)

    if not pool:
        pool = all_parents
        weights = [1.0] * len(pool)

    inventory = set()

    # Keep part of last round's pantry if not forcing reset
    if prev_inventory and not force_reset:
        prev_list = list(prev_inventory)
        random.shuffle(prev_list)
        keep_k = max(0, int(len(prev_list) * keep_ratio))
        inventory |= set(prev_list[:keep_k])

    # Sample the remaining items with weighted random choice
    k = random.randint(min_items, max_items)
    remaining_k = max(0, k - len(inventory))

    local_pool, local_w = pool[:], weights[:]
    for _ in range(min(remaining_k, len(local_pool))):
        idx = random.choices(range(len(local_pool)), weights=local_w, k=1)[0]
        inventory.add(local_pool[idx])
        del local_pool[idx]
        del local_w[idx]

    return list(inventory)


# Generate initial training data for a user if no features exist
def cold_start_classifier(user_id="user_1", recipes_file=None, n_rounds=2000, topk=5,
               batch_size=5000, switch_interval=100):

    PROFILE_FILE  = os.path.join("user_data", user_id, "user_profile.json")
    FEATURES_FILE = os.path.join("user_data", user_id, "user_features.csv")

    if os.path.exists(FEATURES_FILE):
        print(f"Cold-start features already exist for {user_id}")
        return

    # Load user profile
    with open(PROFILE_FILE, "r", encoding="utf-8") as f:
        user_profile = json.load(f)

    # Load recipes dataset
    df = pd.read_csv(recipes_file)

    # Parse ingredient-related columns
    for col in ["staple", "main", "seasoning", "other"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_list)
    for col in ["staple_parent", "main_parent", "seasoning_parent", "other_parent",
                "cuisine_attr", "region"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_set)

    # Split into chunks
    n_chunks = (len(df) // batch_size) + 1
    chunks = np.array_split(df, n_chunks)

    samples = []
    prev_inventory = None

    for i in tqdm(range(n_rounds), desc="Generating cold-start data"):
        # Rotate recipe chunks
        chunk_id = (i // switch_interval) % n_chunks
        df_chunk = chunks[chunk_id].copy()

        # Sample pantry with preferences and sticky logic
        user_parents = sample_user_parents(
            str(BASE_DIR / "outputs" / "parent_category_map.json"),
            user_profile=user_profile,
            prev_inventory=prev_inventory,
            min_items=3, max_items=10,
            keep_ratio=0.6,
            reset_interval=20   # force reset every 20 rounds
        )
        user_parents = sample_user_parents(
            str(BASE_DIR / "outputs" / "parent_category_map.json"),
            user_profile=user_profile,
            prev_inventory=prev_inventory,
            min_items=3, max_items=10,
            keep_ratio=0.6,
            reset_interval=20,   # force reset every 20 rounds
            round_idx=i
        )
        prev_inventory = user_parents

        # Compute overlaps with user pantry
        df_chunk["matched_main"] = df_chunk["main_parent"].apply(lambda s: len(s & set(user_parents)))
        df_chunk["matched_staple"] = df_chunk["staple_parent"].apply(lambda s: len(s & set(user_parents)))
        df_chunk["matched_other"] = df_chunk["other_parent"].apply(lambda s: len(s & set(user_parents)))

        # Generate candidate recipes
        candidates = rule_generate_candidates(
            df_chunk,
            user_parents=user_parents,
            user_profile=user_profile,
        ).head(topk)

        if candidates.empty:
            continue

        # Step 1: if any recipe perfectly covers main+other, force it as positive
        perfect_cover = []
        for j, row in candidates.iterrows():
            if (row["matched_main"] == len(row["main_parent"]) and
                row["matched_other"] == len(row["other_parent"])):
                perfect_cover.append((j, row))

        if perfect_cover:
            chosen_id = random.choice(perfect_cover)[0]
        else:
            chosen_id = None

        # Step 2: assign labels
        for idx, (j, row) in enumerate(candidates.iterrows()):
            recipe_dict = {
                "main": row["main_parent"],
                "staple": row["staple_parent"],
                "other": row["other_parent"],
                "seasoning": row.get("seasoning_parent", set()),  # seasoning included only for features
                "matched_main": row["matched_main"],
                "matched_staple": row["matched_staple"],
                "matched_other": row["matched_other"],
                "calories": row.get("calories", 0),
                "protein": row.get("protein", 0),
                "fat": row.get("fat", 0),
                "region": row.get("region", ""),
                "cuisine_attr": row.get("cuisine_attr", []),
                "contains_meat": row.get("contains_meat", False),
            }
            features = build_features(recipe_dict, user_profile)

            label = 0
            if chosen_id is not None and j == chosen_id:
                label = 1
            else:
                if   idx == 0: label = 1          # top-1 always positive
                elif idx == 1 and random.random() < 0.5: label = 1
                elif idx == 2 and random.random() < 0.3: label = 1

            features["label"] = label
            samples.append(features)

    # Save cold-start features
    df_out = pd.DataFrame(samples)
    df_out.to_csv(FEATURES_FILE, index=False)
    print(f"Cold-start features saved to {FEATURES_FILE}")


def train_model_classifier(user_id="user_1", max_samples=5000, save_model=True, recipes_file = None):

    FEATURES_FILE = os.path.join("user_data", user_id, "user_features.csv")
    LOGS_FILE     = os.path.join("user_data", user_id, "logs.csv")
    MODEL_FILE    = os.path.join("user_data", user_id, "ranker.pkl")

    # Ensure cold-start exists
    if not os.path.exists(FEATURES_FILE):
        cold_start_classifier(user_id, recipes_file)

    df_features = pd.read_csv(FEATURES_FILE)

    # Merge logs if available
    if os.path.exists(LOGS_FILE):
        df_logs = pd.read_csv(LOGS_FILE)
        print(f"Loaded {len(df_logs)} log samples")
        df = pd.concat([df_features, df_logs], ignore_index=True)
    else:
        print("No logs found, training only on cold start data")
        df = df_features

    # Sliding window
    if len(df) > max_samples:
        df = df.tail(max_samples)

    print(f"Total training samples: {len(df)}")

    # Split
    feature_cols = [c for c in df.columns if c != "label" and np.issubdtype(df[c].dtype, np.number)]
    X = df[feature_cols]
    y = df["label"]

    print(f"Using {len(feature_cols)} features (showing first 10): {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=20
    )

    # Evaluate
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    print(classification_report(y_val, y_pred))
    print("ROC-AUC:", roc_auc_score(y_val, y_prob))

    # Save
    if save_model:
        joblib.dump(model, MODEL_FILE)
        print(f"Model saved to {MODEL_FILE}")

    return model

