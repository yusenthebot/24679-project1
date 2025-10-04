import json
import pandas as pd
from parser import parse_list, parse_set, get_parent
from candidate import ml_generate_candidates
from highlight import print_candidates
import os
from train_model import cold_start_classifier, train_model_classifier

RECIPES_FILE = r"D:\docs\24679\project1\outputs\recipes_cleaned_strict.csv"
df = pd.read_csv(RECIPES_FILE)

# Parse ingredient-related columns
for col in ["staple", "main", "seasoning", "other"]:
    if col in df.columns:
        df[col] = df[col].apply(parse_list)

for col in ["staple_parent", "main_parent", "seasoning_parent", "other_parent",
            "cuisine_attr", "region"]:
    if col in df.columns:
        df[col] = df[col].apply(parse_set)

# Load JSON input
def load_cv_results(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    user_parents = []
    high_conf = []
    low_conf = []
    for ing in data.get("ingredients", []):
        name = ing["name"].strip().lower().replace("_", " ")
        parent = get_parent(name)
        if ing.get("confidence", 0) >= 0.8 and parent:
            user_parents.append(parent)
            high_conf.append((name, parent))
        else:
            low_conf.append(name)
    if high_conf:
        print("High-confidence ingredients mapped to parents:")
        for child, parent in high_conf:
            print(f"   - {child} → parent: {parent}")
    if low_conf:
        print(f"Ignored (low confidence or no parent found): {low_conf}")
    return list(set(user_parents))


# Terminal interface
def main(user_id="user_1"):
    PROFILE_FILE = os.path.join("user_data", user_id, "user_profile.json")
    FEATURES_FILE = os.path.join("user_data", user_id, "user_features.csv")
    MODEL_FILE    = os.path.join("user_data", user_id, "ranker.pkl")

    user_parents = load_cv_results(r"D:\docs\24679\project1\recipe_input.json")

    # Step 1: check profile
    if not os.path.exists(PROFILE_FILE):
        raise FileNotFoundError(f"Missing profile: {PROFILE_FILE}. Please create one first.")
    else:
        with open(PROFILE_FILE, "r", encoding="utf-8") as f:
            user_profile = json.load(f)

    # Step 2: check features + model existing
    if not os.path.exists(FEATURES_FILE):
        print("No features found, running cold-start...")
        cold_start_classifier(user_id, recipes_file=RECIPES_FILE)
        train_model_classifier(user_id)

    elif not os.path.exists(MODEL_FILE):
        print("Features exist but no model found, training model...")
        train_model_classifier(user_id)

    # Step 3: interface
    print(f"\nLaunching interface for {user_id}...\n")
    topk = 5

    candidates = ml_generate_candidates(
    df,
    user_parents,
    user_profile=user_profile,
    model_path=MODEL_FILE,
    topk=5
    )

    print(f"\nFound {len(candidates)} candidate recipes:\n")
    print_candidates(candidates, user_parents, topk=topk)

if __name__ == "__main__":
    main("user_1")

