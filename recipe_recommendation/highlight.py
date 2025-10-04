import pandas as pd

def print_candidates(candidates, user_parents, topk=10):
    shown = 0
    for _, row in candidates.head(topk).iterrows():
        print(f"{row['name']} (score {row['match_score']:.2f})")

        # Region & Cuisine
        region = row.get("region", None)
        if isinstance(region, set):
            if len(region) > 0:
                print(f"  region: {next(iter(region))}")
        elif isinstance(region, str) and region.strip():
            print(f"  region: {region}")


        cuisine = row.get("cuisine_attr", None)
        if isinstance(cuisine, list) and len(cuisine) > 0:
            print(f"  cuisine: {', '.join(cuisine)}")

        # Nutrition
        print(f"  calories: {row.get('calories', 'N/A')}")

        def mark_list(lst):
            return [("✅ " + ing) if ing in user_parents else ("❌ " + ing) for ing in lst]

        print(f"  staple:    {mark_list(row.get('staple_parent', []))}")
        print(f"  main:      {mark_list(row.get('main_parent', []))}")
        print(f"  seasoning: {row.get('seasoning_parent', [])}")  # no check/cross
        print(f"  other:     {mark_list(row.get('other_parent', []))}")
        print("-" * 40)

        shown += 1



