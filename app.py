"""Gradio application for the smart fridge → recipe recommendation pipeline."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import gradio as gr
import pandas as pd

from fridge_detect.config import load_roboflow_credentials
from fridge_detect.detect import detect_and_generate
from recipe_recommendation.main import load_recipes, recommend_recipes

BASE_DIR = Path(__file__).resolve().parent
DEMO_DIR = BASE_DIR / "fridge_detect" / "demo"
ARTIFACTS_DIR = Path(tempfile.gettempdir()) / "fridge_app_artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

RECIPES_DF = load_recipes()
REGION_OPTIONS = sorted({region for row in RECIPES_DF["region"] for region in row})
CUISINE_OPTIONS = sorted({cuisine for row in RECIPES_DF["cuisine_attr"] for cuisine in row})
ROBOFLOW_API_KEY, ROBOFLOW_PROJECT = load_roboflow_credentials()
ROBOFLOW_PROJECT = ROBOFLOW_PROJECT or "nutrition-object-detection"
ROBOFLOW_STATUS = "found" if ROBOFLOW_API_KEY else "missing"
DEFAULT_USER_ID = "user1"


def _load_cached_detection(image_path: Path) -> Optional[Dict[str, Any]]:
    cached_json = image_path.with_suffix(".json")
    if cached_json.exists():
        with cached_json.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def build_user_profile(
    vegetarian: bool,
    vegetarian_type: str,
    low_calorie: bool,
    calorie_threshold: int,
    high_protein: bool,
    low_fat: bool,
    preferred_regions: List[str],
    preferred_cuisine: List[str],
    preferred_main: str,
    disliked_main: str,
    cooking_time_max: int,
) -> Dict[str, Any]:
    """Assemble the user profile dictionary expected by the recommender."""

    preferred_main_list = [item.strip().lower() for item in preferred_main.split(",") if item.strip()]
    disliked_main_list = [item.strip().lower() for item in disliked_main.split(",") if item.strip()]

    profile = {
        "vegetarian": vegetarian,
        "diet": {"vegetarian_type": vegetarian_type if vegetarian else "omnivore"},
        "low_calorie": low_calorie,
        "calorie_threshold": calorie_threshold,
        "high_protein": high_protein,
        "low_fat": low_fat,
        "preferred_regions": preferred_regions,
        "preferred_cuisine": preferred_cuisine,
        "other_preferences": {
            "preferred_main": preferred_main_list,
            "disliked_main": disliked_main_list,
            "cooking_time_max": cooking_time_max if cooking_time_max else None,
        },
    }

    return profile


def format_detection_markdown(payload: Dict[str, Any]) -> str:
    if not payload.get("ingredients"):
        return "No ingredients detected yet."

    lines = ["### Detected Ingredients", "| Ingredient | Size | Confidence |", "|---|---|---|"]
    for ing in payload.get("ingredients", []):
        lines.append(
            f"| {ing.get('name', 'unknown').replace('_', ' ')} | {ing.get('quantity', '?')} | {ing.get('confidence', 0):.2f} |"
        )

    high = payload.get("high_confidence_ingredients", [])
    low = payload.get("low_confidence_ingredients", [])
    if high:
        lines.append("\n**High confidence hits:** " + ", ".join(high))
    if low:
        lines.append("\n**Low confidence / uncertain:** " + ", ".join(low))
    return "\n".join(lines)


def format_recommendation_markdown(high_conf: List[str], low_conf: List[str], user_parents: List[str]) -> str:
    lines = ["### Recommendation context"]
    if user_parents:
        lines.append("- Parent categories used for matching: " + ", ".join(sorted(set(user_parents))))
    if high_conf:
        lines.append("- High confidence matches: " + ", ".join(high_conf))
    if low_conf:
        lines.append("- Low confidence / ignored: " + ", ".join(low_conf))
    return "\n".join(lines)


def format_candidates_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    table = df.copy()
    for col in ["main_parent", "staple_parent", "other_parent", "cuisine_attr", "region"]:
        if col in table.columns:
            table[col] = table[col].apply(lambda x: ", ".join(sorted(x)) if isinstance(x, (set, list)) else x)
    if "ml_score" in table.columns:
        table["ml_score"] = table["ml_score"].apply(lambda x: round(float(x), 3))
    if "match_score" in table.columns:
        table["match_score"] = table["match_score"].apply(lambda x: round(float(x), 3))
    selected_cols = [
        col
        for col in [
            "name",
            "ml_score",
            "match_score",
            "calories",
            "protein",
            "fat",
            "minutes",
            "region",
            "cuisine_attr",
            "main_parent",
            "staple_parent",
            "other_parent",
        ]
        if col in table.columns
    ]
    return table[selected_cols]


def run_pipeline(
    image: str,
    api_key: str,
    project_name: str,
    version: int,
    use_cached_detection: bool,
    vegetarian: bool,
    vegetarian_type: str,
    low_calorie: bool,
    calorie_threshold: int,
    high_protein: bool,
    low_fat: bool,
    preferred_regions: List[str],
    preferred_cuisine: List[str],
    preferred_main: str,
    disliked_main: str,
    cooking_time_max: int,
    topk: int,
) -> Tuple[str, str, str, pd.DataFrame]:
    if not image:
        return None, "Please upload or select an image of your fridge.", "", pd.DataFrame()

    image_path = Path(image)
    detection_payload = None
    annotated_image = str(image_path)

    if use_cached_detection:
        cached = _load_cached_detection(image_path)
        if cached:
            detection_payload = cached
        else:
            use_cached_detection = False  # fall back to live detection

    if detection_payload is None:
        api_key, project_name = load_roboflow_credentials()
        project_name = project_name or ROBOFLOW_PROJECT

        if not api_key:
            return (
                str(image_path),
                "Roboflow API key missing. Add it to roboflow_credentials.txt or enable cached detection for demo images.",
            )
        if not api_key:
            return (
                str(image_path),
                "Roboflow API key missing. Provide a key or enable cached detection for demo images.",
                "",
                pd.DataFrame(),
            )
        unique = uuid4().hex
        output_json = ARTIFACTS_DIR / f"{unique}_recipe.json"
        output_image = ARTIFACTS_DIR / f"{unique}_annotated.jpg"
        try:
            detection_payload, annotated_image = detect_and_generate(
                str(image_path),
                api_key=api_key,
                project_name=project_name,
                version=int(version),
                output_json=str(output_json),
                output_image=str(output_image),
            )
        except Exception as exc:
            return (
                str(image_path),
                f"Detection failed: {exc}",
                "",
                pd.DataFrame(),
            )

    detection_md = format_detection_markdown(detection_payload)

    calorie_threshold = int(calorie_threshold) if calorie_threshold is not None else 600
    cooking_limit = int(cooking_time_max) if cooking_time_max else 0
    topk = int(topk) if topk else 5
    user_profile = build_user_profile(
        vegetarian,
        vegetarian_type,
        low_calorie,
        calorie_threshold,
        high_protein,
        low_fat,
        preferred_regions or [],
        preferred_cuisine or [],
        preferred_main,
        disliked_main,
        cooking_limit,
    )

    try:
        candidates, user_parents, high_conf, low_conf = recommend_recipes(
            detection_payload,
            user_profile,
            user_id=DEFAULT_USER_ID,
            recipes_df=RECIPES_DF,
            topk=topk,
        )
    except Exception as exc:
        return (
            annotated_image,
            detection_md,
            f"Recommendation failed: {exc}",
            pd.DataFrame(),
        )

    recommendation_md = format_recommendation_markdown(high_conf, low_conf, user_parents)
    candidates_table = format_candidates_table(candidates)

    return annotated_image, detection_md, recommendation_md, candidates_table


def build_interface() -> gr.Blocks:
    with gr.Blocks(css=".gradio-container {max-width: 1200px}") as demo:
        gr.Markdown(
            """
            # Smart Fridge → Recipe Recommender
            Upload a photo of your fridge, detect what's inside with a Roboflow model, and instantly receive personalised recipe ideas.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Fridge photo", type="filepath")
                example_images = [
                    [str(DEMO_DIR / "t1.jpg")],
                    [str(DEMO_DIR / "t2.jpg")],
                    [str(DEMO_DIR / "t3.jpg")],
                ]
                gr.Examples(
                    examples=example_images,
                    inputs=image_input,
                    label="Demo fridge photos",
                    examples_per_page=3,
                )
                use_cached = gr.Checkbox(
                    label="Use cached detections for demo images (no API key needed)",
                    value=True,
                )
                gr.Markdown(
                    f"Using Roboflow credentials from `roboflow_credentials.txt` (project: `{ROBOFLOW_PROJECT}`, key {ROBOFLOW_STATUS})."
                )
                api_key_input = gr.Textbox(
                    label="Roboflow API key",
                    type="password",
                    placeholder="Paste your key or set ROBOFLOW_API_KEY environment variable",
                    value=os.getenv("ROBOFLOW_API_KEY", ""),
                )
                project_input = gr.Textbox(
                    label="Roboflow project name",
                    value="nutrition-object-detection",
                )
                version_input = gr.Number(label="Model version", value=1, precision=0)
                topk_input = gr.Slider(
                    label="Number of recipes",
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=5,
                )

            with gr.Column(scale=1):
                gr.Markdown("### Dietary preferences")
                vegetarian_toggle = gr.Checkbox(label="Vegetarian", value=False)
                vegetarian_type = gr.Dropdown(
                    label="Vegetarian type",
                    choices=["flexible_vegetarian", "vegetarian", "vegan", "omnivore"],
                    value="flexible_vegetarian",
                )
                low_calorie_toggle = gr.Checkbox(label="Low calorie focus", value=False)
                calorie_threshold_slider = gr.Slider(
                    label="Max calories per recipe", minimum=200, maximum=1000, value=600, step=50
                )
                high_protein_toggle = gr.Checkbox(label="Boost high-protein meals", value=False)
                low_fat_toggle = gr.Checkbox(label="Limit high-fat recipes", value=False)
                region_select = gr.CheckboxGroup(
                    label="Preferred regions",
                    choices=REGION_OPTIONS,
                    value=[REGION_OPTIONS[0]] if REGION_OPTIONS else [],
                )
                cuisine_select = gr.CheckboxGroup(
                    label="Preferred cuisine styles",
                    choices=CUISINE_OPTIONS,
                    value=[],
                )
                preferred_main_input = gr.Textbox(
                    label="Preferred main ingredients (comma separated)",
                    placeholder="e.g. chicken, tofu",
                )
                disliked_main_input = gr.Textbox(
                    label="Disliked main ingredients (comma separated)",
                    placeholder="e.g. beef",
                )
                cooking_time_input = gr.Number(label="Max cooking time (minutes)", value=30, precision=0)

        run_button = gr.Button("Detect & Recommend", variant="primary")

        annotated_output = gr.Image(label="Annotated detection")
        detection_output = gr.Markdown()
        recommendation_output = gr.Markdown()
        candidates_output = gr.Dataframe(label="Top recipe ideas", wrap=True)

        run_button.click(
            fn=run_pipeline,
            inputs=[
                image_input,
                api_key_input,
                project_input,
                version_input,
                use_cached,
                vegetarian_toggle,
                vegetarian_type,
                low_calorie_toggle,
                calorie_threshold_slider,
                high_protein_toggle,
                low_fat_toggle,
                region_select,
                cuisine_select,
                preferred_main_input,
                disliked_main_input,
                cooking_time_input,
                topk_input,
            ],
            outputs=[
                annotated_output,
                detection_output,
                recommendation_output,
                candidates_output,
            ],
        )

    return demo


def main() -> None:
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
