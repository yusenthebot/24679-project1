# -*- coding: utf-8 -*-
"""
Detect ingredients using a Roboflow model with preprocessing:
- Resize images to 640x640 if needed.
- Perform detection.
- Classify object sizes via K-Means.
- Generate JSON and annotated image outputs.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from roboflow import Roboflow
from sklearn.cluster import KMeans

from .config import load_roboflow_credentials

def compute_area_ratios(predictions, img_shape):
    """Compute area ratio (bbox area / image area) for each detection."""
    img_area = float(img_shape[0] * img_shape[1])
    ratios = []
    for pred in predictions:
        area = pred["width"] * pred["height"]
        ratios.append(area / img_area)
    return np.array(ratios).reshape(-1, 1)

def cluster_sizes(area_ratios):
    """Cluster area ratios into two groups using K-Means and return size labels."""
    kmeans = KMeans(n_clusters=2, init="k-means++", random_state=0)
    labels = kmeans.fit_predict(area_ratios)
    centroids = kmeans.cluster_centers_.flatten()
    large_cluster = np.argmax(centroids)
    return ["large" if lbl == large_cluster else "small" for lbl in labels]

def detect_and_generate(
    image_path: str,
    api_key: Optional[str] = None,
    project_name: Optional[str] = None,
    project_name: str = "nutrition-object-detection",
    version: int = 1,
    conf_threshold: float = 0.4,
    overlap_threshold: float = 0.3,
    conf_split: float = 0.7,
    output_json: Optional[str] = None,
    output_image: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    """Run Roboflow detection and create assets for the recipe recommender.

    Parameters
    ----------
    image_path:
        Path to the input image.
    api_key:
        Roboflow API key. If ``None``, the function will look for the
        ``ROBOFLOW_API_KEY`` environment variable.
    project_name:
        Roboflow project slug.
    version:
        Version number of the deployed model.
    conf_threshold:
        Minimum confidence threshold (0–1) for keeping detections.
    overlap_threshold:
        Non-maximum suppression overlap threshold (0–1).
    conf_split:
        Confidence threshold used to split ingredients into high/low
        confidence buckets for the recommender.
    output_json:
        Optional path where the generated JSON payload should be saved. If
        ``None`` the file will be saved next to ``image_path`` with the name
        ``recipe_input.json``.
    output_image:
        Optional path for the annotated image. If ``None`` the annotated image
        will be created next to ``image_path`` with the name
        ``annotated_image.jpg``.

    Returns
    -------
    Tuple[Dict[str, Any], str]
        A tuple containing the generated recipe JSON payload and the path to
        the annotated image on disk.
    """

    credentials_api, credentials_project = load_roboflow_credentials()

    if project_name is None:
        project_name = credentials_project or "nutrition-object-detection"

    api_key = api_key or os.getenv("ROBOFLOW_API_KEY") or credentials_api
    if not api_key:
        raise ValueError(
            "A Roboflow API key is required. Provide it as a function argument, "
            "set the ROBOFLOW_API_KEY environment variable, or add it to "
            "roboflow_credentials.txt."
    api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError(
            "A Roboflow API key is required. Provide it as a function argument "
            "or set the ROBOFLOW_API_KEY environment variable."
        )

    image_path = str(image_path)
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    height, width = original_img.shape[:2]

    # Preprocess: resize to 640x640 if needed, and save to a temp file
    if height != 640 or width != 640:
        resized_img = cv2.resize(original_img, (640, 640))
        # create temporary file via mkstemp; close fd to avoid locking
        fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        cv2.imwrite(tmp_path, resized_img)
        detection_path = tmp_path
        img_for_annotation = resized_img
    else:
        detection_path = image_path
        img_for_annotation = original_img

    # Initialize Roboflow model
    rf = Roboflow(api_key=api_key)
    model = rf.workspace().project(project_name).version(version).model

    # Run prediction
    response = model.predict(
        detection_path,
        confidence=int(conf_threshold * 100),
        overlap=int(overlap_threshold * 100)
    ).json()
    predictions = response["predictions"]

    # Classify sizes using K-Means
    area_ratios = compute_area_ratios(predictions, img_for_annotation.shape)
    size_labels = cluster_sizes(area_ratios)

    # Build JSON structure
    ingredients = []
    high_conf = []
    low_conf = []
    for pred, size_label in zip(predictions, size_labels):
        name = pred["class"]
        conf = pred["confidence"]
        ingredients.append({
            "name": name,
            "quantity": size_label,
            "confidence": round(conf, 2)
        })
        if conf >= conf_split:
            high_conf.append(name)
        else:
            low_conf.append(name)

    recipe_json = {
        "ingredients": ingredients,
        "high_confidence_ingredients": high_conf,
        "low_confidence_ingredients": low_conf
    }

    input_path = Path(image_path).resolve()
    if output_json is None:
        output_json = str(input_path.with_name("recipe_input.json"))
    if output_image is None:
        output_image = str(input_path.with_name("annotated_image.jpg"))

    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(recipe_json, jf, indent=4)

    # Annotate image with bounding boxes and confidence labels
    detections = sv.Detections.from_inference(response)
    label_annotator = sv.LabelAnnotator()
    box_annotator = sv.BoxAnnotator()

    labels_for_annotation = [
        f"{pred['class']} ({pred['confidence']:.2f})" for pred in predictions
    ]

    annotated_img = box_annotator.annotate(
        scene=img_for_annotation.copy(),
        detections=detections
    )
    annotated_img = label_annotator.annotate(
        scene=annotated_img,
        detections=detections,
        labels=labels_for_annotation
    )

    cv2.imwrite(output_image, annotated_img)

    # Clean up temporary file
    if height != 640 or width != 640:
        try:
            os.remove(tmp_path)
        except PermissionError:
            # If still locked on Windows, delay deletion or log a warning
            pass

    return recipe_json, output_image


def describe_ingredients(payload: Dict[str, Any]) -> List[str]:
    """Return formatted ingredient strings for UI previews."""

    lines: List[str] = []
    for ing in payload.get("ingredients", []):
        name = ing.get("name", "unknown")
        quantity = ing.get("quantity", "?")
        confidence = ing.get("confidence", 0)
        lines.append(f"{name} ({quantity}, conf={confidence:.2f})")
    return lines


if __name__ == "__main__":
    example_image = Path(__file__).parent / "demo" / "t2.jpg"
    try:
        result_json, annotated_path = detect_and_generate(str(example_image))
        print(json.dumps(result_json, indent=4))
        print(f"Annotated image saved to: {annotated_path}")
    except Exception as exc:
        print(f"Detection failed: {exc}")
