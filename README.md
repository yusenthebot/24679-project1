# Smart Fridge Ingredient Detection & Recipe Recommendation

This repository combines a Roboflow-powered ingredient detector with a personalised recipe recommender. It exposes a ready-to-run Gradio interface so you can upload a fridge photo, detect the available ingredients, and receive curated recipe suggestions that respect your dietary goals.

## Features

- **Ingredient detection** – Resize incoming photos, run a Roboflow YOLO model, cluster the detections by size, and produce an annotated image plus a `recipe_input.json` payload.
- **Recipe recommendation** – Map detected ingredients to parent categories, score recipes with a hybrid rule/ML ranker, and highlight matches against user preferences.
- **Interactive GUI** – Launch a single `python app.py` command to open the Gradio experience with configurable dietary preferences and three demo fridge images.
- **Demo-friendly** – Toggle "Use cached detections" to explore the workflow without a Roboflow API key, or configure `roboflow_credentials.txt` for live inference.
- **Demo-friendly** – Toggle "Use cached detections" to explore the workflow without a Roboflow API key, or supply your own key for live inference.

## Project layout

```
.
├── app.py                          # Gradio application entry point
├── requirements.txt                # Python dependencies
├── fridge_detect/                  # Ingredient detection utilities & demo assets
└── recipe_recommendation/          # Recommendation engine, datasets, and user data
```

## Getting started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the app**

   ```bash
   python app.py
   ```

3. **Configure Roboflow (optional)**

   - Edit `roboflow_credentials.txt` to add your API key and project slug.
   - Alternatively, set the `ROBOFLOW_API_KEY` environment variable before launching if you prefer not to store it on disk.
3. **Provide a Roboflow key (optional)**

   - Set the environment variable once:

     ```bash
     export ROBOFLOW_API_KEY="your-key"
     ```

   - Or paste it directly into the Gradio sidebar each time you launch the app.

4. **Explore the UI**

   - Click one of the demo fridge photos to load sample detections.
   - Adjust dietary preferences (vegetarian mode, calorie limits, cuisine likes, etc.).
   - Press **Detect & Recommend** to generate an annotated image, ingredient summary, and ranked recipes.

## Data & models

- `fridge_detect/demo/` contains sample fridge photos and cached detection JSON files for offline demos.
- `recipe_recommendation/outputs/recipes_cleaned_strict.csv` is a compact recipe dataset tailored for this project.
- `recipe_recommendation/user-data/user1/` hosts a pre-trained gradient boosted ranker (`ranker.pkl`) and feature store used for scoring recipes.

## Deploying to Hugging Face

The app is Hugging Face Spaces ready. Once dependencies are installed, a standard `python app.py` launch will serve the Gradio interface. Set the `ROBOFLOW_API_KEY` secret within your Space for live detections.

## Development notes

- Detection code lives in `fridge_detect/detect.py` and can be reused programmatically to generate annotated assets and JSON payloads.
- The recommendation pipeline is orchestrated by `recipe_recommendation/main.py` and relies on helper modules in the same package.
- Training utilities (`train_model.py`) are included for completeness should you wish to retrain the ranking model with additional data.

Feel free to adapt the dataset, user profiles, or UI to suit your culinary preferences!
