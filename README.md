# Multimodal House Price Regression (Tabular + Satellite Imagery)

## Overview
- Predict house prices by fusing tabular features with satellite imagery.
- Programmatically fetch satellite images from ESRI World Imagery using property latitude and longitude via ArcGIS REST services.
- Perform EDA + geospatial analysis (GeoPandas) to inspect spatial patterns and visual factors.
- Build CNN-based visual embeddings, fuse with tabular MLP (late fusion) in PyTorch, compare against tabular-only baselines (XGBoost/RandomForest/LightGBM optional).
- Explainability with Grad-CAM overlays on satellite tiles.

## Stack
- Data: Pandas, NumPy, GeoPandas
- DL: PyTorch, Torchvision (ResNet18/50), optional TensorFlow/Keras equivalent
- ML: scikit-learn, XGBoost
- Imaging: ESRI World Imagery (ArcGIS REST API), PIL, requests
- Viz: Matplotlib, Seaborn
- Explainability: Grad-CAM (torch hooks), optional SHAP for tabular

## Data
- Train: `train(1).xlsx` (columns incl. `price`, `lat`, `long`)
- Test: `test2.xlsx` (same features minus `price`)
- Place under `data/` or set paths in `src/config.py`.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# set API key
# ESRI World Imagery is accessed via public ArcGIS REST services
# No API key is required

# 1) Fetch images
python -m src.data_fetcher --config src/config.py

# 2) (Optional) Run preprocessing/EDA notebook
jupyter notebook notebooks/preprocessing.ipynb

# 3) Train multimodal model
python -m src.train --config src/config.py

# 4) Outputs
# - models/best_model.pt
# - outputs/submission.csv (id,predicted_price)
# - outputs/gradcam/*.png

## Fusion Choices
- Default: Late fusion (ResNet encoder + MLP on tabular, concatenated head).
- Try earlier fusion (concat tabular to CNN penultimate) or gated fusion (attention over modalities) in 'src/model.py'.

## Explainability
-Grad-CAM overlays stored in 'outputs/gradcam/'.
-For tabular feature importance, enable SHAP in 'src/train.py' (see TODO)

## Deliverables
- 'outputs/submission.csv' for predictions.
- Code in 'src/' + notebooks in 'notebooks/'.
- Report: export notebook to PDF covering Overview, EDA, Insights, Architecture diagram, Results (Tabular vs. Tabular+Image).