import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

class cfg:
    # Paths
    train_xlsx = os.path.join(BASE_DIR, "data", "train.xlsx")
    test_xlsx  = os.path.join(BASE_DIR, "data", "test.xlsx")

    image_dir  = os.path.join(BASE_DIR, "data", "images")
    output_dir = os.path.join(BASE_DIR, "outputs")
    model_dir  = os.path.join(BASE_DIR, "models")

    # Tabular features (MUST exist in Excel)
    tab_feats = [
        "bedrooms",
        "bathrooms",
        "sqft_living"
    ]

    target_col = "price"
    target = target_col  

    # Training
    val_split = 0.1
    seed = 42
    batch_size = 20 
    epochs = 20   
    lr = 1e-4   
    weight_decay = 1e-4 
    zoom = 18
    img_size = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
