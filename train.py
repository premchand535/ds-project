import os
import math
import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm

from src.config import cfg
from src.data_fetcher import download
from src.datasets import HouseDataset
from src.model import FusionModel
from src.gradcam import GradCAM


# -------------------- Metrics --------------------
def rmse(pred, true):
    return math.sqrt(mean_squared_error(true, pred))


# -------------------- Training --------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for img, tab, y in tqdm(loader, desc="Training", leave=False):
        img = img.to(cfg.device)
        tab = tab.to(cfg.device)
        y = y.to(cfg.device)

        optimizer.zero_grad()
        pred = model(img, tab)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)




# -------------------- Validation --------------------
@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    ys, ps = [], []
    total_loss = 0.0

    for img, tab, y in loader:
        img = img.to(cfg.device)
        tab = tab.to(cfg.device)
        y = y.to(cfg.device)

        pred = model(img, tab)
        loss = criterion(pred, y)

        total_loss += loss.item() * len(y)
        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)

    # 🔴 Convert back to original price
    ys_real = np.expm1(ys)
    ps_real = np.expm1(ps)

    return (
        total_loss / len(loader.dataset),
        rmse(ps_real, ys_real),
        r2_score(ys_real, ps_real)
    )



# -------------------- Grad-CAM --------------------
def run_gradcam(model, dataset):
    model.eval()  # SAFETY
    os.makedirs(os.path.join(cfg.output_dir, "gradcam"), exist_ok=True)

    cam_generator = GradCAM(model)

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for i in range(min(cfg.grad_cam_samples, len(dataset))):
        img, tab, y = dataset[i]
        cam = cam_generator(img, tab)

        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * std + mean) * 255
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        heatmap = np.uint8(cam * 255)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        cv2.imwrite(
            os.path.join(cfg.output_dir, "gradcam", f"sample_{i}_price_{int(y)}.png"),
            overlay
        )


# -------------------- Main --------------------
def main():
    print("Device:", cfg.device)
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)

    train_df = pd.read_excel(cfg.train_xlsx)
    test_df  = pd.read_excel(cfg.test_xlsx)

    # LOG TRANSFORM TARGET
    train_df["price"] = np.log1p(train_df["price"])


    # Download satellite images
    img_paths = download(pd.concat([train_df, test_df], axis=0))

    # Keep rows with available images
    train_df = train_df[train_df["id"].isin(img_paths)]
    test_df  = test_df[test_df["id"].isin(img_paths)]

    # Scale tabular features
    scaler = StandardScaler()
    scaler.fit(train_df[cfg.tab_feats])

    # Train / Validation split
    tr_df, val_df = train_test_split(
        train_df,
        test_size=cfg.val_split,
        random_state=cfg.seed
    )

    tr_ds = HouseDataset(tr_df, img_paths, scaler, train=True)
    val_ds = HouseDataset(val_df, img_paths, scaler, train=True)
    te_ds  = HouseDataset(test_df, img_paths, scaler, train=False)

    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    te_loader  = DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False)

    # Model
    model = FusionModel(tab_in=len(cfg.tab_feats)).to(cfg.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    criterion = nn.SmoothL1Loss()

    best_rmse = float("inf")

    # Training loop
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, tr_loader, optimizer, criterion)
        val_loss, val_rmse, val_r2 = evaluate(model, val_loader, criterion)

        print(
            f"Epoch [{epoch+1}/{cfg.epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"RMSE: {val_rmse:.2f} | "
            f"R2: {val_r2:.4f}"
        )

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(
                {
                    "model": model.state_dict(),
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_,
                },
                os.path.join(cfg.model_dir, "best_model.pt")
            )

    # Load best model
    ckpt = torch.load(
    os.path.join(cfg.model_dir, "best_model.pt"),
    map_location=cfg.device,
    weights_only=False
)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Predict test data
    preds, ids = [], []
    with torch.no_grad():
        for img, tab, pid in te_loader:
            img = img.to(cfg.device)
            tab = tab.to(cfg.device)

            pred = model(img, tab).cpu().numpy()
            pred = np.expm1(pred)   # reverse log transform
            preds.extend(pred.squeeze().tolist())
            ids.extend(pid.tolist())

    submission = pd.DataFrame({
        "id": ids,
        "predicted_price": preds
    })

    submission.to_csv(
        os.path.join(cfg.output_dir, "submission.csv"),
        index=False
    )

    print("Saved outputs/submission.csv")

    # save submission
    # save_submission(model, test_ds)

    # Grad-CAM
    # run_gradcam(model, val_ds)


# -------------------- Entry --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    main()

