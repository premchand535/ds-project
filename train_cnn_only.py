import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data_fetcher import download
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from src.config import cfg
from src.datasets import HouseDataset
from src.cnn_only_model import CNNOnlyModel


def rmse(pred, true):
    return math.sqrt(mean_squared_error(true, pred))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []

    for img, _, y in loader:   # ignore tabular data
        img = img.to(cfg.device)
        y = y.to(cfg.device)

        pred = model(img)
        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())

    if len(ys) == 0:
     return float("nan"), float("nan")

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    return rmse(ps, ys), r2_score(ys, ps)



def main():
    print("Device:", cfg.device)

    df = pd.read_excel(cfg.train_xlsx)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # download / load image paths (same as fusion)
    img_paths = download(train_df)

    # keep only rows with images
    train_df = train_df[train_df["id"].isin(img_paths)]
    val_df   = val_df[val_df["id"].isin(img_paths)]

    train_ds = HouseDataset(train_df, img_paths, None, train=True)
    val_ds   = HouseDataset(val_df, img_paths, None, train=True)



    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = CNNOnlyModel().to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(10):
        model.train()
        for img, _, y in train_loader:
            img, y = img.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
            loss = criterion(model(img), y)
            loss.backward()
            optimizer.step()

        val_rmse, val_r2 = evaluate(model, val_loader)
        print(f"Epoch [{epoch+1}/10] | RMSE: {val_rmse:.2f} | R2: {val_r2:.4f}")


if __name__ == "__main__":
    main()
