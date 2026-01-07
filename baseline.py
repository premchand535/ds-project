import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from src.config import cfg
import math

def run_tabular_baseline():
    # load data
    df = pd.read_excel(cfg.train_xlsx)
    X = df[cfg.tab_feats].values
    y = df[cfg.target].values

    # scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # train / validation split
    Xtr, Xva, ytr, yva = train_test_split(X,y, test_size=cfg.val_split, random_state=cfg.seed)

    # model
    model = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
    )

    # train
    model.fit(Xtr, ytr)

    # predict
    preds = model.predict(Xva)

    # metrics
    rmse = math.sqrt(mean_squared_error(yva, preds))
    r2 = r2_score(yva, preds)
    print(f"Tabular-only XGB RMSE: {rmse:.2f} | R2: {r2:.3f}")
    return model, scaler

if __name__ == "__main__":
    print(run_tabular_baseline())
