"""
expected_level.py
=================
Models the natural / expected groundwater level for observation wells.
Audited & Fair Comparison:
  Model 1 (Fair Baseline): Autoregressive Ridge Regression (AR-12 + 12-week Rain + Sin/Cos Seasonality).
                           Given the exact same 12-week history as the LSTM for an honest, apples-to-apples comparison.
  Model 2 (PyTorch LSTM) : 1-layer LSTM on past 12 weeks of [level, rain, sin_week, cos_week] -> next week level.

Strict temporal split: Last 4 years = held-out test set.
Scalers fitted ONLY on training data.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class GroundwaterLSTM(nn.Module):
    """Explainable 1-layer LSTM for weekly groundwater level forecasting."""
    def __init__(self, input_dim=4, hidden_dim=32):
        super(GroundwaterLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def prepare_features(df):
    """Extracts calendar and lagged features."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Week of year (1-53)
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    
    # Rainfall
    rain = df['rain_mm'].fillna(0.0) if 'rain_mm' in df.columns else pd.Series(0.0, index=df.index)
    df['rain_1w'] = rain
    df['rain_4w'] = rain.rolling(4, min_periods=1).sum()
    df['rain_12w'] = rain.rolling(12, min_periods=1).sum()
    
    # Smooth annual cycle
    df['sin_week'] = np.sin(2 * np.pi * df['week_of_year'] / 52.0)
    df['cos_week'] = np.cos(2 * np.pi * df['week_of_year'] / 52.0)
    
    # 12-week lagged levels and lagged rain for Fair Baseline
    for lag in range(1, 13):
        df[f'lag_level_{lag}'] = df['depth_m'].shift(lag)
        df[f'lag_rain_{lag}'] = rain.shift(lag).fillna(0.0)
        
    return df


def fit_fair_baseline(train_df, test_df):
    """
    Model 1 (Fair Baseline):
      Autoregressive Ridge regression receiving the EXACT same inputs as LSTM:
      - 12 lagged level values [lag_level_1 ... lag_level_12]
      - 12 lagged rain values [lag_rain_1 ... lag_rain_12]
      - Calendar seasonality [sin_week, cos_week]
    """
    lag_cols = [f'lag_level_{i}' for i in range(1, 13)] + [f'lag_rain_{i}' for i in range(1, 13)] + ['sin_week', 'cos_week']
    
    scaler_x = StandardScaler()
    X_train = scaler_x.fit_transform(train_df[lag_cols])
    X_test = scaler_x.transform(test_df[lag_cols])
    
    y_train = train_df['depth_m'].values
    y_test = test_df['depth_m'].values
    
    ridge = Ridge(alpha=10.0, random_state=SEED)
    ridge.fit(X_train, y_train)
    
    pred_train = ridge.predict(X_train)
    pred_test = ridge.predict(X_test)
    
    return pred_train, pred_test, ridge, scaler_x, lag_cols


def create_lstm_sequences(df, seq_len=12, scaler_level=None, scaler_rain=None, is_train=True):
    """Builds sliding window sequences of length 12 for LSTM."""
    features = np.zeros((len(df), 4))
    
    level_vals = df['depth_m'].values.reshape(-1, 1)
    rain_vals = df['rain_1w'].values.reshape(-1, 1)
    
    if is_train:
        scaler_level = StandardScaler()
        scaler_rain = StandardScaler()
        norm_level = scaler_level.fit_transform(level_vals)
        norm_rain = scaler_rain.fit_transform(rain_vals)
    else:
        norm_level = scaler_level.transform(level_vals)
        norm_rain = scaler_rain.transform(rain_vals)
        
    features[:, 0] = norm_level.squeeze()
    features[:, 1] = norm_rain.squeeze()
    features[:, 2] = df['sin_week'].values
    features[:, 3] = df['cos_week'].values
    
    X, y = [], []
    for i in range(seq_len, len(df)):
        X.append(features[i - seq_len:i])
        y.append(features[i, 0])
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler_level, scaler_rain


def fit_lstm_model(train_df, test_df, seq_len=12, epochs=30, lr=0.01):
    """
    Model 2 (LSTM):
      Trained on sequence of 12 steps x 4 features -> next week level.
    """
    X_train, y_train, scaler_level, scaler_rain = create_lstm_sequences(
        train_df, seq_len=seq_len, is_train=True
    )
    
    combined_test_df = pd.concat([train_df.iloc[-seq_len:], test_df], ignore_index=True)
    X_test, y_test, _, _ = create_lstm_sequences(
        combined_test_df, seq_len=seq_len,
        scaler_level=scaler_level, scaler_rain=scaler_rain, is_train=False
    )
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    torch.manual_seed(SEED)
    model = GroundwaterLSTM(input_dim=4, hidden_dim=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        pred_train_norm = model(torch.from_numpy(X_train)).numpy()
        pred_test_norm = model(torch.from_numpy(X_test)).numpy()
        
    pred_train_m = scaler_level.inverse_transform(pred_train_norm).squeeze()
    pred_test_m = scaler_level.inverse_transform(pred_test_norm).squeeze()
    
    return pred_train_m, pred_test_m, model, scaler_level, scaler_rain


def evaluate_well(df_well, well_id, test_years=4, seq_len=12):
    """Evaluates Fair Baseline vs LSTM with aligned 12-week lag context."""
    df_feat = prepare_features(df_well)
    
    max_date = df_feat['date'].max()
    split_date = max_date - pd.DateOffset(years=test_years)
    
    # Train / test split
    train_df = df_feat[df_feat['date'] < split_date].copy().reset_index(drop=True)
    test_df = df_feat[df_feat['date'] >= split_date].copy().reset_index(drop=True)
    
    if len(train_df) < 100 or len(test_df) < 50:
        return None
        
    # We evaluate predictions starting at seq_len so both models have complete 12-week lag inputs
    y_test = test_df['depth_m'].values
    y_train_eval = train_df['depth_m'].iloc[seq_len:].values
    
    # 1. Model 1: Fair Baseline (AR-Ridge + Rain + Season)
    lag_cols = [f'lag_level_{i}' for i in range(1, 13)] + [f'lag_rain_{i}' for i in range(1, 13)] + ['sin_week', 'cos_week']
    
    scaler_ridge = StandardScaler()
    X_train_ridge = scaler_ridge.fit_transform(train_df.iloc[seq_len:][lag_cols])
    
    # For test, combine last seq_len of train to test so lag_level features at start of test are exact
    combined_test_df = pd.concat([train_df.iloc[-seq_len:], test_df], ignore_index=True)
    combined_test_feat = prepare_features(combined_test_df).iloc[seq_len:].reset_index(drop=True)
    X_test_ridge = scaler_ridge.transform(combined_test_feat[lag_cols])
    
    ridge = Ridge(alpha=10.0, random_state=SEED)
    ridge.fit(X_train_ridge, y_train_eval)
    b_train = ridge.predict(X_train_ridge)
    b_test = ridge.predict(X_test_ridge)
    
    mae_baseline = mean_absolute_error(y_test, b_test)
    rmse_baseline = np.sqrt(mean_squared_error(y_test, b_test))
    
    # 2. Model 2: LSTM
    l_train, l_test, lstm_mod, scaler_lvl, scaler_rn = fit_lstm_model(train_df, test_df, seq_len=seq_len)
    mae_lstm = mean_absolute_error(y_test, l_test)
    rmse_lstm = np.sqrt(mean_squared_error(y_test, l_test))
    
    # Decision rule: Keep LSTM only if it clearly outperforms baseline by > 5% RMSE
    lstm_clear_winner = (rmse_lstm < 0.95 * rmse_baseline)
    selected_model_name = "LSTM" if lstm_clear_winner else "Fair Baseline (AR-12 Ridge)"
    
    selected_test_pred = l_test if lstm_clear_winner else b_test
    selected_train_pred = l_train if lstm_clear_winner else b_train
    
    # Train gaps SD for z_gap scaling
    train_gaps = y_train_eval - selected_train_pred
    std_train_gap = float(np.std(train_gaps))
    if std_train_gap < 1e-4:
        std_train_gap = 1e-4
        
    well_result = {
        'well_id': well_id,
        'train_samples': len(y_train_eval),
        'test_samples': len(y_test),
        'test_date_range': [str(test_df['date'].min().date()), str(test_df['date'].max().date())],
        'baseline_mae_m': float(mae_baseline),
        'baseline_rmse_m': float(rmse_baseline),
        'lstm_mae_m': float(mae_lstm),
        'lstm_rmse_m': float(rmse_lstm),
        'selected_model': selected_model_name,
        'std_train_gap_m': std_train_gap
    }
    
    # Combine dataframes starting at seq_len for complete evaluation consistency
    eval_train_df = train_df.iloc[seq_len:].copy().reset_index(drop=True)
    full_df = pd.concat([eval_train_df, test_df], ignore_index=True)
    full_expected = np.concatenate([selected_train_pred, selected_test_pred])
    full_df['expected_depth_m'] = full_expected
    full_df['gap_m'] = full_df['depth_m'] - full_df['expected_depth_m']
    full_df['is_test'] = full_df['date'] >= split_date
    full_df['std_train_gap_m'] = std_train_gap
    
    return well_result, full_df


def run_expected_level_pipeline(dataset_type='gems'):
    """Runs fair expected-level modeling across all candidate wells."""
    prefix = "california" if dataset_type in ["cal", "california"] else "gems"
    csv_path = f"data/processed/{'california' if prefix == 'california' else 'gems_ger'}_harmonized_15wells.csv"
    
    df_all = pd.read_csv(csv_path)
    wells = df_all['well_id'].unique()
    
    print("=" * 80)
    print(f"RUNNING AUDITED PART A: EXPECTED LEVEL MODELING [{prefix.upper()}] ({len(wells)} wells)")
    print("Fair Comparison: Both models receive identical 12-week lag history + rain + season")
    print("=" * 80)
    print(f"{'Well ID':<16} | {'Base MAE':<9} | {'Base RMSE':<9} | {'LSTM MAE':<9} | {'LSTM RMSE':<9} | Selected Model")
    print("-" * 80)
    
    results = []
    full_dfs = []
    
    for wid in wells:
        df_well = df_all[df_all['well_id'] == wid]
        eval_out = evaluate_well(df_well, wid)
        if eval_out is None:
            continue
        res, fdf = eval_out
        results.append(res)
        full_dfs.append(fdf)
        
        print(f"{res['well_id']:<16} | {res['baseline_mae_m']:<9.4f} | {res['baseline_rmse_m']:<9.4f} | "
              f"{res['lstm_mae_m']:<9.4f} | {res['lstm_rmse_m']:<9.4f} | {res['selected_model']}")
              
    res_df = pd.DataFrame(results)
    print("-" * 80)
    print(f"Average Fair Baseline Test MAE: {res_df['baseline_mae_m'].mean():.4f} m, RMSE: {res_df['baseline_rmse_m'].mean():.4f} m")
    print(f"Average LSTM Test MAE:          {res_df['lstm_mae_m'].mean():.4f} m, RMSE: {res_df['lstm_rmse_m'].mean():.4f} m")
    
    lstm_wins = (res_df['selected_model'] == 'LSTM').sum()
    base_wins = len(res_df) - lstm_wins
    print(f"\nSelection: Baseline chosen for {base_wins}/{len(res_df)} wells, LSTM for {lstm_wins}/{len(res_df)} wells.")
    if base_wins >= lstm_wins:
        print(">> VERDICT: Fair Autoregressive Baseline is preferred or equivalent for most wells.")
        print("   Linear storage inertia explains short-term dynamics without requiring a neural network.")
    else:
        print(">> VERDICT: LSTM captures non-linear reservoir discharge/recharge thresholds.")
        
    os.makedirs("models", exist_ok=True)
    with open(f"models/expected_level_{prefix}_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
        
    combined_expected = pd.concat(full_dfs, ignore_index=True)
    combined_expected.to_csv(f"data/processed/{prefix}_with_expected.csv", index=False)
    print(f"Saved predictions and gaps to data/processed/{prefix}_with_expected.csv\n")
    return res_df, combined_expected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gems", "cal", "both"], default="both")
    args = parser.parse_args()
    
    if args.dataset in ["gems", "both"]:
        run_expected_level_pipeline("gems")
    if args.dataset in ["cal", "both"]:
        run_expected_level_pipeline("cal")
