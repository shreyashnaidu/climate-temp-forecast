import pandas as pd

def create_features(df: pd.DataFrame):
    df = df.copy()

    # Lag features
    for lag in range(1, 6):
        df[f"lag_{lag}"] = df["anomaly"].shift(lag)

    # Rolling windows (shifted to avoid leakage)
    df["roll_5"]  = df["anomaly"].shift(1).rolling(5).mean()
    df["roll_11"] = df["anomaly"].shift(1).rolling(11).mean()

    # Trend
    df["year_idx"] = range(len(df))

    return df.dropna().reset_index(drop=True)
