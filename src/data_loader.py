import pandas as pd

def load_gistemp(path: str = "data/annual.csv"):
    df = pd.read_csv(path)

    # Standardize column names
    df = df.rename(columns=lambda c: c.strip().lower())

    # Rename mean → anomaly
    df = df.rename(columns={"mean": "anomaly"})

    # Remove duplicate years by averaging anomalies
    df = df.groupby("year", as_index=False)["anomaly"].mean()

    # Sort by ascending year
    df = df.sort_values("year").reset_index(drop=True)

    return df
