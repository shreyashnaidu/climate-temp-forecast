import matplotlib.pyplot as plt

def plot_anomaly_series(df):
    plt.figure(figsize=(10, 4))
    plt.plot(df["year"], df["anomaly"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Temperature Anomaly (°C)")
    plt.title("Global Mean Temperature Anomalies")
    plt.tight_layout()
    plt.show()

def plot_rolling_means(df, window_short=5, window_long=11):
    s = df.copy()
    s["roll_short"] = s["anomaly"].rolling(window_short).mean()
    s["roll_long"] = s["anomaly"].rolling(window_long).mean()

    plt.figure(figsize=(10, 4))
    plt.plot(s["year"], s["anomaly"], alpha=0.4, label="Annual anomalies")
    plt.plot(s["year"], s["roll_short"], label=f"{window_short}-year rolling mean")
    plt.plot(s["year"], s["roll_long"], label=f"{window_long}-year rolling mean")
    plt.xlabel("Year")
    plt.ylabel("Temperature Anomaly (°C)")
    plt.title("Anomalies with Rolling Means")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_pred_vs_actual(years, actuals, preds, model_name):
    plt.figure(figsize=(8, 4))
    plt.plot(years, actuals, marker="o", label="Actual")
    plt.plot(years, preds, marker="x", label="Predicted")
    plt.xlabel("Year")
    plt.ylabel("Temperature Anomaly (°C)")
    plt.title(f"Actual vs Predicted – {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()
