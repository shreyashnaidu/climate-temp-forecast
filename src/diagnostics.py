import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.stattools import durbin_watson

def compute_residuals(actuals, preds):
    return np.array(actuals) - np.array(preds)

def print_residual_stats(residuals, model_name):
    print(f"\n=== Residual Diagnostics – {model_name} ===")
    print(f"Mean residual: {np.mean(residuals):.4f}")
    print(f"Std residual:  {np.std(residuals):.4f}")
    print(f"Durbin–Watson: {durbin_watson(residuals):.4f}")

def plot_residuals_vs_year(years, residuals, model_name):
    plt.figure(figsize=(8, 4))
    plt.axhline(0, linestyle="--")
    plt.plot(years, residuals, marker="o")
    plt.xlabel("Year")
    plt.ylabel("Residual (y - ŷ)")
    plt.title(f"Residuals vs Year – {model_name}")
    plt.tight_layout()
    plt.show()

def plot_residual_hist(residuals, model_name):
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=15)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title(f"Residual Histogram – {model_name}")
    plt.tight_layout()
    plt.show()

def plot_residual_acf(residuals, model_name, lags=20):
    plt.figure(figsize=(6, 4))
    plot_acf(residuals, lags=lags)
    plt.title(f"Residual ACF – {model_name}")
    plt.tight_layout()
    plt.show()
