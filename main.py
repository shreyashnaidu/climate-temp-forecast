############################################################
# FINAL MAIN FILE FOR GLOBAL TEMP ANOMALY FORECAST PROJECT
# (ALL FIGURE GENERATION COMMENTED OUT + METRICS SAVING)
############################################################

from src.data_loader import load_gistemp
from src.feature_engineering import create_features
from src.models import get_models
from src.evaluation import evaluate_time_series

# Diagnostics
from src.diagnostics import (
    compute_residuals,
    print_residual_stats,
    # plot_residuals_vs_year,
    # plot_residual_hist,
    # plot_residual_acf,
)

# Interpretability
from src.interpretability import (
    compute_permutation_importance,
    # plot_permutation_importance,
    # plot_pdp,
)

# Forecasting
from src.forecast import forecast_future_years

# Calibration
from src.calibration import bootstrap_forecast_intervals

# Ablation
from src.ablation import run_ablation

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

############################################################
# 1. LOAD RAW DATA
############################################################

df = load_gistemp("data/annual.csv")

print("=== RAW DATA CHECK ===")
print(df.head(10))
print(df.tail(10))
print("Shape:", df.shape)

############################################################
# 2. FEATURE ENGINEERING
############################################################

df_feat = create_features(df)

print("\n=== FEATURE DATA CHECK ===")
print(df_feat.head(5))
print(df_feat.tail(5))
print("Shape:", df_feat.shape)

feature_names = [c for c in df_feat.columns if c not in ["year", "anomaly"]]
X = df_feat[feature_names].values
y = df_feat["anomaly"].values

############################################################
# 3. TRAIN MODELS (Expanding Window Evaluation)
############################################################

models = get_models()
print("\n=== MODEL RESULTS (Expanding Window) ===")
results = {}

for name, model in models.items():
    res = evaluate_time_series(model, X, y, min_train=50)
    results[name] = res
    print(f"\n{name.upper()}")
    print(f"RMSE: {res['rmse']:.4f}")
    print(f"MAE:  {res['mae']:.4f}")
    print(f"R²:   {res['r2']:.4f}")

############################################################
# 4. RESIDUAL DIAGNOSTICS (OLS)
############################################################

ols_res = results["ols"]
ols_residuals = compute_residuals(ols_res["actuals"], ols_res["preds"])
test_years = df_feat["year"].values[ols_res["test_indices"]]

print_residual_stats(ols_residuals, "OLS")

# COMMENTED OUT:
# plot_residuals_vs_year(test_years, ols_residuals, "OLS")
# plot_residual_hist(ols_residuals, "OLS")
# plot_residual_acf(ols_residuals, "OLS")

############################################################
# 5. INTERPRETABILITY (RF model)
############################################################

rf_model = get_models()["rf"]
rf_model.fit(X, y)

importance = compute_permutation_importance(rf_model, X, y, feature_names)

# COMMENTED OUT:
# plot_permutation_importance(importance, "Random Forest")
# if "lag_1" in feature_names:
#     plot_pdp(rf_model, X, feature_names, "lag_1")
# if "roll_11" in feature_names:
#     plot_pdp(rf_model, X, feature_names, "roll_11")

############################################################
# 6. FORECAST FUTURE YEARS (RF)
############################################################

future_df = forecast_future_years(rf_model, df, years_ahead=10)
print("\n=== RF FORECAST – NEXT 10 YEARS ===")
print(future_df)

############################################################
# 7. CALIBRATION / PREDICTION INTERVALS (BOOTSTRAP RF)
############################################################

lower, mid, upper = bootstrap_forecast_intervals(
    lambda: RandomForestRegressor(n_estimators=100, random_state=42),
    X, y, feature_names,
    n_bootstrap=100
)

print("\nBootstrap interval example (last point):")
print("Lower:", lower[-1], "Mid:", mid[-1], "Upper:", upper[-1])

############################################################
# 8. ABLATION STUDY
############################################################

print("\n=== RUNNING ABLATION STUDY ===")
ablation_results = run_ablation(models, df_feat)

for group, model_res in ablation_results.items():
    print(f"\n--- {group.upper()} ---")
    for model_name, metrics in model_res.items():
        print(f"{model_name}: RMSE={metrics['rmse']:.3f}, R²={metrics['r2']:.3f}")

############################################################
# 9. SAVE ALL METRICS TO results/metrics/
############################################################

# MAIN RESULTS (OLS / RF / GBR)
main_rows = []
for name, res in results.items():
    main_rows.append({
        "model": name,
        "RMSE": res["rmse"],
        "MAE": res["mae"],
        "R2": res["r2"]
    })
pd.DataFrame(main_rows).to_csv("results/metrics/main_results.csv", index=False)

# ABLATION RESULTS
abl_rows = []
for group, model_res in ablation_results.items():
    for model_name, metrics in model_res.items():
        abl_rows.append({
            "feature_group": group,
            "model": model_name,
            "RMSE": metrics["rmse"],
            "R2": metrics["r2"]
        })
pd.DataFrame(abl_rows).to_csv("results/metrics/ablation_results.csv", index=False)

# CALIBRATION INTERVALS
cal_df = pd.DataFrame({
    "index": range(len(mid)),
    "lower": lower,
    "median": mid,
    "upper": upper
})
cal_df.to_csv("results/metrics/calibration_intervals.csv", index=False)

# FUTURE FORECASTS
future_df.to_csv("results/metrics/forecast_next_10_years.csv", index=False)

############################################################

print("\nAll coding tasks and metric exports completed successfully (no figures generated).")
############################################################
