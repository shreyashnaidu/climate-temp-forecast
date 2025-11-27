from src.evaluation import evaluate_time_series

def run_ablation(models, df_feat):
    results = {}

    # FEATURE GROUPS
    feature_groups = {
        "lags_only": [c for c in df_feat.columns if c.startswith("lag_")],
        "rolling_only": ["roll_5", "roll_11"],
        "trend_only": ["year_idx"],
        "lags_plus_roll": [c for c in df_feat.columns if c.startswith("lag_")] + ["roll_5", "roll_11"],
        "full": [c for c in df_feat.columns if c not in ["year", "anomaly"]],
    }

    for group_name, fg in feature_groups.items():
        X = df_feat[fg].values
        y = df_feat["anomaly"].values

        results[group_name] = {}

        for name, model in models.items():
            res = evaluate_time_series(model, X, y, min_train=50)
            results[group_name][name] = res

    return results
