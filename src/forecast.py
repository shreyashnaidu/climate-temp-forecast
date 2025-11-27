import pandas as pd
from src.feature_engineering import create_features

def forecast_future_years(model, df, years_ahead=10):
    """
    Iteratively forecast 'years_ahead' years into the future
    using the given trained model and the same feature pipeline.
    """
    df_hist = df.copy()

    last_year = int(df_hist["year"].iloc[-1])
    future_records = []

    for step in range(1, years_ahead + 1):
        target_year = last_year + step

        # Build features from history
        feat_df = create_features(df_hist)
        latest_row = feat_df.iloc[-1]  # last available feature row

        X_latest = latest_row.drop(["anomaly", "year"]).values.reshape(1, -1)
        y_pred = model.predict(X_latest)[0]

        # Append prediction as new "observed" anomaly for next step
        df_hist = pd.concat(
            [
                df_hist,
                pd.DataFrame({"year": [target_year], "anomaly": [y_pred]})
            ],
            ignore_index=True
        )

        future_records.append({"year": target_year, "forecast": y_pred})

    return pd.DataFrame(future_records)
