import numpy as np

def bootstrap_forecast_intervals(model_class, X, y, feature_names,
                                 n_bootstrap=200,
                                 alpha=0.05):

    preds = []

    for i in range(n_bootstrap):
        # sample with replacement
        idx = np.random.choice(len(X), len(X), replace=True)
        X_boot = X[idx]
        y_boot = y[idx]

        model = model_class()
        model.fit(X_boot, y_boot)

        preds.append(model.predict(X))

    preds = np.array(preds)

    lower = np.quantile(preds, alpha/2, axis=0)
    median = np.quantile(preds, 0.5, axis=0)
    upper = np.quantile(preds, 1 - alpha/2, axis=0)

    return lower, median, upper
