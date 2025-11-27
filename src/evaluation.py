import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def expanding_window_split(X, y, min_train=50):
    n = len(X)
    for end in range(min_train, n):
        train_idx = np.arange(0, end)
        test_idx = np.array([end])
        yield train_idx, test_idx

def evaluate_time_series(model, X, y, min_train=50):
    preds, actuals = [], []
    test_indices = []

    for train_idx, test_idx in expanding_window_split(X, y, min_train=min_train):
        model.fit(X[train_idx], y[train_idx])
        p = model.predict(X[test_idx])[0]

        preds.append(p)
        actuals.append(y[test_idx][0])
        test_indices.append(test_idx[0])

    rmse = mean_squared_error(actuals, preds) ** 0.5
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "preds": np.array(preds),
        "actuals": np.array(actuals),
        "test_indices": np.array(test_indices),
    }
