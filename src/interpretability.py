import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

def compute_permutation_importance(model, X, y, feature_names, n_repeats=30):
    """Returns a sorted permutation importance result."""
    r = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=42
    )

    idx = np.argsort(r.importances_mean)[::-1]

    ordered_features = [feature_names[i] for i in idx]
    ordered_importances_mean = r.importances_mean[idx]
    ordered_importances_std = r.importances_std[idx]

    return {
        "features": ordered_features,
        "mean": ordered_importances_mean,
        "std": ordered_importances_std,
    }

def plot_permutation_importance(imp_result, model_name):
    features = imp_result["features"]
    mean = imp_result["mean"]
    std = imp_result["std"]

    plt.figure(figsize=(8, 5))
    y_pos = np.arange(len(features))

    plt.barh(y_pos, mean, xerr=std)
    plt.yticks(y_pos, features)
    plt.xlabel("Mean decrease in score")
    plt.title(f"Permutation Importance – {model_name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def plot_pdp(model, X, feature_names, feature):
    """Plot 1D partial dependence for a single feature."""
    feature_index = feature_names.index(feature)
    disp = PartialDependenceDisplay.from_estimator(
        model,
        X,
        [feature_index],
        feature_names=feature_names
    )
    disp.figure_.suptitle(f"Partial Dependence – {feature}")
    plt.tight_layout()
    plt.show()
