from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def get_models():
    return {
        "ols": LinearRegression(),
        "rf": RandomForestRegressor(
            n_estimators=300,
            random_state=42
        ),
        "gbr": GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=3,
            random_state=42
        )
    }
