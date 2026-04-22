import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from common import (
    load_model_data, add_log_shrout, get_time_splits, get_winsor_bounds,
    apply_winsor_bounds, TARGET, evaluate_predictions, ensure_results_dir
)

df = add_log_shrout(load_model_data())

features = [
    "bm", "npm", "gpm", "roa", "roe", "gprof",
    "debt_capital", "de_ratio", "quick_ratio",
    "inv_turn", "at_turn", "divyield", "vol", "excess_ret", "log_shrout"
]
features = [c for c in features if c in df.columns]

train_df, valid_df, test_df = get_time_splits(df)
bounds = get_winsor_bounds(train_df, features)
train_df = apply_winsor_bounds(train_df, bounds)
valid_df = apply_winsor_bounds(valid_df, bounds)
test_df = apply_winsor_bounds(test_df, bounds)

X_train, y_train = train_df[features], train_df[TARGET]
X_valid, y_valid = valid_df[features], valid_df[TARGET]
X_test, y_test = test_df[features], test_df[TARGET]

imputer = SimpleImputer(strategy="median")
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=features, index=X_train.index)
X_valid_imp = pd.DataFrame(imputer.transform(X_valid), columns=features, index=X_valid.index)
X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=features, index=X_test.index)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_leaf=50,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train_imp, y_train)

results = pd.DataFrame([
    {"split": "train", **evaluate_predictions(y_train, model.predict_proba(X_train_imp)[:, 1])},
    {"split": "valid", **evaluate_predictions(y_valid, model.predict_proba(X_valid_imp)[:, 1])},
    {"split": "test", **evaluate_predictions(y_test, model.predict_proba(X_test_imp)[:, 1])},
])

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

outdir = ensure_results_dir()
results.to_csv(outdir / "random_forest_metrics.csv", index=False)
importance.to_csv(outdir / "random_forest_feature_importance.csv", index=False)

print(results)
print(importance.head(15))
