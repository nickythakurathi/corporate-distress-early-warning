import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
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

preprocessor = ColumnTransformer(
    transformers=[("num_bins", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("bins", KBinsDiscretizer(n_bins=5, encode="onehot-dense", strategy="quantile"))
    ]), features)],
    remainder="drop"
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(C=1.0, penalty="l2", max_iter=5000, random_state=42))
])

model.fit(X_train, y_train)

results = pd.DataFrame([
    {"split": "train", **evaluate_predictions(y_train, model.predict_proba(X_train)[:, 1])},
    {"split": "valid", **evaluate_predictions(y_valid, model.predict_proba(X_valid)[:, 1])},
    {"split": "test", **evaluate_predictions(y_test, model.predict_proba(X_test)[:, 1])},
])

outdir = ensure_results_dir()
results.to_csv(outdir / "scorecard_logit_metrics.csv", index=False)
print(results)
