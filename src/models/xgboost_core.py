import pandas as pd
from xgboost import XGBClassifier
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from common import load_model_data, TARGET, evaluate_predictions, ensure_results_dir

df = load_model_data()

features = [
    "bm", "npm", "gpm", "roa", "roe", "gprof",
    "invt_act", "rect_act", "curr_debt",
    "debt_capital", "de_ratio", "quick_ratio", "curr_ratio",
    "at_turn", "ret", "excess_ret", "vol", "shrout"
]
features = [c for c in features if c in df.columns]

train_df = df[df["obs_year"].between(2000, 2014)].copy()
valid_df = df[df["obs_year"].between(2015, 2018)].copy()
test_df = df[df["obs_year"].between(2019, 2024)].copy()

X_train, y_train = train_df[features], train_df[TARGET]
X_valid, y_valid = valid_df[features], valid_df[TARGET]
X_test, y_test = test_df[features], test_df[TARGET]

n_negative = int((y_train == 0).sum())
n_positive = int((y_train == 1).sum())
scale_pos_weight = n_negative / max(n_positive, 1)

model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    scale_pos_weight=scale_pos_weight,
)

model.fit(X_train, y_train)

results = pd.DataFrame([
    {"split": "train", **evaluate_predictions(y_train, model.predict_proba(X_train)[:, 1])},
    {"split": "valid", **evaluate_predictions(y_valid, model.predict_proba(X_valid)[:, 1])},
    {"split": "test", **evaluate_predictions(y_test, model.predict_proba(X_test)[:, 1])},
])

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

outdir = ensure_results_dir()
results.to_csv(outdir / "xgboost_core_metrics.csv", index=False)
importance.to_csv(outdir / "xgboost_core_feature_importance.csv", index=False)
pd.DataFrame({"xgb_prob": model.predict_proba(X_test)[:, 1]}, index=X_test.index).to_csv(
    outdir / "xgboost_core_test_predictions.csv"
)

print(results)
print(importance.head(15))
