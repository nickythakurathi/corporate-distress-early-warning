import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
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
calib_df = df[df["obs_year"].between(2015, 2018)].copy()
test_df = df[df["obs_year"].between(2019, 2024)].copy()

X_train, y_train = train_df[features], train_df[TARGET]
X_calib, y_calib = calib_df[features], calib_df[TARGET]
X_test, y_test = test_df[features], test_df[TARGET]

n_negative = int((y_train == 0).sum())
n_positive = int((y_train == 1).sum())
scale_pos_weight = n_negative / max(n_positive, 1)

xgb = XGBClassifier(
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
xgb.fit(X_train, y_train)

calibrated_xgb = CalibratedClassifierCV(xgb, method="isotonic", cv="prefit")
calibrated_xgb.fit(X_calib, y_calib)

logit_baseline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=5000, random_state=42))
])
logit_baseline.fit(X_train, y_train)

results = pd.DataFrame([
    {"model": "logit_baseline", "split": "test", **evaluate_predictions(y_test, logit_baseline.predict_proba(X_test)[:, 1])},
    {"model": "xgb_raw", "split": "test", **evaluate_predictions(y_test, xgb.predict_proba(X_test)[:, 1])},
    {"model": "xgb_calibrated", "split": "test", **evaluate_predictions(y_test, calibrated_xgb.predict_proba(X_test)[:, 1])},
])

scored = test_df.copy()
scored["xgb_prob"] = xgb.predict_proba(X_test)[:, 1]
scored["xgb_calibrated_prob"] = calibrated_xgb.predict_proba(X_test)[:, 1]

outdir = ensure_results_dir()
results.to_csv(outdir / "xgboost_calibrated_test_metrics.csv", index=False)
scored.to_csv(outdir / "xgboost_calibrated_test_predictions.csv", index=False)

print(results)
