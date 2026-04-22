import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
from common import ensure_results_dir

RESULTS_DIR = ensure_results_dir()
pred_file = RESULTS_DIR / "xgboost_calibrated_test_predictions.csv"

if not pred_file.exists():
    raise FileNotFoundError("Run src/models/xgboost_calibrated.py first.")

test_scored = pd.read_csv(pred_file, low_memory=False)
test_scored["obs_date"] = pd.to_datetime(test_scored["obs_date"], errors="coerce")
test_scored["obs_year"] = test_scored["obs_date"].dt.year

rank_col = "xgb_calibrated_prob"
test_2024 = test_scored[test_scored["obs_year"] == 2024].copy()
test_2024 = test_2024.sort_values(rank_col, ascending=False)

test_2024["risk_percentile"] = test_2024[rank_col].rank(pct=True)
test_2024["risk_bucket"] = pd.cut(
    test_2024["risk_percentile"],
    bins=[0, 0.50, 0.80, 0.95, 1.00],
    labels=["Low", "Moderate", "High", "Severe"]
)

keep_cols = [c for c in [
    "permno", "ticker", "comnam", "obs_date", "obs_year", "distress_next_4q",
    rank_col, "risk_bucket", "bm", "npm", "roa", "roe", "debt_capital",
    "de_ratio", "quick_ratio", "vol", "excess_ret"
] if c in test_2024.columns]

watchlist = test_2024[keep_cols].copy()
watchlist.to_csv(RESULTS_DIR / "watchlist_2024.csv", index=False)

lift_table = pd.DataFrame({
    "group": ["Top 25", "Top 50", "Top 100", "Top 200", "All 2024"],
    "hit_rate": [
        watchlist.head(25)["distress_next_4q"].mean(),
        watchlist.head(50)["distress_next_4q"].mean(),
        watchlist.head(100)["distress_next_4q"].mean(),
        watchlist.head(200)["distress_next_4q"].mean(),
        watchlist["distress_next_4q"].mean(),
    ],
})
lift_table["lift_vs_base"] = lift_table["hit_rate"] / max(lift_table.iloc[-1]["hit_rate"], 1e-9)
lift_table.to_csv(RESULTS_DIR / "watchlist_2024_lift_table.csv", index=False)

print(watchlist.head(25))
print(lift_table)
