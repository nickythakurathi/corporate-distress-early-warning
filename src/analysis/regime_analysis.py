import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import MODEL_READY_FILE, RESULTS_DIR

if not MODEL_READY_FILE.exists():
    raise FileNotFoundError("Run src/data/build_credit_panel.py first.")

df = pd.read_csv(MODEL_READY_FILE, low_memory=False)
df["obs_date"] = pd.to_datetime(df["obs_date"], errors="coerce")
df["obs_year"] = df["obs_date"].dt.year

yearly_components = (
    df.groupby("obs_year")
    .agg(
        distress_rate=("distress_next_4q", "mean"),
        distress_count=("distress_next_4q", "sum"),
        bad_dlret_rate=("bad_dlret_next_4q", "mean"),
        exit_rate=("exit_next_4q", "mean"),
    )
    .reset_index()
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
yearly_components.to_csv(RESULTS_DIR / "yearly_distress_regime_summary.csv", index=False)
print(yearly_components)
