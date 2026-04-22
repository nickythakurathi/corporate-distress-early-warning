import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import MODEL_READY_FILE, RESULTS_DIR

if not MODEL_READY_FILE.exists():
    raise FileNotFoundError("Run src/data/build_credit_panel.py first.")

df = pd.read_csv(MODEL_READY_FILE, low_memory=False)
df["obs_date"] = pd.to_datetime(df["obs_date"], errors="coerce")
df = df.sort_values(["permno", "obs_date"]).copy()

vars_to_track = [
    "roa", "roe", "npm", "gpm", "de_ratio",
    "quick_ratio", "curr_ratio", "ret", "excess_ret", "vol"
]
vars_to_track = [c for c in vars_to_track if c in df.columns]

rows = []
distressed = df[df["distress_next_4q"] == 1][["permno", "obs_date"]].copy()
distressed = distressed.rename(columns={"obs_date": "distress_date"})

for _, row in distressed.iterrows():
    firm = df[df["permno"] == row["permno"]].copy()
    firm["quarters_before_distress"] = ((row["distress_date"] - firm["obs_date"]).dt.days / 91.25).round().astype("Int64")
    firm = firm[firm["quarters_before_distress"].between(1, 6, inclusive="both")]
    for q in range(1, 7):
        chunk = firm[firm["quarters_before_distress"] == q]
        if len(chunk) == 0:
            continue
        summary = {"quarters_before_distress": q}
        for var in vars_to_track:
            summary[var] = chunk[var].mean()
        rows.append(summary)

trajectory = pd.DataFrame(rows)
trajectory = trajectory.groupby("quarters_before_distress", as_index=False).mean(numeric_only=True)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
trajectory.to_csv(RESULTS_DIR / "pre_distress_trajectory_summary.csv", index=False)
print(trajectory)
