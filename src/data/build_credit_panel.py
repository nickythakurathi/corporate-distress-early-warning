import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RATIO_FILE, CRSP_FILE, DATA_PROCESSED, MERGED_PANEL_FILE, MODEL_READY_FILE


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\ufeff", "", regex=False)
        .str.replace(r"\s+", "_", regex=True)
        .str.lower()
    )
    return df


def convert_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def main() -> None:
    if not RATIO_FILE.exists() or not CRSP_FILE.exists():
        raise FileNotFoundError(
            "Raw CSV files were not found. Put them in data/raw/ before running this script."
        )

    ratios = pd.read_csv(RATIO_FILE, low_memory=False)
    crsp = pd.read_csv(CRSP_FILE, low_memory=False)

    ratios = clean_columns(ratios)
    crsp = clean_columns(crsp)

    ratios = ratios.rename(columns={
        "permno.": "permno",
        "perm_no": "permno",
        "publicdate": "public_date",
        "g_prof": "gprof",
    })

    crsp = crsp.rename(columns={
        "permno.": "permno",
        "perm_no": "permno",
    })

    required_ratio_cols = ["permno", "qdate"]
    required_crsp_cols = ["permno", "qdate"]

    for col in required_ratio_cols:
        if col not in ratios.columns:
            raise ValueError(f"Missing required column in ratios file: {col}")
    for col in required_crsp_cols:
        if col not in crsp.columns:
            raise ValueError(f"Missing required column in CRSP file: {col}")

    ratio_date_cols = ["qdate", "adate", "public_date"]
    crsp_date_cols = ["qdate", "date", "paydt", "rcrddt"]

    for col in ratio_date_cols:
        if col in ratios.columns:
            ratios[col] = pd.to_datetime(ratios[col], errors="coerce")
    for col in crsp_date_cols:
        if col in crsp.columns:
            crsp[col] = pd.to_datetime(crsp[col], errors="coerce")

    ratio_numeric_cols = [
        "permno", "gvkey", "bm", "npm", "gpm", "roa", "roe", "gprof",
        "invt_act", "rect_act", "curr_debt", "cash_debt", "debt_capital",
        "de_ratio", "quick_ratio", "curr_ratio", "inv_turn", "at_turn", "year",
    ]
    crsp_numeric_cols = [
        "permno", "shrcd", "siccd", "dlret", "vol", "ret", "shrout", "sprtrn", "year",
    ]

    ratios = convert_numeric_columns(ratios, ratio_numeric_cols)
    crsp = convert_numeric_columns(crsp, crsp_numeric_cols)

    if "divyield" in ratios.columns:
        ratios["divyield"] = (
            ratios["divyield"].astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
            .replace({"": np.nan, "nan": np.nan, "none": np.nan})
        )
        ratios["divyield"] = pd.to_numeric(ratios["divyield"], errors="coerce") / 100

    ratios["obs_date"] = ratios["public_date"] if "public_date" in ratios.columns else pd.NaT
    if "qdate" in ratios.columns:
        ratios["obs_date"] = ratios["obs_date"].fillna(ratios["qdate"])

    crsp["mkt_date"] = crsp["date"] if "date" in crsp.columns else pd.NaT
    if "qdate" in crsp.columns:
        crsp["mkt_date"] = crsp["mkt_date"].fillna(crsp["qdate"])

    ratios = ratios.dropna(subset=["permno", "obs_date"]).copy()
    crsp = crsp.dropna(subset=["permno", "mkt_date"]).copy()

    ratios["permno"] = pd.to_numeric(ratios["permno"], errors="coerce")
    crsp["permno"] = pd.to_numeric(crsp["permno"], errors="coerce")
    ratios = ratios.dropna(subset=["permno"]).copy()
    crsp = crsp.dropna(subset=["permno"]).copy()

    ratios["permno"] = ratios["permno"].astype(int)
    crsp["permno"] = crsp["permno"].astype(int)

    ratios = ratios.sort_values(["obs_date", "permno"]).reset_index(drop=True)
    crsp = crsp.sort_values(["mkt_date", "permno"]).reset_index(drop=True)

    merged = pd.merge_asof(
        ratios,
        crsp,
        left_on="obs_date",
        right_on="mkt_date",
        by="permno",
        direction="backward",
        allow_exact_matches=True,
        suffixes=("", "_crsp"),
    )

    dlret_threshold = -0.30
    horizon_q = 4

    crsp_event = crsp.copy().sort_values(["permno", "mkt_date"]).reset_index(drop=True)

    for h in range(1, horizon_q + 1):
        crsp_event[f"dlret_lead_{h}"] = crsp_event.groupby("permno")["dlret"].shift(-h)
        crsp_event[f"date_lead_{h}"] = crsp_event.groupby("permno")["mkt_date"].shift(-h)

    lead_dlret_cols = [f"dlret_lead_{h}" for h in range(1, horizon_q + 1)]
    future_obs_cols = [f"date_lead_{h}" for h in range(1, horizon_q + 1)]

    crsp_event["bad_dlret_next_4q"] = crsp_event[lead_dlret_cols].le(dlret_threshold).any(axis=1).astype(int)
    crsp_event["future_obs_count_4q"] = crsp_event[future_obs_cols].notna().sum(axis=1)

    global_max_date = crsp_event["mkt_date"].max()
    crsp_event["enough_time_to_observe"] = (
        crsp_event["mkt_date"] <= (global_max_date - pd.Timedelta(days=365))
    ).astype(int)

    crsp_event["exit_next_4q"] = (
        (crsp_event["future_obs_count_4q"] == 0) &
        (crsp_event["enough_time_to_observe"] == 1)
    ).astype(int)

    crsp_event["distress_next_4q"] = (
        (crsp_event["bad_dlret_next_4q"] == 1) |
        (crsp_event["exit_next_4q"] == 1)
    ).astype(int)

    event_df = crsp_event[
        [
            "permno", "mkt_date", "bad_dlret_next_4q", "exit_next_4q",
            "distress_next_4q", "future_obs_count_4q", "enough_time_to_observe",
        ]
    ].copy()

    final_df = merged.merge(event_df, on=["permno", "mkt_date"], how="left")

    if "ret" in final_df.columns and "sprtrn" in final_df.columns:
        final_df["excess_ret"] = final_df["ret"] - final_df["sprtrn"]

    for col in ["roa", "roe", "de_ratio", "quick_ratio", "curr_ratio", "npm", "gpm"]:
        if col in final_df.columns:
            final_df[f"{col}_lag1"] = final_df.groupby("permno")[col].shift(1)
            final_df[f"{col}_chg1"] = final_df[col] - final_df[f"{col}_lag1"]

    model_df = final_df.copy()
    if "enough_time_to_observe" in model_df.columns:
        model_df = model_df[model_df["enough_time_to_observe"] == 1].copy()

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(MERGED_PANEL_FILE, index=False)
    model_df.to_csv(MODEL_READY_FILE, index=False)

    print("Saved:")
    print(MERGED_PANEL_FILE)
    print(MODEL_READY_FILE)


if __name__ == "__main__":
    main()
