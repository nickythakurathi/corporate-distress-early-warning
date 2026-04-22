from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

RATIO_FILE = DATA_RAW / "quarterly_Financial_Ratios_2000_2024.csv"
CRSP_FILE = DATA_RAW / "quarterly_CRSP_All_Ratios_2000_2024.csv"
MODEL_READY_FILE = DATA_PROCESSED / "model_ready_credit_panel.csv"
MERGED_PANEL_FILE = DATA_PROCESSED / "merged_credit_panel_with_distress_label.csv"
