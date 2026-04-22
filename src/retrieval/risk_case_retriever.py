import pandas as pd
from pathlib import Path
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import MODEL_READY_FILE, RESULTS_DIR

if not MODEL_READY_FILE.exists():
    raise FileNotFoundError("Run src/data/build_credit_panel.py first.")

df = pd.read_csv(MODEL_READY_FILE, low_memory=False)
df["obs_date"] = pd.to_datetime(df["obs_date"], errors="coerce")
df["obs_year"] = df["obs_date"].dt.year

text_vars = [c for c in [
    "bm", "npm", "gpm", "roa", "roe", "gprof",
    "debt_capital", "de_ratio", "quick_ratio", "curr_ratio",
    "at_turn", "ret", "excess_ret", "vol", "shrout"
] if c in df.columns]

def to_case_text(row: pd.Series) -> str:
    fields = [f"permno {row.get('permno', '')}", f"year {row.get('obs_year', '')}"]
    for col in text_vars:
        fields.append(f"{col} {row.get(col, '')}")
    if "distress_next_4q" in row.index:
        fields.append(f"distress_next_4q {row['distress_next_4q']}")
    return " ".join(map(str, fields))

df["case_text"] = df.apply(to_case_text, axis=1)

vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(df["case_text"])

def retrieve_cases(query: str, top_k: int = 10) -> pd.DataFrame:
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, matrix).flatten()
    out = df.copy()
    out["similarity"] = sims
    return out.sort_values("similarity", ascending=False).head(top_k)

sample = retrieve_cases("high distress risk with weak liquidity and poor profitability", top_k=10)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
sample.to_csv(RESULTS_DIR / "retrieved_cases_sample.csv", index=False)
print(sample[["permno", "obs_date", "distress_next_4q", "similarity"]])
