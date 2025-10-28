
import pandas as pd
from pathlib import Path

# ===== paths & files =====
IN_INP   = Path("outputs/71_run/71_inputs.csv")        
OUT_OBS  = Path("outputs/71_run/71_observed_clean.csv")

# ===== params =====
FILL_METHOD = None 

# ===== helpers =====
def _need(df, cols, name):
    miss = [c for c in cols if c not in df.columns]
    if miss: raise RuntimeError(f"missing {miss} in {name}")

# ===== main =====
def main():
    df = pd.read_csv(IN_INP)
    _need(df, ["t","A","B","C","D","E"], IN_INP.name)
    df = df.sort_values("t").reset_index(drop=True)
    df.to_csv(OUT_OBS, index=False)
    print(f"[71_03] wrote {OUT_OBS} rows={len(df)}")

if __name__ == "__main__":
    main()
