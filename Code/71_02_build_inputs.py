
import pandas as pd
from pathlib import Path

# ===== paths & files =====
IN_OBS  = Path("outputs/71_run/71_observed.csv")   
OUT_INP = Path("outputs/71_run/71_inputs.csv")    

# ===== params =====
PASS = True

# ===== helpers =====
def _need(df, cols, name):
    miss = [c for c in cols if c not in df.columns]
    if miss: raise RuntimeError(f"missing {miss} in {name}")

# ===== main =====
def main():
    df = pd.read_csv(IN_OBS)
    _need(df, ["t","A","B","C","D","E"], IN_OBS.name)
    df = df.sort_values("t").reset_index(drop=True)
    df.to_csv(OUT_INP, index=False)
    print(f"[71_02] wrote {OUT_INP} rows={len(df)}")

if __name__ == "__main__":
    main()
