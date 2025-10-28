
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ===== paths & files =====
DL_DIR  = Path("inputs/71_data")
RAW_DIR = Path("inputs/71_raw"); RAW_DIR.mkdir(parents=True, exist_ok=True)

MAP = {
    "MORTGAGE30US.csv": ("71_A_rate.csv",          "A"),  
    "DRTSCILM.csv":     ("71_B_lending.csv",       "B"),  
    "PERMIT.csv":       ("71_C_permits.csv",       "C"),  
    "USCONS.csv":       ("71_D_construction.csv",  "D"),  
    "PRRESCON.csv":     ("71_E_res_invest.csv",    "E"),  
}

# ===== params =====
NEEDED = ["DATE", "VALUE"] 

# ===== helpers =====
def _need(df, cols, name):
    miss = [c for c in cols if c not in df.columns]
    if miss: raise RuntimeError(f"missing columns {miss} in {name}")

def _month_index_from_min(dates: pd.Series, base: pd.Timestamp) -> pd.Series:
    dt = pd.to_datetime(dates)
    return (dt.dt.year - base.year) * 12 + (dt.dt.month - base.month)

# ===== main =====
def main():
    dfs = {}
    for fname in MAP.keys():
        fpath = DL_DIR / fname
        if not fpath.exists():
            raise FileNotFoundError(f"missing {fpath}")
        df = pd.read_csv(fpath)
        _need(df, NEEDED, fpath.name)
        df = df.copy()
        df["DATE"] = pd.to_datetime(df["DATE"])
        dfs[fname] = df

    base_date = min(df["DATE"].min() for df in dfs.values())

    for fname, (out_name, col) in tqdm(MAP.items(), desc="[71_00] convert to RAW"):
        df = dfs[fname].copy()
        df["t"] = _month_index_from_min(df["DATE"], base_date).astype(int)
        out = df[["t", "VALUE"]].rename(columns={"VALUE": col}).sort_values("t").reset_index(drop=True)
        out.to_csv(RAW_DIR / out_name, index=False)

    print(f"[71_00] wrote RAW to {RAW_DIR}")

if __name__ == "__main__":
    main()
