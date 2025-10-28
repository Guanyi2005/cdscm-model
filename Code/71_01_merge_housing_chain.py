
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ===== paths & files =====
IN_DIR  = Path("outputs/71_data")                         
OUT_DIR = Path("outputs/71_run"); OUT_DIR.mkdir(parents=True, exist_ok=True)
F_A = IN_DIR / "MORTGAGE30US.csv"  
F_B = IN_DIR / "DRTSCILM.csv"       
F_C = IN_DIR / "PERMIT.csv"         
F_D = IN_DIR / "USCONS.csv"        
F_E = IN_DIR / "PRRESCON.csv"       
OUT_OBS = OUT_DIR / "71_observed.csv"          

# ===== params =====
SERIES_COL = {
    "MORTGAGE30US.csv": "MORTGAGE30US",
    "DRTSCILM.csv":     "DRTSCILM",
    "PERMIT.csv":       "PERMIT",
    "USCONS.csv":       "USCONS",
    "PRRESCON.csv":     "PRRESCON",
}
DATE_COL = "observation_date" 

FREQ_MAP = {
    "MORTGAGE30US.csv": "W",  
    "DRTSCILM.csv":     "Q",  
    "PERMIT.csv":       "M",  
    "USCONS.csv":       "M",  
    "PRRESCON.csv":     "M",  
}

NAME_MAP = {
    "MORTGAGE30US.csv": "A",
    "DRTSCILM.csv":     "B",
    "PERMIT.csv":       "C",
    "USCONS.csv":       "D",
    "PRRESCON.csv":     "E",
}

# ===== helpers =====
def _need(df: pd.DataFrame, cols, name: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"missing columns {miss} in {name}")

def _read_fred_csv(path: Path) -> pd.DataFrame:
    """Read FRED csv with fixed series column, rename to DATE, VALUE."""
    ser_col = SERIES_COL[path.name]
    df = pd.read_csv(path)
    _need(df, [DATE_COL, ser_col], path.name)
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.rename(columns={DATE_COL: "DATE", ser_col: "VALUE"})
    df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df = df.dropna(subset=["VALUE"]).reset_index(drop=True)
    return df 

def _to_monthly(df: pd.DataFrame, freq_in: str) -> pd.DataFrame:
    """Convert to monthly series with columns: DATE, VALUE (one row per month)."""
    s = df.set_index("DATE")["VALUE"]

    if freq_in == "M":
        m = s.resample("ME").last()

    elif freq_in == "W":
        m = s.resample("ME").mean()

    elif freq_in == "Q":
        q = s.resample("QE-DEC").last()
        m = q.resample("ME").ffill()

    else:
        raise RuntimeError(f"unsupported freq_in={freq_in}")

    m = m.dropna()
    return m.to_frame(name="VALUE").reset_index()

def _month_index_from_min(dates: pd.Series, base: pd.Timestamp) -> pd.Series:
    dt = pd.to_datetime(dates)
    return ((dt.dt.year - base.year) * 12 + (dt.dt.month - base.month)).astype(int)

# ===== main =====
def main():
    files = [F_A, F_B, F_C, F_D, F_E]
    monthly = {}
    for fp in tqdm(files, desc="[71_01] load & monthly-ize"):
        if not fp.exists():
            raise FileNotFoundError(f"missing file: {fp}")
        fin = _read_fred_csv(fp)                   
        freq = FREQ_MAP[fp.name]
        mdf  = _to_monthly(fin, freq)                
        monthly[fp.name] = mdf

    base_date = min(mdf["DATE"].min() for mdf in monthly.values())
    for k in monthly.keys():
        mdf = monthly[k]
        mdf["t"] = _month_index_from_min(mdf["DATE"], base_date)
        monthly[k] = mdf[["t", "VALUE"]].sort_values("t").reset_index(drop=True)

    tset = None
    for k in monthly.keys():
        s = set(monthly[k]["t"].tolist())
        tset = s if tset is None else (tset & s)
    tlist = sorted(list(tset))

    out = pd.DataFrame({"t": tlist})
    for fp in files:
        var = NAME_MAP[fp.name]
        dfv = monthly[fp.name]
        dfv = dfv[dfv["t"].isin(tlist)][["t", "VALUE"]].rename(columns={"VALUE": var})
        out = out.merge(dfv, on="t", how="left")

    out = out.sort_values("t").reset_index(drop=True)
    _need(out, ["t","A","B","C","D","E"], "assembled")
    out.to_csv(OUT_OBS, index=False)
    print(f"[71_01] wrote {OUT_OBS} rows={len(out)}  t_range=[{out['t'].min()}, {out['t'].max()}]")

if __name__ == "__main__":
    main()
