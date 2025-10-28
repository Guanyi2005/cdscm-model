
import pandas as pd
from tqdm import tqdm

# ===== paths & files =====
IN_OBS  = "outputs/62_run/62_observed.csv"       
OUT_ERR = "outputs/64_run/64_additivity_error.csv"
OUT_LAG = "outputs/64_run/64_additivity_by_lag.csv"

# ===== params =====
COL_ID = "stay_id"
COL_LAG = "t"
COL_E   = "E_hat"

# ===== utils =====
def _need_cols(df: pd.DataFrame, cols, path: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"missing columns in {path}: {miss}")

# ===== main =====
def main():
    print("[6401] read observed …")
    df = pd.read_csv(IN_OBS)
    _need_cols(df, [COL_ID, COL_LAG, COL_E], IN_OBS)

    print("[6401] compute per-lag mean and residual …")
    g = df[[COL_LAG, COL_E]].dropna()
    per_lag = (g.groupby(COL_LAG, as_index=False)[COL_E]
                 .mean()
                 .rename(columns={COL_LAG: "lag", COL_E: "E_mean"})
                 .sort_values("lag")
                 .reset_index(drop=True))

    per_lag["resid_mean"] = per_lag["E_mean"].diff().fillna(0)
    per_lag[["lag", "resid_mean"]].to_csv(OUT_LAG, index=False)
    per_lag[["lag", "resid_mean"]].rename(columns={"lag": COL_LAG}).to_csv(OUT_ERR, index=False)

    print(f"[6401] wrote: {OUT_LAG} , {OUT_ERR}")

if __name__ == "__main__":
    main()
