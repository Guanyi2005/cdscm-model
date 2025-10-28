# ===== paths & files =====
import os, time
import numpy as np
import pandas as pd

IN_CSV  = "outputs/62_run/62_observed.csv"
OUT_DIR = "outputs/62_run"

OUT_BOUNDS_CSV   = os.path.join(OUT_DIR, "6201_series_bounds.csv")
OUT_LAGDEPS_CSV  = os.path.join(OUT_DIR, "6201_lag_dependencies.csv")
OUT_SPECRAD_CSV  = os.path.join(OUT_DIR, "6201_spectral_radius.csv")   
OUT_EON_MONO_CSV = os.path.join(OUT_DIR, "6201_Eon_monotonic.csv")     

# ===== params =====
ID_COL  = "stay_id"
T_COL   = "t"
HAT_COLS = ["B_hat", "C_hat", "D_hat"]   
A_FLAG   = "A_low"                       
E_ON    = "E_on"
LAG_MAX = 6
EPS     = 1e-12

ROLL_WINDOW = 24   
ROLL_STEP   = 6   

# ===== helpers =====
def _log(msg): print(f"[6201-FIG] {msg}", flush=True)

def _ensure_dir(p):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def _need_cols(df: pd.DataFrame, cols, where=""):
    miss = [c for c in cols if c not in df.columns]
    if miss: raise KeyError(f"{where} missing columns {miss}")

def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    v = pd.concat([x, y], axis=1).dropna()
    if len(v) < 3: return np.nan
    sx, sy = v.iloc[:,0].std(ddof=0), v.iloc[:,1].std(ddof=0)
    if sx < EPS or sy < EPS: return np.nan
    return float(v.iloc[:,0].corr(v.iloc[:,1]))

# ----- A1 bounds -----
def _series_bounds(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in HAT_COLS:
        s = pd.to_numeric(df[c], errors="coerce")
        rows.append({
            "var": c,
            "n": int(s.notna().sum()),
            "min": float(np.nanmin(s)) if s.notna().any() else np.nan,
            "p05": float(np.nanpercentile(s, 5)) if s.notna().any() else np.nan,
            "p50": float(np.nanmedian(s)) if s.notna().any() else np.nan,
            "p95": float(np.nanpercentile(s, 95)) if s.notna().any() else np.nan,
            "max": float(np.nanmax(s)) if s.notna().any() else np.nan,
            "mean": float(np.nanmean(s)) if s.notna().any() else np.nan,
            "std": float(np.nanstd(s, ddof=0)) if s.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)

# ----- A2 lag deps -----
def _lag_dependencies(df: pd.DataFrame, lag_max: int = LAG_MAX) -> pd.DataFrame:
    pairs = [(A_FLAG, "B_hat"), ("B_hat", "C_hat"), ("C_hat", "D_hat"),
             (A_FLAG, "C_hat"), (A_FLAG, "D_hat"), ("B_hat", "D_hat")]
    rows = []
    g = df[[ID_COL, T_COL, A_FLAG] + HAT_COLS].sort_values([ID_COL, T_COL]).copy()
    for lag in range(1, lag_max + 1):
        for p, c in pairs:
            p_lag = g.groupby(ID_COL, sort=False)[p].shift(lag)
            r = _safe_corr(p_lag, g[c])
            n = int(pd.concat([p_lag, g[c]], axis=1).dropna().shape[0])
            rows.append({"parent": p, "child": c, "lag": lag, "pearson_r": r, "n": n})
    return pd.DataFrame(rows)

# ----- A3 spectral radius  -----
def _operator_rho(block: pd.DataFrame) -> float:
    if block.empty:
        return np.nan
    g = block[[ID_COL, T_COL] + HAT_COLS].sort_values([ID_COL, T_COL]).copy()

    Z = {}
    for c in HAT_COLS:  
        s = pd.to_numeric(g[c], errors="coerce")
        mu, sd = s.mean(), s.std(ddof=0)
        Z[c] = (s - mu)/sd if sd >= EPS else pd.Series(np.nan, index=g.index)
    Z = pd.DataFrame(Z)

    edges = [("B_hat","C_hat"), ("C_hat","D_hat"), ("B_hat","D_hat")]
    idx = {n:i for i,n in enumerate(HAT_COLS)} 

    K = np.zeros((3,3), dtype=float)
    for p, c in edges:
        lagged = Z.groupby(g[ID_COL], sort=False)[p].shift(1)  # τ=1
        X = pd.concat([lagged, Z[c]], axis=1).dropna()
        beta = float(X.iloc[:,0].corr(X.iloc[:,1])) if X.shape[0] >= 5 else np.nan
        if not np.isnan(beta):
            K[idx[c], idx[p]] = beta

    for i in range(3):
        for j in range(i, 3):
            K[i, j] = 0.0

    ev = np.linalg.eigvals(K)
    return float(np.max(np.abs(ev))) if ev.size else np.nan


def _rolling_specr(df: pd.DataFrame) -> pd.DataFrame:
    tmin, tmax = int(df[T_COL].min()), int(df[T_COL].max())
    rows = []
    for start in range(tmin, tmax - ROLL_WINDOW + 1, ROLL_STEP):
        mid = start + ROLL_WINDOW // 2
        block = df[(df[T_COL] >= start) & (df[T_COL] < start + ROLL_WINDOW)]
        rho = _operator_rho(block)
        rows.append({"t_mid": mid, "rho": rho})

    if not rows:  
        rows = [{"t_mid": (tmin + tmax)//2, "rho": _operator_rho(df)}]

    out = pd.DataFrame(rows)
    out.to_csv(OUT_SPECRAD_CSV, index=False)
    return out

# ----- A4 terminal curves  -----
def _eon_curves(df: pd.DataFrame) -> pd.DataFrame:
    g = df[[ID_COL, T_COL, E_ON]].sort_values([ID_COL, T_COL]).copy()
    g[E_ON] = pd.to_numeric(g[E_ON], errors="coerce").fillna(0).astype(int)

    pt = g.groupby(T_COL, sort=False)[E_ON].mean().reset_index(name="mean_Eon")

    g["ever"] = (g.groupby(ID_COL, sort=False)[E_ON].cumsum() > 0).astype(int)
    cm = g.groupby(T_COL, sort=False)["ever"].mean().reset_index(name="cum_mean")

    out = pd.merge(pt, cm, on=T_COL, how="outer").sort_values(T_COL)
    out.rename(columns={T_COL: "t"}, inplace=True)
    out.to_csv(OUT_EON_MONO_CSV, index=False)
    return out

# ===== main =====
def main():
    t0 = time.time()
    _ensure_dir(OUT_DIR)
    use_cols = [ID_COL, T_COL] + HAT_COLS + [A_FLAG, E_ON]
    _log(f"read: {IN_CSV}")
    df = pd.read_csv(IN_CSV, usecols=use_cols)
    _need_cols(df, use_cols, "observed")

    _log("series bounds")
    _series_bounds(df).to_csv(OUT_BOUNDS_CSV, index=False)

    _log("lag dependencies")
    _lag_dependencies(df, lag_max=LAG_MAX).to_csv(OUT_LAGDEPS_CSV, index=False)

    _log("rolling spectral radius")
    _rolling_specr(df)  

    _log("terminal curves (mean_Eon & cum_mean)")
    _eon_curves(df)     

    _log(f"done, elapsed={time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
