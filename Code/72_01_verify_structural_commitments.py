
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ===== paths & files =====
IN_OBS = Path("outputs/71_run/71_observed.csv")     
OUT_DIR = Path("outputs/72_run")
F_SPECR = OUT_DIR / "7201_spectral_radius.csv"      
F_ANCH  = OUT_DIR / "7201_anchors_A.csv"           
F_AB    = OUT_DIR / "7201_chain_AB.csv"             
F_BC    = OUT_DIR / "7201_chain_BC.csv"
F_CD    = OUT_DIR / "7201_chain_CD.csv"
F_DE    = OUT_DIR / "7201_chain_DE.csv"

# ===== params =====
COL_T = "t"
COLS  = ["A","B","C","D","E"]


R_WIN  = 36
R_STEP = 6
RIDGE  = 1e-6


LAG_MIN, LAG_MAX = -6, 12
N_SHUFFLE = 2000
ANCH_DIFF_Q = 0.10  


USE_BLOCK_SHUFFLE = True   
BLOCK_SIZE_MONTHS = 6      
USE_DIFF_SERIES   = True   

# ===== helpers =====
def need(df: pd.DataFrame, cols, name: str):
    miss = [c for c in cols if c not in df.columns]
    if miss: raise RuntimeError(f"missing columns {miss} in {name}")

def zscore_arr(a: np.ndarray) -> np.ndarray:
    m = float(np.nanmean(a))
    s = float(np.nanstd(a, ddof=0))
    return (a - m) / (s if s > 0 else 1.0)

def diff_z(a: np.ndarray) -> np.ndarray:
    a1 = np.diff(a.astype(float), n=1)
    return zscore_arr(a1)

def rolling_ordered_operator_radius(df: pd.DataFrame, cols):
    X = df[cols].to_numpy(dtype=float)
    t = df[COL_T].to_numpy(dtype=int)
    p = len(cols)
    rows = []
    for start in tqdm(range(0, len(df) - R_WIN + 1, R_STEP),
                      desc="[72_01] spectral radius ρ(K)", ncols=80):
        end = start + R_WIN
        Yw = X[start+1:end, :]
        Xw = X[start:end-1, :]
        K = np.zeros((p, p), dtype=float)
        for i in range(1, p):
            parents = list(range(0, i))
            Xi = Xw[:, parents]
            yi = Yw[:, i]
            if Xi.size == 0: 
                continue
            if not (np.isfinite(Xi).all() and np.isfinite(yi).all()):
                continue
            XtX = Xi.T @ Xi + RIDGE * np.eye(Xi.shape[1])
            Xty = Xi.T @ yi
            try:
                beta = np.linalg.solve(XtX, Xty)
                K[i, parents] = beta
            except np.linalg.LinAlgError:
                pass
        for i in range(p):
            for j in range(i, p):
                K[i, j] = 0.0
        eig = np.linalg.eigvals(K)
        rho = float(np.max(np.abs(eig))) if eig.size else np.nan
        t_mid = int(round(t[start:end].mean()))
        rows.append((t_mid, rho))
    return pd.DataFrame(rows, columns=["t_mid","rho"])

def make_anchors_A(df: pd.DataFrame):
    dA = df["A"].astype(float).diff()
    thr = float(np.nanquantile(dA, ANCH_DIFF_Q))
    anchor_t = (dA <= thr).astype(int)
    anchor_t.iloc[0] = 0
    return pd.DataFrame({COL_T: df[COL_T].astype(int), "anchor_t": anchor_t})

def corr_at_lag(x: np.ndarray, y: np.ndarray, lag: int) -> float:
    if lag == 0:
        a, b = x, y
    elif lag > 0:
        a, b = x[:-lag], y[lag:]
    else:
        a, b = x[-lag:], y[:lag]
    if len(a) < 3: return np.nan
    ax = a - a.mean(); bx = b - b.mean()
    den = ax.std(ddof=0) * bx.std(ddof=0)
    return float((ax @ bx) / (len(ax) * den)) if den > 0 else np.nan

def _block_shuffle(y: np.ndarray, block: int, rng) -> np.ndarray:
    n = len(y)
    y2 = y.copy()
    for s in range(0, n, block):
        e = min(s + block, n)
        rng.shuffle(y2[s:e])
    return y2

def make_chain(df: pd.DataFrame, up: str, dn: str) -> pd.DataFrame:
    xu = df[up].to_numpy(dtype=float)
    yd = df[dn].to_numpy(dtype=float)

    if USE_DIFF_SERIES:
        xu = diff_z(xu)
        yd = diff_z(yd)
    else:
        xu = zscore_arr(xu)
        yd = zscore_arr(yd)

    lags = list(range(LAG_MIN, LAG_MAX + 1))
    ord_vals = [corr_at_lag(xu, yd, k) for k in lags]

    rng = np.random.default_rng(2025)
    shf_acc = np.zeros(len(lags), dtype=float)
    for _ in tqdm(range(N_SHUFFLE), desc=f"[72_01] shuffle {up}->{dn}", leave=False, ncols=80):
        if USE_BLOCK_SHUFFLE:
            y_shf = _block_shuffle(yd, BLOCK_SIZE_MONTHS, rng)
        else:
            y_shf = yd.copy(); rng.shuffle(y_shf)
        for i, k in enumerate(lags):
            shf_acc[i] += corr_at_lag(xu, y_shf, k)
    shf_mean = shf_acc / N_SHUFFLE
    delta = np.array(ord_vals) - shf_mean
    return pd.DataFrame({"lag": lags, "mean_ord": ord_vals, "mean_shf": shf_mean, "delta": delta})

# ===== main =====
def main():
    print("[72_01] start")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    obs = pd.read_csv(IN_OBS)
    need(obs, [COL_T] + COLS, IN_OBS.name)
    obs = obs.sort_values(COL_T).reset_index(drop=True)

    # A) spectral radius with ordered-lag lower-triangular operator
    rho = rolling_ordered_operator_radius(obs, COLS)
    rho.to_csv(F_SPECR, index=False)

    # B) anchors for A
    anch = make_anchors_A(obs)
    anch.to_csv(F_ANCH, index=False)

    # C) chains
    make_chain(obs, "A", "B").to_csv(F_AB, index=False)
    make_chain(obs, "B", "C").to_csv(F_BC, index=False)
    make_chain(obs, "C", "D").to_csv(F_CD, index=False)
    make_chain(obs, "D", "E").to_csv(F_DE, index=False)

    print(f"[72_01] wrote:\n  {F_SPECR}\n  {F_ANCH}\n  {F_AB}\n  {F_BC}\n  {F_CD}\n  {F_DE}")

if __name__ == "__main__":
    main()
