
import os, time
import numpy as np
import pandas as pd

# ===== paths & files =====
IN_CSV  = "outputs/62_run/62_observed.csv"
OUT_DIR = "outputs/62_run"
OUT_AB  = os.path.join(OUT_DIR, "6203_chain_AB.csv")
OUT_BC  = os.path.join(OUT_DIR, "6203_chain_BC.csv")
OUT_CD  = os.path.join(OUT_DIR, "6203_chain_CD.csv")
OUT_DE  = os.path.join(OUT_DIR, "6203_chain_DE.csv")

# ===== params =====
ID_COL = "stay_id"
T_COL  = "t"
A_LOW, B_ON, C_LOW, D_HIGH, E_ON = "A_low", "B_on", "C_low", "D_high", "E_on"


LAG_PRE, LAG_POST = -12, 24
USE_BLOCK_SHUFFLE = True
BLOCK_HOURS = 12
SEED0 = 7

def _log(msg): print(f"[6203-C] {msg}", flush=True)

def _need_cols(df, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss: raise KeyError(f"missing columns: {miss}")

def _onset_vec(x: np.ndarray) -> np.ndarray:

    x = x.astype(np.int8, copy=False)
    prev = np.r_[0, x[:-1]]
    return ((x == 1) & (prev == 0)).astype(np.int8)

def _block_id(ts: np.ndarray, block_hours: int) -> np.ndarray:
    return (ts.astype(np.int64) // int(block_hours)).astype(np.int64)

def _aggregate_mean_over_lags(O: np.ndarray, Y: np.ndarray, valid: np.ndarray,
                              lag_pre: int, lag_post: int) -> tuple[np.ndarray, np.ndarray]:
    """
    返回 (num[lag_idx], den[lag_idx])，lag_idx 对应 range(lag_pre, lag_post+1)
    num = sum O[t] * Y[t+lag] (只在 valid[t+lag]=True 计入)
    den = sum O[t] *        1 (只在 valid[t+lag]=True 计入)
    """
    n = O.shape[0]
    lags = np.arange(lag_pre, lag_post+1, dtype=np.int32)
    num = np.zeros_like(lags, dtype=np.float64)
    den = np.zeros_like(lags, dtype=np.float64)

    O_f = O.astype(np.float64, copy=False)
    V_f = valid.astype(np.float64, copy=False)
    for i, L in enumerate(lags):
        if L == 0:
            y = Y
            v = V_f
            o = O_f
        elif L > 0:
            if n - L <= 0: continue
            y = Y[L:]
            v = V_f[L:]
            o = O_f[:-L]
        else:  # L < 0
            k = -L
            if n - k <= 0: continue
            y = Y[:-k]
            v = V_f[:-k]
            o = O_f[k:]

        if y.size == 0: 
            continue

        y_ = np.nan_to_num(y, copy=False)
        w  = o * v
        num[i] += float((w * y_).sum())
        den[i] += float(w.sum())
    return num, den

def _compute_curve(df: pd.DataFrame, vx: str, vy: str,
                   block_hours: int, seed: int) -> pd.DataFrame:
    lags = np.arange(LAG_PRE, LAG_POST+1, dtype=np.int32)

    num_ord = np.zeros_like(lags, dtype=np.float64)
    den_ord = np.zeros_like(lags, dtype=np.float64)

    num_shf = np.zeros_like(lags, dtype=np.float64)
    den_shf = np.zeros_like(lags, dtype=np.float64)

    rng = np.random.default_rng(seed)

    for sid, g in df.groupby(ID_COL, sort=False):
        x = g[vx].to_numpy()
        y = g[vy].to_numpy(dtype=float)
        ts= g[T_COL].to_numpy(dtype=np.int64)

        O  = _onset_vec(x)                      
        M  = ~np.isnan(y)                       
        bid= _block_id(ts, block_hours)      

        # A) ordered 累计
        n1, d1 = _aggregate_mean_over_lags(O, y, M, LAG_PRE, LAG_POST)
        num_ord += n1; den_ord += d1

        O_shf = np.zeros_like(O, dtype=np.int8)

        if O.sum() > 0:
            idx = np.flatnonzero(np.r_[True, np.diff(bid) != 0, True])

            for k in range(len(idx)-1):
                s, e = int(idx[k]), int(idx[k+1])  # [s, e)
                m = int(O[s:e].sum())
                if m <= 0:
                    continue
                Lblock = e - s
                m = min(m, Lblock)

                picks = rng.choice(Lblock, size=m, replace=False)
                O_shf[s + picks] = 1

        n2, d2 = _aggregate_mean_over_lags(O_shf, y, M, LAG_PRE, LAG_POST)
        num_shf += n2; den_shf += d2

    mean_ord = np.divide(num_ord, den_ord, out=np.full_like(num_ord, np.nan, dtype=float), where=den_ord>0)
    mean_shf = np.divide(num_shf, den_shf, out=np.full_like(num_shf, np.nan, dtype=float), where=den_shf>0)
    delta    = mean_ord - mean_shf

    out = pd.DataFrame({"lag": lags, "mean_ord": mean_ord, "mean_shf": mean_shf, "delta": delta})
    return out

def _run_chain(df, vx, vy, seed, path, tag):
    t = time.time()

    n_onsets = int((_onset_vec(df[vx].to_numpy())==1).sum())
    _log(f"{tag}: onsets={n_onsets}")
    curv = _compute_curve(df, vx, vy, BLOCK_HOURS, seed)
    curv.to_csv(path, index=False)
    _log(f"{tag}: wrote {path}  ({time.time()-t:.1f}s)")

def main():
    _log("start")
    use = [ID_COL, T_COL, A_LOW, B_ON, C_LOW, D_HIGH, E_ON]
    t0 = time.time()
    df = pd.read_csv(IN_CSV, usecols=use, memory_map=True).sort_values([ID_COL, T_COL]).reset_index(drop=True)
    _log(f"read {IN_CSV} shape={df.shape}  time={time.time()-t0:.1f}s")
    _need_cols(df, use)
    os.makedirs(OUT_DIR, exist_ok=True)

    _run_chain(df, A_LOW,  B_ON,  SEED0+1, OUT_AB, "A->B")
    _run_chain(df, B_ON,   C_LOW, SEED0+2, OUT_BC, "B->C")
    _run_chain(df, C_LOW,  D_HIGH,SEED0+3, OUT_CD, "C->D")
    _run_chain(df, D_HIGH, E_ON,  SEED0+4, OUT_DE, "D->E")

    _log("done")

if __name__ == "__main__":
    main()
