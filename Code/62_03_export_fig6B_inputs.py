
import os, time
import numpy as np
import pandas as pd

# ===== paths & files =====
IN_CSV  = "outputs/62_run/62_observed.csv"
OUT_DIR = "outputs/62_run"
OUT_AB  = os.path.join(OUT_DIR, "6202_pair_A_to_B.csv")
OUT_BC  = os.path.join(OUT_DIR, "6202_pair_B_to_C.csv")
OUT_CD  = os.path.join(OUT_DIR, "6202_pair_C_to_D.csv")
OUT_DE  = os.path.join(OUT_DIR, "6202_pair_D_to_E_cum.csv")

# ===== params =====
ID_COL = "stay_id"
T_COL  = "t"
A_LOW  = "A_low"
B_ON   = "B_on"
C_LOW  = "C_low"
D_HIGH = "D_high"
E_ON   = "E_on"   
LAG_PRE  = -12
LAG_POST = 24
BOOT_N   = 500
SEED0    = 13
PROG_STEP = 10

# ===== helpers =====
def _need_cols(df, need):
    miss = [c for c in need if c not in df.columns]
    if miss: raise KeyError(f"missing columns: {miss}")

def _rising_edges(flag: np.ndarray) -> np.ndarray:
    x = flag.astype(np.int8)
    prev = np.r_[0, x[:-1]]
    return (x == 1) & (prev == 0)

def _extract_onsets(df: pd.DataFrame, col: str) -> pd.DataFrame:
    out = []
    for sid, g in df.groupby(ID_COL, sort=False):
        idx = np.flatnonzero(_rising_edges(g[col].to_numpy()))
        if idx.size:
            out.append(pd.DataFrame({ID_COL: sid, "t0": g.iloc[idx][T_COL].to_numpy(dtype=np.int64)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=[ID_COL,"t0"])

def _align_rate(df: pd.DataFrame, trig: pd.DataFrame, resp_col: str,
                lag_pre: int, lag_post: int, boot_n: int, seed: int) -> pd.DataFrame:
    lags = np.arange(lag_pre, lag_post + 1, dtype=int)
    rows = []
    by_sid = {sid: g.set_index(T_COL)[resp_col].to_numpy() for sid, g in df.groupby(ID_COL, sort=False)}
    t_by  = {sid: g[T_COL].to_numpy(dtype=np.int64) for sid, g in df.groupby(ID_COL, sort=False)}
    pos = {sid: {int(t): i for i, t in enumerate(ts)} for sid, ts in t_by.items()}
    tmin = {sid: ts.min() for sid, ts in t_by.items()}
    tmax = {sid: ts.max() for sid, ts in t_by.items()}
    t0 = time.time()
    for i, L in enumerate(lags, 1):
        vals, n_all = [], 0
        for sid, g in trig.groupby(ID_COL, sort=False):
            if sid not in by_sid: continue
            y = by_sid[sid]; idx = pos[sid]
            for t in g["t0"].to_numpy(np.int64):
                tt = int(t + L)
                if tt < tmin[sid] or tt > tmax[sid]: continue
                j = idx.get(tt, None)
                if j is None: continue
                v = y[j]
                if not np.isnan(v):
                    vals.append(float(v)); n_all += 1
        if len(vals) == 0:
            rows.append((int(L), np.nan, np.nan, np.nan, 0))
        else:
            arr = np.asarray(vals, np.float64)
            rng = np.random.default_rng(seed + int(L))
            if boot_n > 0:
                bs = [arr[rng.integers(0, arr.size, arr.size)].mean() for _ in range(boot_n)]
                lo, hi = np.quantile(bs, [0.025, 0.975]); mean = float(arr.mean())
            else:
                mean, lo, hi = float(arr.mean()), np.nan, np.nan
            rows.append((int(L), mean, float(lo), float(hi), int(n_all)))
        if (i % PROG_STEP == 0) or (i == len(lags)):
            print(f"[6202-B] align {i}/{len(lags)}  elapsed={time.time()-t0:.1f}s", flush=True)
    return pd.DataFrame(rows, columns=["lag","mean","lo","hi","n"])

def _to_cumulative(tab: pd.DataFrame) -> pd.DataFrame:
    t = tab.sort_values("lag").reset_index(drop=True)
    for c in ("mean","lo","hi"):
        t["cum_"+c] = t[c].cumsum()
    return t[["lag","cum_mean","cum_lo","cum_hi","n"]]

# ===== main =====
def main():
    print("[6202-B] start", flush=True)
    use = [ID_COL, T_COL, A_LOW, B_ON, C_LOW, D_HIGH, E_ON]
    df = pd.read_csv(IN_CSV, usecols=use).sort_values([ID_COL, T_COL]).reset_index(drop=True)
    _need_cols(df, use)

    trig_A = _extract_onsets(df, A_LOW)
    trig_B = _extract_onsets(df, B_ON)
    trig_C = _extract_onsets(df, C_LOW)
    trig_D = _extract_onsets(df, D_HIGH)

    ab = _align_rate(df.rename(columns={B_ON:"resp"}), trig_A, "resp", LAG_PRE, LAG_POST, BOOT_N, SEED0+1); ab.to_csv(OUT_AB, index=False)
    bc = _align_rate(df.rename(columns={C_LOW:"resp"}), trig_B, "resp", LAG_PRE, LAG_POST, BOOT_N, SEED0+2); bc.to_csv(OUT_BC, index=False)
    cd = _align_rate(df.rename(columns={D_HIGH:"resp"}), trig_C, "resp", LAG_PRE, LAG_POST, BOOT_N, SEED0+3); cd.to_csv(OUT_CD, index=False)
    de = _align_rate(df.rename(columns={E_ON:"resp"}),   trig_D, "resp", LAG_PRE, LAG_POST, BOOT_N, SEED0+4)
    _to_cumulative(de).to_csv(OUT_DE, index=False)

    print(f"[6202-B] done -> {OUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
