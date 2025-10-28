
import os, time
import numpy as np
import pandas as pd


IN_DIR       = "outputs/62_run"
OUT63_DIR       = "outputs/63_run"
IN_FLAGS_CSV  = os.path.join(IN_DIR, "62_compare.csv")
IN_HATS_CSV   = os.path.join(IN_DIR, "62_observed.csv")


DO_FILES      = [
    os.path.join(OUT63_DIR, "631_do__Full_main.csv"),
    os.path.join(OUT63_DIR, "631_do__NoAtoB.csv"),
    os.path.join(OUT63_DIR, "631_do__NoBtoC.csv"),
    os.path.join(OUT63_DIR, "631_do__NoCtoD.csv"),
    os.path.join(OUT63_DIR, "631_do__None.csv"),
]
OUT_AE_SINGLE   = os.path.join(OUT63_DIR, "631_A_to_E_cum_counterfactual.csv")
OUT_ABCD_SINGLE = os.path.join(OUT63_DIR, "631_A_to_BCD_inst_counterfactual.csv")
OUT_AE_ALL      = os.path.join(OUT63_DIR, "631_A_to_E_cum_counterfactual.ALL.csv")
OUT_ABCD_ALL    = os.path.join(OUT63_DIR, "631_A_to_BCD_inst_counterfactual.ALL.csv")
OUT_AE_METRICS  = os.path.join(OUT63_DIR, "631_metrics_AE_all.csv")

DO_FILES_EONLY = [
    os.path.join(OUT63_DIR, "632_do__NoAtoE.csv"),
    os.path.join(OUT63_DIR, "632_do__NoBtoE.csv"),
    os.path.join(OUT63_DIR, "632_do__NoCtoE.csv"),
    os.path.join(OUT63_DIR, "632_do__NoDtoE.csv"),
    os.path.join(OUT63_DIR, "632_do__NoneE.csv"),
]
OUT_AE_EONLY_ALL = os.path.join(OUT63_DIR, "632_A_to_E_cum_counterfactual.EONLY.ALL.csv")

# ===== params =====
ID_COL="stay_id"; T_COL="t"
A_LOW="A_low"; B_FLAG="B_on"; C_FLAG="C_low"; D_FLAG="D_high"; E_FLAG="E_on"
B_HAT="B_hat"; C_HAT="C_hat"; D_HAT="D_hat"; E_HAT="E_hat"
B_HAT_CF="B_hat_doA0"; C_HAT_CF="C_hat_doA0"; D_HAT_CF="D_hat_doA0"; E_HAT_CF="E_hat_doA0"
LAG_PRE=-12; LAG_POST=24
PROG_STEP=10

# ===== utils =====
def _need(df, cols, where):
    miss=[c for c in cols if c not in df.columns]
    if miss: raise KeyError(f"{where}: missing {miss}")

def _onsets(x: np.ndarray)->np.ndarray:
    x=x.astype(np.int8); prev=np.r_[0,x[:-1]]
    return (x==1)&(prev==0)

def extract_onsets(df: pd.DataFrame, flag_col: str)->pd.DataFrame:
    rows=[]
    for sid,g in df.groupby(ID_COL, sort=False):
        idx=np.flatnonzero(_onsets(g[flag_col].to_numpy()))
        if idx.size: rows.append(pd.DataFrame({ID_COL:sid,"t0":g.iloc[idx][T_COL].to_numpy(np.int64)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=[ID_COL,"t0"])

def _build_index(df: pd.DataFrame, val_col: str):
    by,pos,tmin,tmax={}, {}, {}, {}
    for sid,g in df.groupby(ID_COL, sort=False):
        ts=g[T_COL].to_numpy(np.int64); vs=g[val_col].to_numpy()
        by[sid]=(ts,vs); pos[sid]={int(t):i for i,t in enumerate(ts)}
        tmin[sid]=int(ts.min()); tmax[sid]=int(ts.max())
    return by,pos,tmin,tmax

def align_mean(df,trig,val_col,lag_pre,lag_post,tag):
    lags=np.arange(lag_pre,lag_post+1,dtype=int)
    by,pos,tmin,tmax=_build_index(df,val_col)
    rows=[]; total=len(lags)
    for i,L in enumerate(lags,1):
        vals=[]
        for sid,g0 in trig.groupby(ID_COL,sort=False):
            if sid not in by: continue
            ts,vs=by[sid]; m=pos[sid]
            for t0 in g0["t0"].to_numpy(np.int64):
                tt=int(t0+L)
                if tt<tmin[sid] or tt>tmax[sid]: continue
                j=m.get(tt,None)
                if j is None: continue
                v=vs[j]
                if np.isfinite(v): vals.append(float(v))
        rows.append((int(L), float(np.mean(vals)) if vals else np.nan, int(len(vals))))
        if (i%PROG_STEP==0) or (i==total): print(f"[62_06] align {tag} {i}/{total}", flush=True)
    return pd.DataFrame(rows, columns=["lag","mean","n"])

def align_cum_rebased(df,trig,cum_col,lag_pre,lag_post,tag):
    lags=np.arange(lag_pre,lag_post+1,dtype=int)
    by,pos,tmin,tmax=_build_index(df,cum_col)
    rows=[]; total=len(lags)
    for i,L in enumerate(lags,1):
        vals=[]
        for sid,g0 in trig.groupby(ID_COL,sort=False):
            if sid not in by: continue
            ts,vs=by[sid]; m=pos[sid]
            for t0 in g0["t0"].to_numpy(np.int64):
                t_base=int(t0-1); j_base=m.get(t_base,None)
                base=float(vs[j_base]) if (j_base is not None and np.isfinite(vs[j_base])) else (
                      float(vs[m.get(int(t0),None)]) if m.get(int(t0),None) is not None else np.nan)
                if not np.isfinite(base): continue
                tt=int(t0+L)
                if tt<tmin[sid] or tt>tmax[sid]: continue
                j=m.get(tt,None)
                if j is None or not np.isfinite(vs[j]): continue
                inc=float(vs[j])-base
                if inc<0.0: inc=0.0
                vals.append(inc)
        rows.append((int(L), float(np.mean(vals)) if vals else np.nan, int(len(vals))))
        if (i%PROG_STEP==0) or (i==total): print(f"[62_06] align {tag} {i}/{total}", flush=True)
    return pd.DataFrame(rows, columns=["lag","mean","n"])

def _pair(df, evA, v_obs, v_cf, taglbl, obs_name, cf_name, zero_align_at=-1):
    o = align_mean(df, evA, v_obs, LAG_PRE, LAG_POST, f"{taglbl} obs")
    c = align_mean(df, evA, v_cf,  LAG_PRE, LAG_POST, f"{taglbl} cf")
    z = o.merge(c, on="lag", suffixes=("_obs", "_cf"))
    if zero_align_at is not None and (z["lag"] == zero_align_at).any():
        rowz = z.loc[z["lag"] == zero_align_at].iloc[0]
        off = (rowz["mean_obs"] - rowz["mean_cf"]) if np.isfinite(rowz["mean_obs"]) and np.isfinite(rowz["mean_cf"]) else 0.0
        z["mean_cf"] = z["mean_cf"] + off  
    z["delta"] = z["mean_obs"] - z["mean_cf"]
    return z.rename(columns={"mean_obs": obs_name, "mean_cf": cf_name})

def _process_one_do(do_path: str, df_obs: pd.DataFrame, evA: pd.DataFrame) -> dict:
    df_do = pd.read_csv(do_path, dtype={"run_tag": str}, keep_default_na=False) \
           .sort_values([ID_COL, T_COL]).reset_index(drop=True)
    _need(df_do, [ID_COL, T_COL, B_HAT_CF, C_HAT_CF, D_HAT_CF, E_HAT_CF, "run_tag"], os.path.basename(do_path))
    tag = str(df_do["run_tag"].iloc[0])
    if not np.array_equal(df_obs[[ID_COL, T_COL]].to_numpy(),
                          df_do [[ID_COL, T_COL]].to_numpy()):
        raise RuntimeError(f"key mismatch between observed and {os.path.basename(do_path)}")
    df = df_obs.copy()
    for c in [B_HAT_CF, C_HAT_CF, D_HAT_CF, E_HAT_CF]:
        df[c] = df_do[c].to_numpy()
    cf_E = align_cum_rebased(df, evA, E_HAT_CF, LAG_PRE, LAG_POST, f"A->E cf ({tag})")
    cf_E = cf_E.rename(columns={"mean": "cum_cf"})[["lag", "cum_cf"]]
    cf_E["tag"] = tag
    tb = _pair(df, evA, B_HAT, B_HAT_CF, f"A->B ({tag})", "B_obs", "B_cf")
    tc = _pair(df, evA, C_HAT, C_HAT_CF, f"A->C ({tag})", "C_obs", "C_cf")
    td = _pair(df, evA, D_HAT, D_HAT_CF, f"A->D ({tag})", "D_obs", "D_cf")
    tb = tb.rename(columns={"delta": "delta_B", "n_obs": "nB_obs", "n_cf": "nB_cf"})
    tc = tc.rename(columns={"delta": "delta_C", "n_obs": "nC_obs", "n_cf": "nC_cf"})
    td = td.rename(columns={"delta": "delta_D", "n_obs": "nD_obs", "n_cf": "nD_cf"})
    abcd = (tb.merge(tc, on="lag").merge(td, on="lag"))
    abcd["tag"] = tag
    return {"tag": tag, "cf_E": cf_E, "abcd": abcd}

# ===== main =====
def main():
    import multiprocessing as mp
    t0 = time.time()
    print("[62_06] start (parallel)", flush=True)

    need_f = [ID_COL, T_COL, A_LOW, B_FLAG, C_FLAG, D_FLAG, E_FLAG]
    cols_f = pd.read_csv(IN_FLAGS_CSV, nrows=0).columns.tolist()
    miss_f = [c for c in need_f if c not in cols_f]
    if miss_f: raise ValueError(f"schema mismatch in 62_compare, missing: {miss_f}")
    df_flags = (pd.read_csv(IN_FLAGS_CSV, usecols=need_f).sort_values([ID_COL, T_COL]).reset_index(drop=True))

    need_h = [ID_COL, T_COL, B_HAT, C_HAT, D_HAT, E_HAT]
    cols_h = pd.read_csv(IN_HATS_CSV, nrows=0).columns.tolist()
    miss_h = [c for c in need_h if c not in cols_h]
    if miss_h: raise ValueError(f"schema mismatch in 62_observed, missing: {miss_h}")
    df_hats = (pd.read_csv(IN_HATS_CSV, usecols=need_h).sort_values([ID_COL, T_COL]).reset_index(drop=True))

    if not np.array_equal(df_flags[[ID_COL, T_COL]].to_numpy(),
                          df_hats [[ID_COL, T_COL]].to_numpy()):
        raise RuntimeError("key mismatch between 62_compare and 62_observed")

    df_obs = pd.concat([df_flags, df_hats[[B_HAT, C_HAT, D_HAT, E_HAT]]], axis=1)

    evA = extract_onsets(df_obs, A_LOW)
    if evA.empty: raise RuntimeError("no A_low onsets detected")

    obs_E = align_cum_rebased(df_obs, evA, E_HAT, LAG_PRE, LAG_POST, "A->E obs (ALL-ref)")
    obs_E = obs_E.rename(columns={"mean": "cum_obs"})[["lag", "cum_obs"]]

    do_list = [p for p in DO_FILES if os.path.isfile(p)]
    if len(do_list) == 0:
        raise FileNotFoundError("no DO files found (propagation)")
    args = [(p, df_obs, evA) for p in do_list]
    ncpu = max(1, mp.cpu_count() - 1)
    print(f"workers = {ncpu}", flush=True)
    with mp.get_context("spawn").Pool(processes=ncpu) as pool:
        results = pool.starmap(_process_one_do, args)

    ae_rows, abcd_rows, met_rows = [], [], []
    for out in results:
        tag  = out["tag"]
        cf_E = out["cf_E"]
        abcd = out["abcd"]
        ae = obs_E.merge(cf_E, on="lag")
        ae["cum_delta"] = ae["cum_obs"] - ae["cum_cf"]
        ae["tag"] = tag
        ae_rows.append(ae)
        delta = ae["cum_delta"].to_numpy(float)
        d_auc = float(np.nansum(delta))
        d_max = float(np.nanmax(delta)) if delta.size else np.nan
        onset = np.nan
        for L, d in zip(ae["lag"].to_numpy(int), delta):
            if np.isfinite(d) and d > 1e-9:
                onset = int(L); break
        met_rows.append((tag, d_auc, d_max, onset))
        abcd_rows.append(abcd)

    ae_all   = pd.concat(ae_rows,  ignore_index=True).sort_values(["tag","lag"]).reset_index(drop=True)
    abcd_all = pd.concat(abcd_rows, ignore_index=True).sort_values(["tag","lag"]).reset_index(drop=True)
    metrics  = pd.DataFrame(met_rows, columns=["tag","DeltaAUC","DeltaMax","LagOnset"]).sort_values("DeltaAUC", ascending=False)

    os.makedirs(OUT63_DIR, exist_ok=True)
    ae_all.to_csv(OUT_AE_ALL, index=False)
    abcd_all.to_csv(OUT_ABCD_ALL, index=False)
    metrics.to_csv(OUT_AE_METRICS, index=False)
    print(f" saved -> {OUT_AE_ALL}")
    print(f" saved -> {OUT_ABCD_ALL}")
    print(f" saved -> {OUT_AE_METRICS}")

    first = ae_all["tag"].unique()[0]
    ae_all[ae_all["tag"] == first][["lag", "cum_obs", "cum_cf", "cum_delta"]].to_csv(OUT_AE_SINGLE, index=False)
    abcd_all[abcd_all["tag"] == first].drop(columns=["tag"]).sort_values("lag").to_csv(OUT_ABCD_SINGLE, index=False)
    print(f" saved -> {OUT_AE_SINGLE}")
    print(f" saved -> {OUT_ABCD_SINGLE}")

    do_e = [p for p in DO_FILES_EONLY if os.path.isfile(p)]
    if len(do_e) == 0:
        print(" skip E-only (no E-only DO files found)", flush=True)
    else:
        print(" E-only aggregation ...", flush=True)
        args_e = [(p, df_obs, evA) for p in do_e]
        with mp.get_context("spawn").Pool(processes=ncpu) as pool:
            results_e = pool.starmap(_process_one_do, args_e)
        rows_e = []
        for out in results_e:
            tag = out["tag"]
            ae  = obs_E.merge(out["cf_E"], on="lag")
            ae["cum_delta"] = ae["cum_obs"] - ae["cum_cf"]
            ae["tag"] = tag
            rows_e.append(ae)
        ae_eonly_all = pd.concat(rows_e, ignore_index=True).sort_values(["tag","lag"]).reset_index(drop=True)
        ae_eonly_all.to_csv(OUT_AE_EONLY_ALL, index=False)
        print(f" saved -> {OUT_AE_EONLY_ALL}")

    print(f" done in {time.time()-t0:.1f}s", flush=True)

if __name__ == "__main__":
    main()
