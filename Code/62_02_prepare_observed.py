import os, time, yaml
import numpy as np
import pandas as pd

PARAMS_YAML = "62_00_params.yaml"

def _need(df: pd.DataFrame, cols, where: str):
    miss = [c for c in cols if c not in df.columns]
    if miss: raise KeyError(f"{where} missing columns {miss}")

def _clip01(x):
    return np.minimum(1.0, np.maximum(0.0, x))

def _log(s): print(f"[6202] {s}", flush=True)

def main():
    t0 = time.time()
    with open(PARAMS_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    cols  = cfg["columns"]
    eff   = cfg["effects"]  

    in_csv  = paths["v62_compare"]
    out_csv = os.path.join("outputs/62_run", "62_observed.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    _log(f"read: {in_csv}")
    df = pd.read_csv(in_csv)
    need = [
        cols["stay_id"], cols["t"],
        cols["A_value"], cols["B1_value"], cols["B2_value"], cols["B_sum_value"],
        cols["C_value"], cols["D_value"], cols["E_on"],
        cols["A_low_flag"], cols["B_on_flag"], cols["C_low_flag"], cols["D_high_flag"],
    ]
    _need(df, need, "6202 inputs")
    df = df.sort_values([cols["stay_id"], cols["t"]]).reset_index(drop=True)

    B_HAT, C_HAT, D_HAT, E_HAT = "B_hat", "C_hat", "D_hat", "E_hat"
    df[B_HAT] = np.nan
    df[C_HAT] = np.nan
    df[D_HAT] = np.nan
    df[E_HAT] = np.nan

    b0 = float(eff["B"]["baseline"])
    c0 = float(eff["C"]["baseline"])
    d0 = float(eff["D"]["baseline"])
    e0 = float(eff["E"]["baseline"])

    kA_B = float(eff["B"].get("kappa_A", 0.0))

    kB_C = float(eff["C"].get("kappa_B", 0.0))
    kA_C = float(eff["C"].get("kappa_A", 0.0))

    kC_D = float(eff["D"].get("kappa_C", 0.0))
    kB_D = float(eff["D"].get("kappa_B", 0.0))
    kA_D = float(eff["D"].get("kappa_A", 0.0))

    kD_E = float(eff["E"].get("kappa_D", 0.0))
    kC_E = float(eff["E"].get("kappa_C", 0.0))
    kB_E = float(eff["E"].get("kappa_B", 0.0))
    kA_E = float(eff["E"].get("kappa_A", 0.0))

    ID, T = cols["stay_id"], cols["t"]
    Aflag, Bflag, Cflag, Dflag = cols["A_low_flag"], cols["B_on_flag"], cols["C_low_flag"], cols["D_high_flag"]

    for sid, g in df.groupby(ID, sort=False):
        idx = g.index.to_numpy()
        A_prev = g[Aflag].to_numpy(np.float64, copy=False)
        B_prev = g[Bflag].to_numpy(np.float64, copy=False)
        C_prev = g[Cflag].to_numpy(np.float64, copy=False)
        D_prev = g[Dflag].to_numpy(np.float64, copy=False)

        A_lag = np.roll(A_prev, 1); A_lag[0] = 0.0
        B_lag = np.roll(B_prev, 1); B_lag[0] = 0.0
        C_lag = np.roll(C_prev, 1); C_lag[0] = 0.0
        D_lag = np.roll(D_prev, 1); D_lag[0] = 0.0

        B_hat = b0 + kA_B * A_lag

        B_hat_lag = np.roll(B_hat, 1); B_hat_lag[0] = b0
        C_hat = c0 + kB_C * B_hat_lag + kA_C * A_lag

        C_hat_lag = np.roll(C_hat, 1); C_hat_lag[0] = c0
        D_hat = d0 + kC_D * C_hat_lag + kB_D * B_hat_lag + kA_D * A_lag

        D_hat_lag = np.roll(D_hat, 1); D_hat_lag[0] = d0
        E_hat = e0 + kD_E * D_hat_lag + kC_E * C_hat_lag + kB_E * B_hat_lag + kA_E * A_lag

        df.loc[idx, B_HAT] = B_hat
        df.loc[idx, C_HAT] = C_hat
        df.loc[idx, D_HAT] = D_hat
        df.loc[idx, E_HAT] = _clip01(E_hat)

    df.to_csv(out_csv, index=False)
    _log(f"wrote: {out_csv}")
    _log(f"done, elapsed={time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
