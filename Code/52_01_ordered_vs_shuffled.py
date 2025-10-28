

# ===== paths & files =====
from pathlib import Path
OUT_DIR = Path(__file__).parent / "outputs" / "52_C2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

F_SERIES_BASE   = OUT_DIR / "series_52_baseline.csv"
F_SERIES_DO     = OUT_DIR / "series_52_do.csv"
F_TABLE_METRICS = OUT_DIR / "table_52_metrics.csv"
F_FIG_A_PNG     = OUT_DIR / "fig_52a_baseline.png"
F_FIG_A_PDF     = OUT_DIR / "fig_52a_baseline.pdf"
F_FIG_B_PNG     = OUT_DIR / "fig_52b_do.png"
F_FIG_B_PDF     = OUT_DIR / "fig_52b_do.pdf"
F_FIG_C_PNG     = OUT_DIR / "fig_52c_diff.png"
F_FIG_C_PDF     = OUT_DIR / "fig_52c_diff.pdf"
F_CONFIG        = OUT_DIR / "config_52.json"
F_LOG           = OUT_DIR / "run_52.log"

# ===== params =====
SEED = 1
T_HORIZON = 30
ALPHA = 0.8
BETA = 0.2
THETA1 = 1.0
THETA2 = 0.5
SIGMA_Y = 0.05
SIGMA_Z = 0.08
S_MEAN = 2.2
S_AR = 0.85
S_NOISE_STD = 0.15
DO_T = 13          # intervention time index (affects t+1)
DO_DELTA_I = 0.6    

# ===== helpers =====
import json, time
import numpy as np
import pandas as pd

def f_I(x): return np.tanh(x)   
def f_Y(x): return x**2         

def gen_S(T, seed):
    rng = np.random.default_rng(seed)
    S = np.zeros(T+1); S[0] = S_MEAN
    for t in range(1, T+1):
        S[t] = S_MEAN*(1 - S_AR) + S_AR*S[t-1] + rng.normal(0.0, S_NOISE_STD)
    return S

def sim_baseline(T=T_HORIZON, seed=SEED):
    rng = np.random.default_rng(seed)
    S = gen_S(T, seed)
    I = np.zeros(T+1); Y = np.zeros(T+1); Z = np.zeros(T+1); C = np.zeros(T+1, dtype=int)
    for t in range(T):
        I[t+1] = ALPHA*S[t] + BETA*C[t]
        Y[t+1] = f_I(I[t]) + rng.normal(0.0, SIGMA_Y)
        Z[t+1] = f_Y(Y[t]) + rng.normal(0.0, SIGMA_Z)
        C[t+1] = 1 if (Z[t] > THETA1 and Y[t] > THETA2) else C[t]
    return pd.DataFrame({"t": np.arange(T+1), "S": S, "I": I, "Y": Y, "Z": Z, "C": C})

def sim_do(T=T_HORIZON, seed=SEED):
    rng = np.random.default_rng(seed)
    S = gen_S(T, seed)
    I = np.zeros(T+1); Y = np.zeros(T+1); Z = np.zeros(T+1); C = np.zeros(T+1, dtype=int)
    for t in range(T):
        I[t+1] = ALPHA*S[t] + BETA*C[t]
        if t == DO_T:
            I[t+1] = I[t+1] + DO_DELTA_I   
        Y[t+1] = f_I(I[t]) + rng.normal(0.0, SIGMA_Y)
        Z[t+1] = f_Y(Y[t]) + rng.normal(0.0, SIGMA_Z)
        C[t+1] = 1 if (Z[t] > THETA1 and Y[t] > THETA2) else C[t]
    return pd.DataFrame({"t": np.arange(T+1), "S": S, "I": I, "Y": Y, "Z": Z, "C": C})

def variability_std(df):
    return float(np.nanstd(df[["I","Y","Z"]].values))

# ===== main =====
if __name__ == "__main__":
    t0 = time.time()
    df_base = sim_baseline()
    df_do   = sim_do()

    df_base.to_csv(F_SERIES_BASE, index=False)
    df_do.to_csv(F_SERIES_DO, index=False)

    dI = df_do["I"].to_numpy() - df_base["I"].to_numpy()
    dY = df_do["Y"].to_numpy() - df_base["Y"].to_numpy()
    dZ = df_do["Z"].to_numpy() - df_base["Z"].to_numpy()

    m_base = variability_std(df_base)
    m_do   = variability_std(df_do)

    pd.DataFrame([{
        "baseline_std": m_base,
        "do_std": m_do,
        "delta_std": m_do - m_base,
        "max_abs_delta_I": float(np.max(np.abs(dI))),
        "max_abs_delta_Y": float(np.max(np.abs(dY))),
        "max_abs_delta_Z": float(np.max(np.abs(dZ))),
        "do_time": DO_T,
        "do_delta_I": DO_DELTA_I
    }]).to_csv(F_TABLE_METRICS, index=False)

    with open(F_CONFIG, "w", encoding="utf-8") as f:
        json.dump({
            "seed": SEED, "T": T_HORIZON,
            "alpha": ALPHA, "beta": BETA,
            "theta1": THETA1, "theta2": THETA2,
            "sigma_y": SIGMA_Y, "sigma_z": SIGMA_Z,
            "S_mean": S_MEAN, "S_ar": S_AR, "S_noise": S_NOISE_STD,
            "do_t": DO_T, "do_delta_I": DO_DELTA_I
        }, f, ensure_ascii=False, indent=2)

    print(f"[52] data exported: baseline/do series and metrics. ({time.time()-t0:.2f}s)")
