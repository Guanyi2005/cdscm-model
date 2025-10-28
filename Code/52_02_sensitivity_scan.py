
# ===== paths & files =====
from pathlib import Path
OUT_DIR = Path("outputs/52_C2"); OUT_DIR.mkdir(parents=True, exist_ok=True)
F_PIVOT_CSV = OUT_DIR / "sensitivity_52_pivot.csv"

# ===== params =====
SEED = 1
T_HORIZON = 30
ALPHA = 0.8
BETA = 0.2
THETA1 = 1.0
THETA2 = 0.5
S_MEAN = 2.2
S_AR = 0.85
S_NOISE_STD = 0.15
SIGMA_Y_LIST = [0.02, 0.05, 0.1, 0.2, 0.3]
SIGMA_Z_LIST = [0.02, 0.05, 0.1, 0.2, 0.3]

# ===== helpers =====
import numpy as np
import pandas as pd
from tqdm import tqdm

def f_I(x): return np.tanh(x)
def f_Y(x): return x**2

def gen_S(T, seed):
    rng = np.random.default_rng(seed)
    S = np.zeros(T+1); S[0] = S_MEAN
    for t in range(1, T+1):
        S[t] = S_MEAN*(1 - S_AR) + S_AR*S[t-1] + rng.normal(0.0, S_NOISE_STD)
    return S

def sim_ordered(T, sigma_y, sigma_z, seed=SEED):
    rng = np.random.default_rng(seed)
    S = gen_S(T, seed)
    I = np.zeros(T+1); Y = np.zeros(T+1); Z = np.zeros(T+1); C = np.zeros(T+1, dtype=int)
    for t in range(T):
        I[t+1] = ALPHA*S[t] + BETA*C[t]
        Y[t+1] = f_I(I[t]) + rng.normal(0.0, sigma_y)
        Z[t+1] = f_Y(Y[t]) + rng.normal(0.0, sigma_z)
        C[t+1] = 1 if (Z[t] > THETA1 and Y[t] > THETA2) else C[t]
    return pd.DataFrame({"I": I, "Y": Y, "Z": Z})

def variability_std(df):
    return float(np.nanstd(df[["I","Y","Z"]].values))

# ===== main =====
if __name__ == "__main__":
    print("[52] start sensitivity scan …")
    results = []
    for sy in tqdm(SIGMA_Y_LIST, desc="sigma_y"):
        row = {}
        for sz in SIGMA_Z_LIST:
            df = sim_ordered(T=T_HORIZON, sigma_y=sy, sigma_z=sz)
            row[sz] = variability_std(df)
        results.append(row)
    pv = pd.DataFrame(results, index=SIGMA_Y_LIST)
    pv.index.name = "sigma_y"
    pv.to_csv(F_PIVOT_CSV)
    print(f"[52] sensitivity pivot saved: {F_PIVOT_CSV}")
