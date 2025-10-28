
# ===== paths & files =====
from pathlib import Path
OUT_DIR = Path("outputs/53_C3"); OUT_DIR.mkdir(parents=True, exist_ok=True)
F_OBS       = OUT_DIR / "53_observed.csv"
F_DO_FULL   = OUT_DIR / "53_do__Full_main.csv"
F_DO_NOA2B  = OUT_DIR / "53_do__NoAtoB.csv"
F_DO_NOB2C  = OUT_DIR / "53_do__NoBtoC.csv"
F_DO_NOC2D  = OUT_DIR / "53_do__NoCtoD.csv"
F_DO_NONE   = OUT_DIR / "53_do__None.csv"

# ===== params =====
BASE_B = 0.50; BASE_C = 0.40; BASE_D = 0.30; BASE_E = 0.20
PHI_B  = 0; PHI_C  = 0; PHI_D  = 0; PHI_E  = 0
K_AB = 0.60; K_CB = 0.50; K_CA = 0.20; K_DC = 0.55; K_DB = 0.20; K_DA = 0.10
K_ED = 0.60; K_EC = 0.20; K_EB = 0.10; K_EA = 0.00
TONIC_B = 0.05; TONIC_C = 0.04; TONIC_D = 0.03; TONIC_E = 0.02

T_COL="t"; A_FLAG="A_low"
B_CF="B_hat_cf"; C_CF="C_hat_cf"; D_CF="D_hat_cf"; E_CF="E_hat_cf"

# ===== helpers =====
import pandas as pd
from tqdm import tqdm

def simulate_with_A(A_series, cut_AtoB=False, cut_BtoC=False, cut_CtoD=False):

    T = len(A_series) - 1
    B = BASE_B; C = BASE_C; D = BASE_D; E = BASE_E
    rows = []
    rows.append((0, B, C, D, E))  # t=0
    for t in range(1, T+1):
        A_prev = float(A_series.iloc[t-1])
        B = BASE_B + (0.0 if cut_AtoB else K_AB)*A_prev + TONIC_B
        C = BASE_C + (0.0 if cut_BtoC else K_CB)*B + K_CA*A_prev + TONIC_C
        D = BASE_D + (0.0 if cut_CtoD else K_DC)*C + K_DB*B + K_DA*A_prev + TONIC_D
        E = BASE_E + K_ED*D + K_EC*C + K_EB*B + K_EA*A_prev + TONIC_E
        rows.append((t, B, C, D, E))
    return pd.DataFrame(rows, columns=[T_COL, B_CF, C_CF, D_CF, E_CF])


# ===== main =====
if __name__ == "__main__":
    obs = pd.read_csv(F_OBS)
    A_series = obs.set_index(T_COL)[A_FLAG]

    jobs = [
        ("Full_main", False, False, False, F_DO_FULL),
        ("NoAtoB",    True,  False, False, F_DO_NOA2B),
        ("NoBtoC",    False, True,  False, F_DO_NOB2C),
        ("NoCtoD",    False, False, True,  F_DO_NOC2D),
        ("None",      True,  True,  True,  F_DO_NONE),
    ]
    for name, cutA, cutB, cutC, outp in tqdm(jobs, desc="ablations", ncols=80):
        simulate_with_A(A_series, cutA, cutB, cutC).to_csv(outp, index=False)
    print("[53] counterfactuals exported:", [x[4].name for x in jobs])
