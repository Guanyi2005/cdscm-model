

# ===== paths & files =====
from pathlib import Path
OUT_DIR = Path("outputs/53_C3"); OUT_DIR.mkdir(parents=True, exist_ok=True)
F_OBS = OUT_DIR / "53_observed.csv"
F_CFG = OUT_DIR / "53_obs_config.json"

# ===== params =====
SEED        = 1
T           = 48
# baselines (non-zero)
BASE_B      = 0.50
BASE_C      = 0.40
BASE_D      = 0.30
BASE_E      = 0.20
# AR(1) coefficients
PHI_B       = 0
PHI_C       = 0
PHI_D       = 0
PHI_E       = 0
# kappas (fixed edges A→B→C→D→E with skips)
K_AB        = 0.60
K_CB        = 0.50
K_CA        = 0.20
K_DC        = 0.55
K_DB        = 0.20
K_DA        = 0.10
K_ED        = 0.60
K_EC        = 0.20
K_EB        = 0.10
K_EA        = 0.00
# constant background inputs (tonic, non-zero)
TONIC_B     = 0.05
TONIC_C     = 0.04
TONIC_D     = 0.03
TONIC_E     = 0.02
# A trigger window
A_ON_LAG    = 8
A_ON_WIDTH  = 6
# fixed column names
ID_COL      = "stay_id"
T_COL       = "t"
A_FLAG      = "A_low"
B_HAT       = "B_hat"
C_HAT       = "C_hat"
D_HAT       = "D_hat"
E_HAT       = "E_hat"

# ===== helpers =====
import json
import numpy as np
import pandas as pd

def gen_A_low(T, on=A_ON_LAG, width=A_ON_WIDTH):
    a = np.zeros(T+1, dtype=int)
    a[on:on+width] = 1
    return a

# ===== main =====
if __name__ == "__main__":
    np.random.default_rng(SEED)
    A_low = gen_A_low(T)
    B = np.zeros(T+1); C = np.zeros(T+1); D = np.zeros(T+1); E = np.zeros(T+1)
    B[0]=BASE_B; C[0]=BASE_C; D[0]=BASE_D; E[0]=BASE_E
    for t in range(1, T+1):
        A_prev = float(A_low[t-1])
        B[t] = BASE_B + K_AB*A_prev + TONIC_B
        C[t] = BASE_C + K_CB*B[t-1] + K_CA*A_prev + TONIC_C
        D[t] = BASE_D + K_DC*C[t-1] + K_DB*B[t-1] + K_DA*A_prev + TONIC_D
        E[t] = BASE_E + K_ED*D[t-1] + K_EC*C[t-1] + K_EB*B[t-1] + K_EA*A_prev + TONIC_E
    df = pd.DataFrame({T_COL:np.arange(T+1,dtype=int), A_FLAG:A_low, B_HAT:B, C_HAT:C, D_HAT:D, E_HAT:E})
    df[ID_COL] = 1
    df = df[[ID_COL, T_COL, A_FLAG, B_HAT, C_HAT, D_HAT, E_HAT]]
    df.to_csv(F_OBS, index=False)
    with open(F_CFG, "w", encoding="utf-8") as f:
        json.dump({
            "seed":SEED,"T":T,
            "phi":{"B":PHI_B,"C":PHI_C,"D":PHI_D,"E":PHI_E},
            "k":{"AB":K_AB,"CB":K_CB,"CA":K_CA,"DC":K_DC,"DB":K_DB,"DA":K_DA,"ED":K_ED,"EC":K_EC,"EB":K_EB,"EA":K_EA},
            "tonic":{"B":TONIC_B,"C":TONIC_C,"D":TONIC_D,"E":TONIC_E},
            "A_on":{"lag":A_ON_LAG,"width":A_ON_WIDTH},
            "columns":{"id":ID_COL,"t":T_COL,"A":A_FLAG,"B":B_HAT,"C":C_HAT,"D":D_HAT,"E":E_HAT}
        }, f, indent=2, ensure_ascii=False)
    print("[53] observed exported:", F_OBS)
