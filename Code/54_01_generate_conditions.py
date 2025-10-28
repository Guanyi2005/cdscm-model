
# ===== paths & files =====
from pathlib import Path
OUT_DIR = Path("outputs/54_C4"); OUT_DIR.mkdir(parents=True, exist_ok=True)
F_OBS_LOW   = OUT_DIR / "54_observed_A_low.csv"
F_OBS_HIGH  = OUT_DIR / "54_observed_A_high.csv"
F_CFG       = OUT_DIR / "54_config.json"

# ===== params =====
SEED=1; T=48
BASE_B=0.50; BASE_C=0.40; BASE_D=0.30; BASE_E=0.20
PHI_B=0.85; PHI_C=0.85; PHI_D=0.85; PHI_E=0.85
K_AB=0.60; K_CB=0.50; K_CA=0.20; K_DC=0.55; K_DB=0.20; K_DA=0.10
K_ED=0.60; K_EC=0.20; K_EB=0.10; K_EA=0.00
TONIC_B=0.05; TONIC_C=0.04; TONIC_D=0.03; TONIC_E=0.02
A_LOW_LAG=8;  A_LOW_WIDTH=6
A_HIGH_LAG=8; A_HIGH_WIDTH=12
ID_COL="stay_id"; T_COL="t"; A_COL="A_flag"
B_HAT="B_hat"; C_HAT="C_hat"; D_HAT="D_hat"; E_HAT="E_hat"

# ===== helpers =====
import json, numpy as np, pandas as pd

def gen_A(T,on,width,level=1.0):
    a=np.zeros(T+1,dtype=float); a[on:on+width]=level; return a

def simulate(A):
    B=BASE_B; C=BASE_C; D=BASE_D; E=BASE_E
    rows=[(0,A[0],B,C,D,E)]
    for t in range(1,T+1):
        A_prev=float(A[t-1])
        B=BASE_B + K_AB*A_prev + TONIC_B
        C=BASE_C + K_CB*B       + K_CA*A_prev + TONIC_C
        D=BASE_D + K_DC*C       + K_DB*B      + K_DA*A_prev + TONIC_D
        E=BASE_E + K_ED*D       + K_EC*C      + K_EB*B      + K_EA*A_prev + TONIC_E
        rows.append((t,A[t],B,C,D,E))
    df=pd.DataFrame(rows,columns=[T_COL,A_COL,B_HAT,C_HAT,D_HAT,E_HAT]); df[ID_COL]=1
    return df[[ID_COL,T_COL,A_COL,B_HAT,C_HAT,D_HAT,E_HAT]]

# ===== main =====
if __name__=="__main__":
    np.random.default_rng(SEED)
    A_low = gen_A(T,A_LOW_LAG,A_LOW_WIDTH,1.0)
    A_high= gen_A(T,A_HIGH_LAG,A_HIGH_WIDTH,1.0)
    simulate(A_low ).to_csv(F_OBS_LOW, index=False)
    simulate(A_high).to_csv(F_OBS_HIGH,index=False)
    Path(F_CFG).write_text(json.dumps({
        "T":T,"phi":{"B":PHI_B,"C":PHI_C,"D":PHI_D,"E":PHI_E},
        "k":{"AB":K_AB,"CB":K_CB,"CA":K_CA,"DC":K_DC,"DB":K_DB,"DA":K_DA,"ED":K_ED,"EC":K_EC,"EB":K_EB,"EA":K_EA},
        "tonic":{"B":TONIC_B,"C":TONIC_C,"D":TONIC_D,"E":TONIC_E},
        "A_windows":{"low":[A_LOW_LAG,A_LOW_WIDTH],"high":[A_HIGH_LAG,A_HIGH_WIDTH]},
        "columns":{"id":ID_COL,"t":T_COL,"A":A_COL,"B":B_HAT,"C":C_HAT,"D":D_HAT,"E":E_HAT}
    },ensure_ascii=False,indent=2),encoding="utf-8")
    print("[54_01] exported:", F_OBS_LOW.name, F_OBS_HIGH.name)
