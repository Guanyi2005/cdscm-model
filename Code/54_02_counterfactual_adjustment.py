
# ===== paths & files =====
from pathlib import Path
OUT_DIR=Path("outputs/54_C4"); OUT_DIR.mkdir(parents=True,exist_ok=True)
F_OBS_LOW = OUT_DIR/"54_observed_A_low.csv"
F_OBS_HIGH= OUT_DIR/"54_observed_A_high.csv"
F_DO_LOW  = [OUT_DIR/"54_low__Full_main.csv", OUT_DIR/"54_low__NoAtoB.csv", OUT_DIR/"54_low__NoBtoC.csv", OUT_DIR/"54_low__NoCtoD.csv", OUT_DIR/"54_low__None.csv"]
F_DO_HIGH = [OUT_DIR/"54_high__Full_main.csv",OUT_DIR/"54_high__NoAtoB.csv",OUT_DIR/"54_high__NoBtoC.csv",OUT_DIR/"54_high__NoCtoD.csv",OUT_DIR/"54_high__None.csv"]

# ===== params =====
BASE_B=0.50; BASE_C=0.40; BASE_D=0.30; BASE_E=0.20
PHI_B=0; PHI_C=0; PHI_D=0; PHI_E=0
K_AB=0.60; K_CB=0.50; K_CA=0.20; K_DC=0.55; K_DB=0.20; K_DA=0.10
K_ED=0.60; K_EC=0.20; K_EB=0.10; K_EA=0.00
TONIC_B=0.05; TONIC_C=0.04; TONIC_D=0.03; TONIC_E=0.02
T_COL="t"; A_COL="A_flag"
B_CF="B_hat_cf"; C_CF="C_hat_cf"; D_CF="D_hat_cf"; E_CF="E_hat_cf"

# ===== helpers =====
import pandas as pd
from tqdm import tqdm

def simulate_with_A(A_series, cut_AtoB=False, cut_BtoC=False, cut_CtoD=False):
    T=len(A_series)-1
    B=BASE_B; C=BASE_C; D=BASE_D; E=BASE_E
    rows=[(0,B,C,D,E)]
    for t in range(1,T+1):
        A_prev=float(A_series.iloc[t-1])
        B=BASE_B+PHI_B*(B-BASE_B)+(0.0 if cut_AtoB else K_AB)*A_prev+TONIC_B
        C=BASE_C+PHI_C*(C-BASE_C)+(0.0 if cut_BtoC else K_CB)*B+K_CA*A_prev+TONIC_C
        D=BASE_D+PHI_D*(D-BASE_D)+(0.0 if cut_CtoD else K_DC)*C+K_DB*B+K_DA*A_prev+TONIC_D
        E=BASE_E+PHI_E*(E-BASE_E)+K_ED*D+K_EC*C+K_EB*B+K_EA*A_prev+TONIC_E
        rows.append((t,B,C,D,E))
    return pd.DataFrame(rows,columns=[T_COL,B_CF,C_CF,D_CF,E_CF])

def run_one_condition(obs_path, outs):
    obs=pd.read_csv(obs_path)
    if T_COL not in obs.columns or A_COL not in obs.columns: raise RuntimeError(f"[54_02] missing columns in {obs_path}")
    A_series=obs.set_index(T_COL)[A_COL]
    jobs=[("Full_main",False,False,False,outs[0]),
          ("NoAtoB",True,False,False,outs[1]),
          ("NoBtoC",False,True,False,outs[2]),
          ("NoCtoD",False,False,True,outs[3]),
          ("None",True,True,True,outs[4])]
    for name,cA,cB,cC,outp in tqdm(jobs,desc=f"{obs_path.name}",ncols=80):
        df=simulate_with_A(A_series,cA,cB,cC)
        if df.empty or E_CF not in df.columns: raise RuntimeError(f"[54_02] bad df for {name}")
        df.to_csv(outp,index=False)

# ===== main =====
if __name__=="__main__":
    run_one_condition(F_OBS_LOW, F_DO_LOW)
    run_one_condition(F_OBS_HIGH,F_DO_HIGH)
    print("[54_02] exported counterfactuals for low/high A")
