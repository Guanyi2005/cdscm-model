
# ===== paths & files =====
from pathlib import Path
OUT_DIR = Path("outputs/53_C3"); OUT_DIR.mkdir(parents=True, exist_ok=True)
F_OBS       = OUT_DIR / "53_observed.csv"
F_DO_FULL   = OUT_DIR / "53_do__Full_main.csv"
F_DO_NOA2B  = OUT_DIR / "53_do__NoAtoB.csv"
F_DO_NOB2C  = OUT_DIR / "53_do__NoBtoC.csv"
F_DO_NOC2D  = OUT_DIR / "53_do__NoCtoD.csv"
F_DO_NONE   = OUT_DIR / "53_do__None.csv"
F_AE_COMBO  = OUT_DIR / "53_A_to_E_counterfactual.ALL.csv"
F_BCD_INST  = OUT_DIR / "53_A_to_BCD_instant.ALL.csv"

# ===== params =====
T_COL="t"; B_HAT="B_hat"; C_HAT="C_hat"; D_HAT="D_hat"; E_CF_COL="E_hat_cf"
STACK_ORDER=("Full_main","NoAtoB","NoBtoC","NoCtoD","None")

# ===== helpers =====
import pandas as pd

# ===== main =====
if __name__ == "__main__":
    tag2file = {"Full_main":F_DO_FULL,"NoAtoB":F_DO_NOA2B,"NoBtoC":F_DO_NOB2C,"NoCtoD":F_DO_NOC2D,"None":F_DO_NONE}
    frames=[]
    for tag in STACK_ORDER:
        df = pd.read_csv(tag2file[tag])[[T_COL, E_CF_COL]].copy()
        df.rename(columns={E_CF_COL:"E"}, inplace=True)
        df["tag"]=tag
        frames.append(df)
    ae = pd.concat(frames, ignore_index=True); ae.to_csv(F_AE_COMBO, index=False)

    obs = pd.read_csv(F_OBS)[[T_COL,B_HAT,C_HAT,D_HAT]].copy()
    obs.to_csv(F_BCD_INST, index=False)
    print("[53] fig inputs exported:", F_AE_COMBO.name, F_BCD_INST.name)
