
# ===== paths & files =====
from pathlib import Path
OUT_DIR=Path("outputs/54_C4"); OUT_DIR.mkdir(parents=True,exist_ok=True)
F_DO_LOW={"Full_main":OUT_DIR/"54_low__Full_main.csv","NoAtoB":OUT_DIR/"54_low__NoAtoB.csv","NoBtoC":OUT_DIR/"54_low__NoBtoC.csv","NoCtoD":OUT_DIR/"54_low__NoCtoD.csv","None":OUT_DIR/"54_low__None.csv"}
F_DO_HIGH={"Full_main":OUT_DIR/"54_high__Full_main.csv","NoAtoB":OUT_DIR/"54_high__NoAtoB.csv","NoBtoC":OUT_DIR/"54_high__NoBtoC.csv","NoCtoD":OUT_DIR/"54_high__NoCtoD.csv","None":OUT_DIR/"54_high__None.csv"}
F_ALL_LOW = OUT_DIR/"54_ALL_low.csv"
F_ALL_HIGH= OUT_DIR/"54_ALL_high.csv"
F_SUMMARY = OUT_DIR/"54_summary_peaks.csv"

# ===== params =====
ORDER=("Full_main","NoAtoB","NoBtoC","NoCtoD","None")
T_COL="t"; E_CF="E_hat_cf"

# ===== helpers =====
import pandas as pd

def stack_condition(dmap, taglabel):
    frames=[]
    for tag in ORDER:
        f=dmap[tag]; df=pd.read_csv(f)
        if T_COL not in df.columns or E_CF not in df.columns or len(df)==0:
            raise RuntimeError(f"[54_03] missing/empty columns in {f}")
        sub=df[[T_COL,E_CF]].copy(); sub.rename(columns={E_CF:"E"},inplace=True)
        sub["tag"]=tag; sub["cond"]=taglabel; frames.append(sub)
    return pd.concat(frames,ignore_index=True)

# ===== main =====
if __name__=="__main__":
    low = stack_condition(F_DO_LOW,"A_low");  low.to_csv(F_ALL_LOW, index=False)
    high= stack_condition(F_DO_HIGH,"A_high");high.to_csv(F_ALL_HIGH,index=False)

    def peak_stat(df): return df.groupby(["cond","tag"])["E"].max().reset_index(name="E_peak")
    summary=pd.concat([peak_stat(low),peak_stat(high)],ignore_index=True)
    # strict presence check
    for cond in ["A_low","A_high"]:
        for tag in ORDER:
            if len(summary[(summary["cond"]==cond)&(summary["tag"]==tag)])!=1:
                raise RuntimeError(f"[54_03] peak missing for cond={cond}, tag={tag}")
    summary.to_csv(F_SUMMARY,index=False)
    print("[54_03] exported:", F_ALL_LOW.name, F_ALL_HIGH.name, F_SUMMARY.name)
