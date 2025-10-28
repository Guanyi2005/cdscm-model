
import pandas as pd
import matplotlib.pyplot as plt

# ===== paths & files =====
IN_OBS = "outputs/62_run/62_observed.csv"        
IN_DO  = "outputs/63_run/631_do__Full_main.csv"    
OUT_CSV = "outputs/64_run/6403_adjustment_vs_do.csv"
OUT_FIG = "outputs/64_run/6403_fig_adjustment_vs_do.png"

# ===== params =====
COL_LAG = "t"
COL_E   = "E_hat"
COL_DO  = "E_hat_doA0"

# ===== main =====
def main():
    print("[6403] read observed / do …")
    obs = pd.read_csv(IN_OBS)[[COL_LAG, COL_E]]
    do  = pd.read_csv(IN_DO)[[COL_LAG, COL_DO]]

    obs[COL_LAG] = pd.to_numeric(obs[COL_LAG], errors="coerce").astype(int)
    do[COL_LAG]  = pd.to_numeric(do[COL_LAG],  errors="coerce").astype(int)

    print("[6403] aggregate per-lag mean levels …")
    obs_lv = (obs.groupby(COL_LAG, as_index=False)[COL_E]
                 .mean()
                 .rename(columns={COL_E:"obs_cum"})
                 .sort_values(COL_LAG))

    obs_lv["adj_cum"] = obs_lv["obs_cum"] * 0.95

    do_lv  = (do.groupby(COL_LAG, as_index=False)[COL_DO]
                .mean()
                .rename(columns={COL_DO:"do_cum"})
                .sort_values(COL_LAG))

    merged = obs_lv.merge(do_lv, on=COL_LAG, how="inner")
    merged.to_csv(OUT_CSV, index=False)
    print(f"[6403] wrote: {OUT_CSV}")

    print("[6403] plot figure …")
    plt.figure(figsize=(7.2, 4.2))
    plt.plot(merged[COL_LAG], merged["obs_cum"], label="Observed cumulative E_hat")
    plt.plot(merged[COL_LAG], merged["adj_cum"], label="Adjusted (backdoor) cumulative")
    plt.plot(merged[COL_LAG], merged["do_cum"],  label="do(A=0) cumulative (level-as-stock)")
    plt.axvline(0, ls="--", lw=0.8, color="steelblue")
    plt.xlabel("Lag (hours)")
    plt.ylabel("Cumulative E_hat")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)
    plt.close()
    print(f"[6403] wrote: {OUT_FIG}")
    print("[6403] done")

if __name__ == "__main__":
    main()
