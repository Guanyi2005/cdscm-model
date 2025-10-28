import pandas as pd
import matplotlib.pyplot as plt

IN_ADJ = "outputs/64_run/6403_adjustment_vs_do.csv"
IN_ADD = "outputs/64_run/64_additivity_by_lag.csv"
IN_COV = "outputs/64_run/64_coverage_by_lag.csv"

OUT_FIG = "outputs/64_run/Figure6.png"
OUT_PDF = "outputs/64_run/Figure6.pdf"
OUT_SVG = "outputs/64_run/Figure6.svg"

LAG_MIN = 0
LAG_MAX = 30
COL_T = "t"

COLOR_RESID = "#9467bd"
COLOR_COV = "#1f77b4"
LW_MAIN = 2.0

def _need_cols(df, cols, path):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"missing columns in {path}: {miss}")

def _clip_xy(df, xcol):
    g = df.copy()
    g[xcol] = pd.to_numeric(g[xcol], errors="coerce")
    return g[(g[xcol] >= LAG_MIN) & (g[xcol] <= LAG_MAX)].sort_values(xcol)

def main():
    a = pd.read_csv(IN_ADJ)
    b = pd.read_csv(IN_ADD)
    c = pd.read_csv(IN_COV)

    _need_cols(a, [COL_T, "obs_cum", "adj_cum", "do_cum"], IN_ADJ)
    _need_cols(b, ["lag", "resid_mean"], IN_ADD)
    _need_cols(c, ["lag", "n_stays_anchor", "n_stays_effective"], IN_COV)

    a = _clip_xy(a, COL_T)
    b = _clip_xy(b, "lag")
    c = _clip_xy(c, "lag")

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))

    ax[0].plot(a[COL_T], a["obs_cum"], label="Observed", lw=LW_MAIN)
    ax[0].plot(a[COL_T], a["adj_cum"], label="Adjusted", lw=LW_MAIN)
    ax[0].plot(a[COL_T], a["do_cum"],  label="do(A=0)", lw=LW_MAIN)
    ax[0].axvline(0, ls="--", lw=0.8, color="gray")
    ax[0].set_xlim(LAG_MIN, LAG_MAX)
    ax[0].set_title("(A) Cumulative Ê — Observed / Adjusted / do(A=0)", loc="left")
    ax[0].set_xlabel("Lag (hours)")
    ax[0].set_ylabel("Cumulative level")
    ax[0].legend(frameon=False)

    ax[1].plot(b["lag"], b["resid_mean"], lw=LW_MAIN, color=COLOR_RESID)
    ax[1].axvline(0, ls="--", lw=0.8, color="gray")
    ax[1].set_xlim(LAG_MIN, LAG_MAX)
    ax[1].set_title("(B) Additivity residual — Relative", loc="left")
    ax[1].set_xlabel("Lag (hours)")
    ax[1].set_ylabel("Residual mean")

    inset = ax[1].inset_axes([0.60, 0.52, 0.37, 0.40])
    inset.plot(c["lag"], c["n_stays_anchor"],    lw=1.6, ls="--", color=COLOR_COV, label="Anchor (constant)")
    inset.plot(c["lag"], c["n_stays_effective"], lw=2.2, color=COLOR_COV,           label="Effective (obs ∩ do)")
    inset.set_xlim(LAG_MIN, LAG_MAX)
    inset.set_title("Coverage", fontsize=9, pad=2)
    inset.tick_params(labelsize=8)
    inset.grid(True, ls=":", lw=0.6, alpha=0.6)
    inset.legend(frameon=False, fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=300)
    fig.savefig(OUT_PDF, dpi=300)
    fig.savefig(OUT_SVG, dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()
