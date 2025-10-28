
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== paths =====
IN  = Path("outputs/62_run")
OUT = IN
OUT.mkdir(parents=True, exist_ok=True)

F_BOUNDS = IN / "6201_series_bounds.csv"
F_EON    = IN / "6201_Eon_monotonic.csv"
F_AB     = IN / "6203_chain_AB.csv"
F_BC     = IN / "6203_chain_BC.csv"
F_CD     = IN / "6203_chain_CD.csv"
F_DE     = IN / "6203_chain_DE.csv"

OUT_PNG = OUT / "Figure3.png"
OUT_PDF = OUT / "Figure3.pdf"

# ===== style =====
FIGSIZE = (6.8, 3.8)
DPI     = 300
FS_LET   = 12
FS_TITLE = 6
FS_AXIS  = 7
FS_TICK  = 6
LW_MAIN  = 1.4
LW_DELTA = 1.0
COL_ORDER = "#1f77b4"
COL_SHUF  = "#ff7f0e"
COL_DELTA = "#2ca02c"

# ===== helpers =====
def req(df, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"missing columns: {miss}")

def letter(ax, ch):
    ax.text(-0.10, 1.02, ch, transform=ax.transAxes,
            fontsize=FS_LET, fontweight="bold", va="bottom")

def chain(ax, df, title, xlim):
    df = df.sort_values("lag")
    ax.plot(df["lag"], df["mean_ord"], lw=LW_MAIN, color=COL_ORDER, label="ordered")
    ax.plot(df["lag"], df["mean_shf"], lw=LW_MAIN, ls="--", color=COL_SHUF, label="shuffled")
    ax.plot(df["lag"], df["delta"],    lw=LW_DELTA, color=COL_DELTA, label="delta")
    ax.axvline(0, ls="--", color="k", alpha=0.6, lw=0.9)
    ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=FS_TITLE, pad=3)
    ax.set_xlabel("lag (hour)", fontsize=FS_AXIS)
    ax.set_ylabel("rate", fontsize=FS_AXIS)
    ax.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax.tick_params(labelsize=FS_TICK)

def panel_A_boundedness(ax, bounds_df):
    req(bounds_df, ["var","min","p05","p50","p95","max"])
    order = ["B_hat", "C_hat", "D_hat"]
    b = bounds_df.set_index("var").loc[order].reset_index()
    y = np.arange(len(order))[::-1]
    ax.hlines(y, b["min"], b["max"], color="#444444", lw=1.2)
    ax.hlines(y, b["p05"], b["p95"], color=COL_ORDER, lw=4.0, alpha=0.9)
    ax.plot(b["p50"], y, "o", color="#000000", ms=3)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=FS_AXIS)
    ax.set_xlabel("value (normalized)", fontsize=FS_AXIS)
    ax.set_title("boundedness summary (min–p05–p50–p95–max)", fontsize=FS_TITLE, pad=3)
    ax.grid(True, axis="x", ls=":", lw=0.6, alpha=0.6)
    ax.tick_params(labelsize=FS_TICK)

# ===== main =====
def main():
    bounds = pd.read_csv(F_BOUNDS)
    eon    = pd.read_csv(F_EON)[["t","cum_mean"]]
    ab     = pd.read_csv(F_AB)
    bc     = pd.read_csv(F_BC)
    cd     = pd.read_csv(F_CD)
    de     = pd.read_csv(F_DE)

    for df in [ab, bc, cd, de]:
        req(df, ["lag","mean_ord","mean_shf","delta"])

    lag_min = int(min(ab["lag"].min(), bc["lag"].min(), cd["lag"].min(), de["lag"].min()))
    lag_max = int(max(ab["lag"].max(), bc["lag"].max(), cd["lag"].max(), de["lag"].max()))
    xlim = (lag_min, lag_max)

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    gs  = fig.add_gridspec(2, 4, height_ratios=[0.65, 0.45], wspace=0.35, hspace=0.45)

    axA = fig.add_subplot(gs[0, 0:2])
    panel_A_boundedness(axA, bounds)
    letter(axA, "A")

    axB = fig.add_subplot(gs[0, 2:4])
    axB.step(eon["t"], eon["cum_mean"], where="post", lw=LW_MAIN, color=COL_DELTA)
    axB.set_xlabel("time (hour)", fontsize=FS_AXIS)
    axB.set_ylabel("mean cumulative E_on", fontsize=FS_AXIS)
    axB.set_title("terminal monotonicity", fontsize=FS_TITLE, pad=3)
    axB.grid(True, ls=":", lw=0.6, alpha=0.6)
    axB.tick_params(labelsize=FS_TICK)
    letter(axB, "B")

    axC = fig.add_subplot(gs[1,0]); chain(axC, ab, "A→B", xlim); letter(axC, "C")
    axD = fig.add_subplot(gs[1,1]); chain(axD, bc, "B→C", xlim); letter(axD, "D")
    axE = fig.add_subplot(gs[1,2]); chain(axE, cd, "C→D", xlim); letter(axE, "E")
    axF = fig.add_subplot(gs[1,3]); chain(axF, de, "D→E", xlim); letter(axF, "F")


    for ax in [axA, axB]:
        pos = ax.get_position()
        new_pos = [pos.x0, pos.y0 + 0.025, pos.width * 0.9, pos.height * 0.9]
        ax.set_position(new_pos)

    handles = [plt.Line2D([0],[0], color=COL_ORDER, lw=LW_MAIN),
               plt.Line2D([0],[0], color=COL_SHUF,  lw=LW_MAIN, ls="--"),
               plt.Line2D([0],[0], color=COL_DELTA, lw=LW_DELTA)]
    fig.legend(handles, ["ordered","shuffled","delta"],
               loc="upper center", bbox_to_anchor=(0.5, 0.47),
               ncol=3, frameon=False, fontsize=FS_TICK)

    fig.savefig(OUT_PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_PDF, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

if __name__ == "__main__":
    main()
