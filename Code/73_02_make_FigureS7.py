import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

IN_DIR = Path("outputs/73_run")
F_B = IN_DIR / "7301_inst_B.csv"
F_C = IN_DIR / "7301_inst_C.csv"
F_D = IN_DIR / "7301_inst_D.csv"
F_E = IN_DIR / "7301_cumu_E.csv"
OUT1_PNG = IN_DIR / "Supp_FigureS7_1_recursive.png"
OUT1_PDF = IN_DIR / "Supp_FigureS7_1_recursive.pdf"
OUT2_PNG = IN_DIR / "Supp_FigureS7_2_multicause.png"
OUT2_PDF = IN_DIR / "Supp_FigureS7_2_multicause.pdf"

DPI = 300
FS_LET, FS_TITLE, FS_AXIS, FS_TICK = 14, 11, 9, 8
LW = 1.6

COLS = {
    "A": "#0072B2",
    "B": "#E69F00",
    "C": "#009E73",
    "D": "#CC79A7",
    "E": "#9467bd",
    "F": "#7f7f7f"
}

def need(df, cols, name):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"missing {miss} in {name}")

def panel_letter(ax, ch):
    ax.text(-0.12, 1.02, ch, transform=ax.transAxes, fontsize=FS_LET, fontweight="bold", va="bottom")

def grid(ax):
    ax.grid(True, ls=":", lw=0.7, alpha=0.6)
    ax.tick_params(labelsize=FS_TICK)

def load_inst(fp, del_col):
    df = pd.read_csv(fp)
    need(df, ["lag", "Full_main", del_col], fp.name)
    out = df.copy()
    out["delta"] = df["Full_main"] - df[del_col]
    return out[["lag", "delta"]]

def main():
    b = load_inst(F_B, "NoAtoB")
    c = load_inst(F_C, "NoBtoC")
    d = load_inst(F_D, "NoCtoD")

    fig, axs = plt.subplots(1, 3, figsize=(8.2, 2.8), dpi=DPI)
    for ax, df, title, col, let in [
        (axs[0], b, "B | Δ under remove A→B", COLS["A"], "A"),
        (axs[1], c, "C | Δ under remove B→C", COLS["B"], "B"),
        (axs[2], d, "D | Δ under remove C→D", COLS["C"], "C"),
    ]:
        ax.axvline(0, ls="--", lw=0.9, color="k", alpha=0.6)
        ax.plot(df["lag"], df["delta"], color=col, lw=LW)
        ax.set_xlabel("lag (month)", fontsize=FS_AXIS)
        ax.set_ylabel("instantaneous Δ", fontsize=FS_AXIS)
        ax.set_title(title, fontsize=FS_TITLE, pad=2)
        grid(ax)
        panel_letter(ax, let)

    fig.tight_layout()
    fig.savefig(OUT1_PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT1_PDF, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    e = pd.read_csv(F_E)
    need(e, ["lag", "Full_main", "NoAtoE", "NoBtoE", "NoCtoE", "NoDtoE", "NoneE"], F_E.name)
    curves = {
        "NoA→E": e["Full_main"] - e["NoAtoE"],
        "NoB→E": e["Full_main"] - e["NoBtoE"],
        "NoC→E": e["Full_main"] - e["NoCtoE"],
        "NoD→E": e["Full_main"] - e["NoDtoE"],
        "NoneE (remove all)": e["Full_main"] - e["NoneE"],
    }

    fig, ax = plt.subplots(1, 1, figsize=(8.4, 2.6), dpi=DPI)
    ax.axvline(0, ls="--", lw=0.9, color="k", alpha=0.6)
    for (lab, col) in zip(curves.keys(), [COLS["A"], COLS["B"], COLS["C"], COLS["F"], COLS["D"]]):
        ax.plot(e["lag"], curves[lab], lw=LW, label=lab, color=col)
    ax.set_xlabel("lag (month)", fontsize=FS_AXIS)
    ax.set_ylabel("cumulative Δ", fontsize=FS_AXIS)
    ax.set_title("E | multi-cause convergence (cumulative Δ)", fontsize=FS_TITLE, pad=2)
    grid(ax)
    panel_letter(ax, "D")

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.03, 0.95),
        frameon=False,
        fontsize=FS_TICK,
        ncol=1,
        handlelength=2.0,
        borderaxespad=0.0
    )

    fig.tight_layout()
    fig.savefig(OUT2_PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT2_PDF, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

if __name__ == "__main__":
    main()
