
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ===== paths & files =====
RUN72 = "outputs/72_run"
RUN73 = "outputs/73_run"
OUT_DIR = "outputs/figures"
OUT_PNG = os.path.join(OUT_DIR, "Figure7.png")
OUT_PDF = os.path.join(OUT_DIR, "Figure7.pdf")
OUT_SVG = os.path.join(OUT_DIR, "Figure7.svg")

F_AB   = os.path.join(RUN72, "7201_chain_AB.csv")
F_BC   = os.path.join(RUN72, "7201_chain_BC.csv")
F_CD   = os.path.join(RUN72, "7201_chain_CD.csv")
F_DE   = os.path.join(RUN72, "7201_chain_DE.csv")
F_RHO  = os.path.join(RUN72, "7201_spectral_radius.csv")
F_ECUMU= os.path.join(RUN73, "7301_cumu_E.csv")

# ===== params =====
TITLE = "External validation on the housing chain: ordered propagation and multi-cause convergence"
SMALL_PANEL = "rho"          
FIGSIZE = (15.2, 8.8)
FONT_SIZE = 12
WSPACE, HSPACE = 0.18, 0.38 
PAD_FRAC = 0.20

# ===== utils =====
def need_cols(df: pd.DataFrame, cols, name: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"missing columns {miss} in {name}")

def smooth3(a):
    return pd.Series(a).rolling(3, center=True, min_periods=1).mean().to_numpy()

def centered_ylim(ax, y, pad_frac=PAD_FRAC):
    y = np.asarray(y, dtype=float)
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    rng = max(1e-9, y_max - y_min)
    pad = pad_frac * rng
    ax.set_ylim(y_min - pad, y_max + pad)

# ===== plotting =====
def plot_chain(ax, csv_path, title, show_ylabel=False):
    df = pd.read_csv(csv_path).sort_values("lag")
    need_cols(df, ["lag","mean_ord","mean_shf","delta"], os.path.basename(csv_path))
    x  = df["lag"].to_numpy()
    mo = smooth3(df["mean_ord"].to_numpy())
    ms = smooth3(df["mean_shf"].to_numpy())
    dl = smooth3(df["delta"].to_numpy())

    ax.plot(x, mo, lw=1.8, label="ordered")
    ax.plot(x, ms, lw=1.8, ls="--", label="shuffled")
    ax.plot(x, dl, lw=1.8, color="tab:green", label="delta")
    ax.axvline(0, color="k", ls=":", lw=1)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("lag (month)")
    if show_ylabel:
        ax.set_ylabel("rate")
    centered_ylim(ax, np.r_[mo, ms, dl])
    ax.grid(ls=":", alpha=0.5)

def plot_multicause_E(ax):
    df = pd.read_csv(F_ECUMU)
    need_cols(df, ["lag","Full_main","NoneE"], os.path.basename(F_ECUMU))

    x     = df["lag"].to_numpy()
    full  = df["Full_main"].to_numpy()
    none  = df["NoneE"].to_numpy()

    delta = full - none

    ax.axvline(0, color="k", ls=":", lw=1)
    ax.axhline(0, color="0.6", ls=":", lw=0.8)

    ax.plot(x, delta, lw=1.8, color="#ff7f0e", label="Full − NoneE")

    ax.set_title("E | multi-cause convergence (cumulative Δ)", loc="left", fontweight="bold")
    ax.set_xlabel("lag (month)")
    ax.set_ylabel("cumulative Δ (Full − NoneE)")

    centered_ylim(ax, delta)
    ax.grid(ls=":", alpha=0.5)
    ax.legend(fontsize=9, frameon=False, loc="upper left")

def plot_small_rho(ax):
    df = pd.read_csv(F_RHO)
    need_cols(df, ["t_mid","rho"], os.path.basename(F_RHO))
    t = df["t_mid"].to_numpy()
    r = df["rho"].to_numpy()

    ax.plot(t, r, lw=1.8)
    ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
    ax.set_title("F | stability (ρ)", loc="left", fontweight="bold")
    ax.set_xlabel("time (month)")
    ax.set_ylabel("spectral radius ρ")
    r_max = float(np.nanmax(r)) if len(r) else 1.0
    top = max(1.0, r_max * 1.25)
    bot = -0.20 * max(r_max, 1.0)
    ax.set_ylim(bot, top)
    ax.grid(ls=":", alpha=0.5)

def plot_small_terminal(ax):
    df = pd.read_csv(os.path.join(RUN72, "7201_anchors_A.csv"))
    if not set(["t","anchor_t"]).issubset(df.columns):
        ax.text(0.5, 0.5, "terminal panel placeholder", ha="center", va="center")
        ax.set_axis_off()
        return
    t = df["t"].to_numpy()
    e = np.cumsum(df["anchor_t"].to_numpy())
    ax.plot(t, e, color="tab:green", lw=1.8)
    ax.set_title("terminal monotonicity", loc="left", fontweight="bold")
    ax.set_xlabel("time (month)")
    ax.set_ylabel("cumulative E")
    centered_ylim(ax, e)
    ax.grid(ls=":", alpha=0.5)

# ===== main =====
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "axes.titlepad": 8,
        "axes.labelpad": 6
    })

    fig = plt.figure(figsize=FIGSIZE, constrained_layout=False)
    gs = GridSpec(2, 4, figure=fig, height_ratios=[0.9, 1.1], wspace=WSPACE, hspace=HSPACE)

    axAB = fig.add_subplot(gs[0, 0]); plot_chain(axAB, F_AB, "A  A→B", show_ylabel=True)
    axBC = fig.add_subplot(gs[0, 1]); plot_chain(axBC, F_BC, "B  B→C")
    axCD = fig.add_subplot(gs[0, 2]); plot_chain(axCD, F_CD, "C  C→D")
    axDE = fig.add_subplot(gs[0, 3]); plot_chain(axDE, F_DE, "D  D→E")

    axE  = fig.add_subplot(gs[1, 0:3]); plot_multicause_E(axE)
    axF  = fig.add_subplot(gs[1, 3])
    if SMALL_PANEL == "rho":
        plot_small_rho(axF)
    else:
        plot_small_terminal(axF)

    fig.suptitle(TITLE, y=0.965, fontsize=16, fontweight="bold")

    fig.subplots_adjust(left=0.05, right=0.995, bottom=0.06, top=0.88,
                        wspace=WSPACE, hspace=HSPACE)

    fig.savefig(OUT_PNG, dpi=300)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_SVG)
    print("wrote:", OUT_PNG)
    print("wrote:", OUT_PDF)
    print("wrote:", OUT_SVG)

if __name__ == "__main__":
    main()
