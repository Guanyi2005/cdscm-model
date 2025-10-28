
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, NullLocator

plt.rcParams.update({
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})

# ===== paths & files =====
DIR63 = Path("outputs/63_run")

F_63_BCD_INST = DIR63 / "631_A_to_BCD_inst_counterfactual.ALL.csv"       
F_63_AE_CUMU  = DIR63 / "631_A_to_E_cum_counterfactual.ALL.csv"         
F_64_EONLY    = DIR63 / "632_A_to_E_cum_counterfactual.EONLY.ALL.csv"    

# fixed outputs
OUTDIR = Path("outputs/63_run")
OUTDIR.mkdir(parents=True, exist_ok=True)
F_OUT_63_PNG = OUTDIR / "Figure4.png"
F_OUT_63_PDF = OUTDIR / "Figure4.pdf"
F_OUT_63_SVG = OUTDIR / "Figure4.svg"
F_OUT_64_PNG = OUTDIR / "Figure5.png"
F_OUT_64_PDF = OUTDIR / "Figure5.pdf"
F_OUT_64_SVG = OUTDIR / "Figure5.svg"

# ===== params =====
TAGS_63 = ["Full_main", "NoAtoB", "NoBtoC", "NoCtoD"]
COLORS_63  = {"Full_main": "#0072B2", "NoAtoB": "#E69F00", "NoBtoC": "#009E73", "NoCtoD": "#D55E00"}
LSTYLES_63 = {"Full_main": "-",       "NoAtoB": "--",      "NoBtoC": "-.",      "NoCtoD": ":"}

TAGS_64 = ["NoAtoE", "NoBtoE", "NoCtoE", "NoDtoE", "NoneE"]
COLORS_64  = {"NoAtoE": "#E69F00", "NoBtoE": "#009E73", "NoCtoE": "#CC79A7", "NoDtoE": "#D55E00", "NoneE": "#0072B2"}
LSTYLES_64 = {"NoAtoE": "--",      "NoBtoE": "-.",      "NoCtoE": ":",        "NoDtoE": (0, (3,1,1,1)), "NoneE": "-"}

DPI = 300
FIGSIZE_63 = (7.09, 5.0)     
FIGSIZE_64 = (4.72, 3.2)       
TICK_X = MaxNLocator(7)
TICK_Y = MaxNLocator(6)

SMOOTH_K = 1

# ===== helpers =====
def _require_cols(df: pd.DataFrame, cols: list, path: Path):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing columns in {path}: {missing}")

def _read_63_inst(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require_cols(df, ["lag", "tag", "delta_B", "delta_C", "delta_D"], path)
    df = df[df["tag"].isin(TAGS_63)].copy()
    df["tag"] = pd.Categorical(df["tag"], categories=TAGS_63, ordered=True)
    df.sort_values(["tag", "lag"], inplace=True)
    return df

def _read_63_cumu(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"lag", "tag", "cum_delta"}.issubset(df.columns):
        _require_cols(df, ["lag", "tag", "cum_obs", "cum_cf"], path)
        df["cum_delta"] = df["cum_obs"] - df["cum_cf"]
    df = df[df["tag"].isin(TAGS_63)][["lag", "tag", "cum_delta"]].copy()
    df["tag"] = pd.Categorical(df["tag"], categories=TAGS_63, ordered=True)
    df.sort_values(["tag", "lag"], inplace=True)
    return df

def _read_64_eonly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"lag", "tag", "cum_delta"}.issubset(df.columns):
        _require_cols(df, ["lag", "tag", "cum_obs", "cum_cf"], path)
        df["cum_delta"] = df["cum_obs"] - df["cum_cf"]
    df = df[df["tag"].isin(TAGS_64)][["lag", "tag", "cum_delta"]].copy()
    df["tag"] = pd.Categorical(df["tag"], categories=TAGS_64, ordered=True)
    df.sort_values(["tag", "lag"], inplace=True)
    return df

def _style_axes(ax):
    ax.axvline(0, ls=":",  lw=1.0, c="k",    alpha=0.55)
    ax.axhline(0, ls="--", lw=1.0, c="grey", alpha=0.55)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=7, steps=[1, 2, 3, 6]))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5]))
    ax.yaxis.set_minor_locator(NullLocator())   
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.grid(True, ls=":", alpha=0.18, which="major")

def _legend_proxies(tags, colors, lstyles, lw=2.0):
    return [Line2D([0], [0], ls=lstyles[t], color=colors[t], lw=lw, label=t) for t in tags]

def _smooth(y: np.ndarray, k: int) -> np.ndarray:
    if k is None or k <= 1: 
        return y
    k = int(k)
    ker = np.ones(k, dtype=float) / float(k)
    return np.convolve(y, ker, mode="same")

def _auto_ylim_inst(df, colname: str):
    vals = df[colname].to_numpy(dtype=float)
    if vals.size == 0 or np.all(np.isnan(vals)):
        return (-0.1, 0.1)
    m = float(np.nanmax(np.abs(vals)))

    span = max(2.3 * m, 0.12)   
    half = span / 2.0
    return (-half, half)

def _auto_ylim_cumu(series: np.ndarray, pad_frac=0.08, min_span=0.05):
    y = np.asarray(series, dtype=float)
    if y.size == 0 or np.all(np.isnan(y)):
        return (-0.1, 0.1)
    lo, hi = np.nanpercentile(y, [1, 99])
    span = max(hi - lo, min_span)
    pad = span * pad_frac
    return (lo - pad, hi + pad)

def draw_fig_6_3(df_inst: pd.DataFrame, df_cumu: pd.DataFrame, out_png: Path, out_pdf: Path, out_svg: Path):
    fig, axs = plt.subplots(2, 2, figsize=FIGSIZE_63, constrained_layout=False)
    axB, axC, axD, axE = axs[0,0], axs[0,1], axs[1,0], axs[1,1]

    x_min = int(min(df_inst["lag"].min(), df_cumu["lag"].min()))
    x_max = int(max(df_inst["lag"].max(), df_cumu["lag"].max()))

    # ----- (A) A->B
    _style_axes(axB)
    axB.set_xlim(x_min, x_max)
    axB.set_ylim(*_auto_ylim_inst(df_inst, "delta_B"))
    for t in TAGS_63:
        sub = df_inst[df_inst["tag"] == t]
        y = _smooth(sub["delta_B"].values, SMOOTH_K)
        axB.plot(sub["lag"], y, color=COLORS_63[t], ls=LSTYLES_63[t], lw=1.8)
    axB.set_title("(A) A→B — Instantaneous Δ", loc="left")
    axB.set_ylabel("Instantaneous Δ")
    axB.set_xlabel("Lag (hours)")

    # ----- (B) A->C
    _style_axes(axC)
    axC.set_xlim(x_min, x_max)
    axC.set_ylim(*_auto_ylim_inst(df_inst, "delta_C"))
    for t in TAGS_63:
        sub = df_inst[df_inst["tag"] == t]
        y = _smooth(sub["delta_C"].values, SMOOTH_K)
        axC.plot(sub["lag"], y, color=COLORS_63[t], ls=LSTYLES_63[t], lw=1.8)
    axC.set_title("(B) A→C — Instantaneous Δ", loc="left")
    axC.set_ylabel("Instantaneous Δ")
    axC.set_xlabel("Lag (hours)")

    # ----- (C) A->D
    _style_axes(axD)
    axD.set_xlim(x_min, x_max)
    axD.set_ylim(*_auto_ylim_inst(df_inst, "delta_D"))
    for t in TAGS_63:
        sub = df_inst[df_inst["tag"] == t]
        y = _smooth(sub["delta_D"].values, SMOOTH_K)
        axD.plot(sub["lag"], y, color=COLORS_63[t], ls=LSTYLES_63[t], lw=1.8)
    axD.set_title("(C) A→D — Instantaneous Δ", loc="left")
    axD.set_ylabel("Instantaneous Δ")
    axD.set_xlabel("Lag (hours)")

    # ----- (D) A->E cumulative
    _style_axes(axE)
    axE.set_xlim(x_min, x_max)
    axE.set_ylim(*_auto_ylim_cumu(df_cumu["cum_delta"].values))
    for t in TAGS_63:
        sub = df_cumu[df_cumu["tag"] == t]
        y = _smooth(sub["cum_delta"].values, SMOOTH_K)
        axE.plot(sub["lag"], y, color=COLORS_63[t], ls=LSTYLES_63[t], lw=2.2)
    axE.set_title("(D) A→E — Cumulative Δ (single-chain)", loc="left")
    axE.set_ylabel("Cumulative Δ")
    axE.set_xlabel("Lag (hours)")

    proxies = _legend_proxies(TAGS_63, COLORS_63, LSTYLES_63, lw=2.2)
    fig.legend(proxies, [p.get_label() for p in proxies],
               ncol=len(TAGS_63), loc="lower center",
               bbox_to_anchor=(0.5, 0.02), frameon=False,
               columnspacing=1.6, handlelength=2.8)

    plt.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.16, wspace=0.28, hspace=0.35)
    for fp in tqdm([out_png, out_pdf, out_svg], desc="[Fig6.3] save", ncols=88):
        fig.savefig(fp, dpi=DPI)
    plt.close(fig)

# ===== draw: Figure 6.4  =====
def draw_fig_6_4(df_eonly: pd.DataFrame, out_png: Path, out_pdf: Path, out_svg: Path):
    fig, ax = plt.subplots(figsize=FIGSIZE_64, dpi=DPI)

    for t in TAGS_64:
        sub = df_eonly.loc[df_eonly["tag"] == t, ["lag", "cum_delta"]].sort_values("lag")
        y = _smooth(sub["cum_delta"].values, SMOOTH_K)
        ax.plot(sub["lag"], y, label=t,
                color=COLORS_64[t], linestyle=LSTYLES_64[t], linewidth=2.2)

    ax.axhline(0, color="gray", lw=1.1, ls="--", alpha=0.6)
    ax.axvline(0, color="gray", lw=1.1, ls=":",  alpha=0.7)

    ax.set_xlim(df_eonly["lag"].min(), df_eonly["lag"].max())
    ax.set_ylim(*_auto_ylim_cumu(df_eonly["cum_delta"].values))

    ax.set_title("E cumulative Δ under multi-cause deletions")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Cumulative Δ")

    fig.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.02),
           frameon=False, columnspacing=1.6, handlelength=2.6)

    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.98])
    plt.subplots_adjust(bottom=0.18, top=0.94, left=0.10, right=0.98)

    for fp in tqdm([out_png, out_pdf, out_svg], desc="[Fig6.4] save", ncols=88):
        fig.savefig(fp, dpi=DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

# ===== main =====
if __name__ == "__main__":
    print("[Load] reading inputs for 6.3 / 6.4 …")
    for _ in tqdm(range(1), desc="[Load] 63 inst", ncols=88):
        df63_inst = _read_63_inst(F_63_BCD_INST)
    for _ in tqdm(range(1), desc="[Load] 63 cumu", ncols=88):
        df63_cumu = _read_63_cumu(F_63_AE_CUMU)
    for _ in tqdm(range(1), desc="[Load] 64 eonly", ncols=88):
        df64_eonly = _read_64_eonly(F_64_EONLY)

    draw_fig_6_3(df63_inst, df63_cumu, F_OUT_63_PNG, F_OUT_63_PDF, F_OUT_63_SVG)
    draw_fig_6_4(df64_eonly, F_OUT_64_PNG, F_OUT_64_PDF, F_OUT_64_SVG)
    print(f"[Done] outputs -> {OUTDIR}")
