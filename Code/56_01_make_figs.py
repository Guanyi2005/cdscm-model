from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

OUT_DIR = Path("outputs/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DIR_51 = Path("outputs/51_C1")
F_51_SERIES = DIR_51 / "series_51.csv"
F_51_DEP = DIR_51 / "dependency_footprint_51.csv"
DIR_52 = Path("outputs/52_C2")
F_52_BASE = DIR_52 / "series_52_baseline.csv"
F_52_DO = DIR_52 / "series_52_do.csv"
F_52_METR = DIR_52 / "table_52_metrics.csv"
DIR_53 = Path("outputs/53_C3")
F_53_E_ALL = DIR_53 / "53_A_to_E_counterfactual.ALL.csv"
F_53_BCD = DIR_53 / "53_A_to_BCD_instant.ALL.csv"
DIR_54 = Path("outputs/54_C4")
F_54_PEAKS = DIR_54 / "54_summary_peaks.csv"
F_54_LOWALL = DIR_54 / "54_ALL_low.csv"
F_54_HIGALL = DIR_54 / "54_ALL_high.csv"
F_OUT_PNG = OUT_DIR / "Figure2.png"
F_OUT_PDF = OUT_DIR / "Figure2.pdf"
F_OUT_SVG = OUT_DIR / "Figure2.svg"

FIGSIZE = (12.0, 9.5)
FIG_DPI = 300
PANEL_LETTER_SIZE = 17
TITLE_SIZE = 12
LABEL_SIZE = 9
TICK_SIZE = 8
GRID_ARGS = dict(ls="--", alpha=0.35)
ORDER = ["Full_main", "NoAtoB", "NoBtoC", "NoCtoD", "None"]
STYLE = {"Full_main": "-", "NoAtoB": "--", "NoBtoC": ":", "NoCtoD": "-.", "None": "-"}
WIDTH = {"Full_main": 2.2, "NoAtoB": 1.8, "NoBtoC": 1.8, "NoCtoD": 1.8, "None": 2.6}
COLOR = {"I": "#1f77b4", "Y": "#ff7f0e", "Z": "#2ca02c", "C": "#d62728"}
COLOR_E = {"Full_main": "#1f77b4", "NoAtoB": "#ff7f0e", "NoBtoC": "#2ca02c", "NoCtoD": "#d62728", "None": "#9467bd"}
DISPLAY = {"Full_main": "Full_main", "NoAtoB": "NoAtoB", "NoBtoC": "NoBtoC", "NoCtoD": "NoCtoD", "None": "NoneE"}

A_MAIN_POS = [0.07, 0.56, 0.40, 0.38]
B_MAIN_POS = [0.55, 0.56, 0.40, 0.38]
C_MAIN_POS = [0.07, 0.08, 0.40, 0.38]
D_MAIN_POS = [0.55, 0.08, 0.40, 0.38]

TITLE_A = "C1: Bounded and ordered trajectories"
TITLE_B = "C2: Ordered updates ensure stability"
TITLE_B_S = "Δ (do − baseline)"
TITLE_C = "C3: Interventional closure (path ablations)"
TITLE_D = "C4: Compatibility with causal adjustment"

def _require(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"missing file: {p}")

def _letter(ax, s: str):
    ax.text(0.0, 1.02, s, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=PANEL_LETTER_SIZE, fontweight="bold")

def _style_axes(ax):
    ax.grid(**GRID_ARGS)
    ax.tick_params(labelsize=TICK_SIZE)

def _as_str_series(s: pd.Series) -> pd.Series:
    return s.fillna("None").astype(str).str.strip()

def _plot_dep_matrix(ax, dep_df: pd.DataFrame):
    ax.imshow(dep_df.values, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(dep_df.shape[1]))
    ax.set_yticks(range(dep_df.shape[0]))
    ax.set_xticklabels(dep_df.columns, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(dep_df.index, fontsize=7)
    for i in range(dep_df.shape[0]):
        for j in range(dep_df.shape[1]):
            v = dep_df.iat[i, j]
            ax.text(j, i, str(int(v)), ha="center", va="center", fontsize=7, color=("0.1" if v == 1 else "0.4"))
    for s in ax.spines.values():
        s.set_visible(False)

def _groupbar(ax, xlabels, y_low, y_high):
    idx = np.arange(len(xlabels))
    ax.bar(idx - 0.18, y_low, width=0.36, label="A_low")
    ax.bar(idx + 0.18, y_high, width=0.36, label="A_high")
    ax.set_xticks(idx)
    ax.set_xticklabels(xlabels, rotation=0, fontsize=TICK_SIZE)
    ax.legend(frameon=False, fontsize=LABEL_SIZE)
    _style_axes(ax)

def _strip_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

if __name__ == "__main__":
    for p in [F_51_SERIES, F_51_DEP, F_52_BASE, F_52_DO, F_52_METR, F_53_E_ALL, F_53_BCD, F_54_PEAKS, F_54_LOWALL, F_54_HIGALL]:
        _require(p)

    frames = {}
    for name, fp in tqdm([
        ("51_series", F_51_SERIES),
        ("51_dep", F_51_DEP),
        ("52_base", F_52_BASE),
        ("52_do", F_52_DO),
        ("52_metr", F_52_METR),
        ("53_eall", F_53_E_ALL),
        ("53_bcd", F_53_BCD),
        ("54_peaks", F_54_PEAKS),
        ("54_low", F_54_LOWALL),
        ("54_high", F_54_HIGALL),
    ], desc="[Figure2] read tables", ncols=90):
        frames[name] = pd.read_csv(fp, keep_default_na=False, na_filter=False)

    s51 = frames["51_series"][["t", "I", "Y", "Z", "C"]].copy()
    dep = frames["51_dep"].copy()
    s52b = frames["52_base"][["t", "I", "Y", "Z"]].copy()
    s52d = frames["52_do"][["t", "I", "Y", "Z"]].copy()
    metr = frames["52_metr"][["do_time"]].copy()
    if not np.array_equal(s52b["t"].values, s52d["t"].values):
        raise RuntimeError("[Figure2] 52 series t misaligned")
    do_time = int(metr["do_time"].iloc[0])
    e53 = frames["53_eall"][["t", "E", "tag"]].copy()
    e53["tag"] = _as_str_series(e53["tag"])
    bcd = frames["53_bcd"][["t", "B_hat", "C_hat", "D_hat"]].copy()
    peaks = frames["54_peaks"][["cond", "tag", "E_peak"]].copy()
    peaks["cond"] = _as_str_series(peaks["cond"])
    peaks["tag"] = _as_str_series(peaks["tag"])
    low53 = frames["54_low"][["t", "E", "tag", "cond"]].copy()
    low53["tag"] = _as_str_series(low53["tag"])
    low53["cond"] = _as_str_series(low53["cond"])
    hig53 = frames["54_high"][["t", "E", "tag", "cond"]].copy()
    hig53["tag"] = _as_str_series(hig53["tag"])
    hig53["cond"] = _as_str_series(hig53["cond"])

    t52 = s52b["t"].values
    dI = s52d["I"].values - s52b["I"].values
    dY = s52d["Y"].values - s52b["Y"].values
    dZ = s52d["Z"].values - s52b["Z"].values

    fig = plt.figure(figsize=FIGSIZE, dpi=FIG_DPI)

    axA = fig.add_axes(A_MAIN_POS)
    axA.plot(s51["t"], s51["Y"], color=COLOR["Y"], lw=1.8, label="Y")
    axA.plot(s51["t"], s51["Z"], color=COLOR["Z"], lw=1.8, label="Z")
    axA.step(s51["t"], s51["C"], where="post", color=COLOR["C"], lw=1.8, label="C")
    axA2 = axA.twinx()
    axA2.plot(s51["t"], s51["I"], color=COLOR["I"], lw=1.6, alpha=0.9, label="I")
    axA.set_title(TITLE_A, fontsize=TITLE_SIZE)
    axA.set_xlabel("t", fontsize=LABEL_SIZE)
    axA.set_ylabel("Y / Z / C", fontsize=LABEL_SIZE)
    axA2.set_ylabel("I", fontsize=LABEL_SIZE)
    _style_axes(axA)
    axA2.grid(False)
    axA.legend(frameon=False, fontsize=LABEL_SIZE, loc="upper left")
    axA2.legend(frameon=False, fontsize=LABEL_SIZE, loc="upper right")
    _letter(axA, "A")
    axAin = inset_axes(
        axA,
        width="100%", height="100%",
        loc="lower left",
        bbox_to_anchor=(0.66, 0.08, 0.30, 0.34),  # x0, y0, w, h in axes fraction
        bbox_transform=axA.transAxes,
        borderpad=0.0
    )
    _plot_dep_matrix(axAin, dep)

    axB = fig.add_axes(B_MAIN_POS)
    for col in ["I", "Y", "Z"]:
        axB.plot(s52b["t"], s52b[col], lw=2.2, alpha=0.30, color=COLOR[col], label=f"{col} baseline")
        axB.plot(s52d["t"], s52d[col], lw=1.2, alpha=0.95, color=COLOR[col], label=f"{col} do")
    axB.axvline(do_time, color="0.2", lw=1.0, ls="--")
    axB.text(do_time + 0.3, axB.get_ylim()[1] * 0.95, f"do at t={do_time}", fontsize=8)
    axB.set_title(TITLE_B, fontsize=TITLE_SIZE)
    axB.set_xlabel("t", fontsize=LABEL_SIZE)
    axB.set_ylabel("level", fontsize=LABEL_SIZE)
    _style_axes(axB)
    axB.legend(frameon=False, fontsize=LABEL_SIZE - 0.5, loc="upper right")
    _letter(axB, "B")
    axBin = inset_axes(axB, width="26%", height="30%", loc="lower right", borderpad=0.8)
    axBin.plot(t52, dI, lw=1.0, color=COLOR["I"])
    axBin.plot(t52, dY, lw=1.0, color=COLOR["Y"])
    axBin.plot(t52, dZ, lw=1.0, color=COLOR["Z"])
    axBin.axhline(0.0, color="0.3", lw=0.6)
    axBin.axvline(do_time, color="0.3", lw=0.6, ls="--")
    _strip_axes(axBin)
    axBin.text(0.03, 0.85, TITLE_B_S, transform=axBin.transAxes, fontsize=7)

    axC = fig.add_axes(C_MAIN_POS)
    for tag in ORDER:
        sub = e53[e53["tag"] == tag]
        axC.plot(sub["t"], sub["E"], STYLE[tag], lw=WIDTH[tag], color=COLOR_E[tag], label=DISPLAY[tag])
    axC.set_title(TITLE_C, fontsize=TITLE_SIZE)
    axC.set_xlabel("t", fontsize=LABEL_SIZE)
    axC.set_ylabel("E(t)", fontsize=LABEL_SIZE)
    _style_axes(axC)
    axC.legend(frameon=False, fontsize=LABEL_SIZE, loc="upper right")
    _letter(axC, "C")

    axD = fig.add_axes(D_MAIN_POS)
    y_low, y_high = [], []
    for tag in ORDER:
        v_low = peaks.loc[(peaks["cond"] == "A_low") & (peaks["tag"] == tag), "E_peak"]
        v_high = peaks.loc[(peaks["cond"] == "A_high") & (peaks["tag"] == tag), "E_peak"]
        y_low.append(float(v_low.iloc[0]))
        y_high.append(float(v_high.iloc[0]))
    _groupbar(axD, [DISPLAY[t] for t in ORDER], y_low, y_high)
    axD.set_title(TITLE_D, fontsize=TITLE_SIZE)
    axD.set_ylabel("Peak E", fontsize=LABEL_SIZE)
    _letter(axD, "D")

    for outp in tqdm([F_OUT_PNG, F_OUT_PDF, F_OUT_SVG], desc="[Figure2] export", ncols=90):
        fig.savefig(outp, bbox_inches="tight", dpi=FIG_DPI)
    plt.close(fig)
    print(f"[Figure2] done -> {F_OUT_PNG} ; {F_OUT_PDF} ; {F_OUT_SVG}")
