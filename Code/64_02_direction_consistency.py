
from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

OUT_DIR = Path("outputs/64_run")
OUT_DIR.mkdir(parents=True, exist_ok=True)

F_COV    = OUT_DIR / "64_coverage_by_lag.csv"     
F_SERIES = OUT_DIR / "6402_direction_series.csv"  
F_ADD    = OUT_DIR / "64_additivity_by_lag.csv"   

IN_OBS = "outputs/62_run/62_observed.csv"       
IN_DO  = "outputs/63_run/631_do__Full_main.csv"   

# Outputs
F_OUT_PNG = OUT_DIR / "fig_64_panel.png"
F_OUT_PDF = OUT_DIR / "fig_64_panel.pdf"
F_OUT_SVG = OUT_DIR / "fig_64_panel.svg"

# Params
LAG_MIN = 0
LAG_MAX = 72
PANEL_W, PANEL_H = 9.0, 3.0
COLOR_COV = "#1f77b4"
COLOR_ADJ = "#ff7f0e"
COLOR_DO  = "#2ca02c"
COLOR_RES = "#7f7f7f"
LW_MAIN, LW_BASE = 2.2, 1.8
ALPHA_CI = 0.15


def _need_cols(df: pd.DataFrame, cols: list, where: Path | str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing columns in {where}: {missing}")


def _clip(df: pd.DataFrame, lag_col: str = "lag") -> pd.DataFrame:
    g = df.copy()
    g[lag_col] = pd.to_numeric(g[lag_col], errors="coerce").astype(int)
    return (g[(g[lag_col] >= LAG_MIN) & (g[lag_col] <= LAG_MAX)]
              .sort_values(lag_col)
              .reset_index(drop=True))


def _build_cov():
    print("[6402] build coverage metrics …")
    # 读 E_on
    obs = pd.read_csv(IN_OBS, usecols=["stay_id", "t", "E_hat", "E_on"])
    do  = pd.read_csv(IN_DO,  usecols=["stay_id", "t", "E_hat_doA0"])
    obs["t"] = pd.to_numeric(obs["t"], errors="coerce").astype(int)
    do["t"]  = pd.to_numeric(do["t"],  errors="coerce").astype(int)

    anchor_ids = (obs[(obs["t"]==0) & (obs["E_on"]==0)]["stay_id"].drop_duplicates())
    n_anchor = int(anchor_ids.size)
    anchor_set = set(anchor_ids.tolist())
    obs_alive = {t: set(g.loc[g["E_on"]==0, "stay_id"].tolist())
                 for t, g in obs.groupby("t", sort=False)}
    do_rows   = {t: set(g["stay_id"].tolist()) for t, g in do.groupby("t", sort=False)}

    rows = []
    for lag in range(LAG_MIN, LAG_MAX+1):
        s_obs = obs_alive.get(lag, set())
        s_do  = do_rows.get(lag, set())
        n_eff = len(anchor_set & s_obs & s_do)
        rows.append({"lag": lag, "n_stays_anchor": n_anchor, "n_stays_effective": n_eff})

    pd.DataFrame(rows, columns=["lag","n_stays_anchor","n_stays_effective"]).to_csv(F_COV, index=False)
    print(f"[6402] wrote: {F_COV}")


def _read_cov() -> pd.DataFrame:
    df = pd.read_csv(F_COV)
    _need_cols(df, ["lag", "n_stays_anchor", "n_stays_effective"], F_COV)
    return _clip(df, "lag")


def _read_series() -> pd.DataFrame:
    s = pd.read_csv(F_SERIES)
    _need_cols(s, ["lag", "adj_mean", "do_mean"], F_SERIES)

    for pair in (("adj_lo", "adj_hi"), ("do_lo", "do_hi")):
        any_present = any(c in s.columns for c in pair)
        if any_present:
            _need_cols(s, list(pair), F_SERIES)
    return _clip(s, "lag")


def _read_additivity() -> pd.DataFrame:
    a = pd.read_csv(F_ADD)
    _need_cols(a, ["lag", "resid_mean"], F_ADD)
    return _clip(a, "lag")


def _format_axes(axs):
    for ax in axs:
        ax.grid(True, ls=":", lw=0.8, alpha=0.6)
        ax.set_xlim(LAG_MIN, LAG_MAX)


def draw_panel_A(ax, cov: pd.DataFrame):
    ax.plot(cov["lag"], cov["n_stays_anchor"],    color=COLOR_COV, lw=LW_BASE, ls="--", label="Anchor cohort")
    ax.plot(cov["lag"], cov["n_stays_effective"], color=COLOR_COV, lw=LW_MAIN,           label="Effective (obs ∩ do)")
    ax.set_title("Panel A  Coverage by lag")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Number of stays")
    ax.legend(loc="best", frameon=False)


def draw_panel_B(ax, ser: pd.DataFrame):
    ax.plot(ser["lag"], ser["adj_mean"], color=COLOR_ADJ, lw=LW_MAIN, label="Adjusted (level)")
    ax.plot(ser["lag"], ser["do_mean"],  color=COLOR_DO,  lw=LW_MAIN, label="Interventional do(A=0) (level)")
    if {"adj_lo", "adj_hi"}.issubset(ser.columns):
        ax.fill_between(ser["lag"], ser["adj_lo"], ser["adj_hi"], color=COLOR_ADJ, alpha=ALPHA_CI, lw=0)
    if {"do_lo", "do_hi"}.issubset(ser.columns):
        ax.fill_between(ser["lag"], ser["do_lo"], ser["do_hi"], color=COLOR_DO, alpha=ALPHA_CI, lw=0)
    ax.set_title("Panel B  Adjusted vs interventional")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Mean level of E")
    ax.legend(loc="best", frameon=False)


def draw_panel_C(ax, addf: pd.DataFrame):
    ax.plot(addf["lag"], addf["resid_mean"], color=COLOR_RES, lw=LW_MAIN, label="Additivity residual")
    ax.axhline(0.0, color="k", lw=1.0, alpha=0.6)
    ax.set_title("Panel C  Additivity residuals")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Residual")
    ax.legend(loc="best", frameon=False)


if __name__ == "__main__":
    _build_cov()
    with tqdm(total=3, ncols=88, desc="[6402] read inputs …") as p:
        cov = _read_cov();    p.update(1)
        ser = _read_series(); p.update(1)
        addf = _read_additivity(); p.update(1)

    fig, axes = plt.subplots(1, 3, figsize=(PANEL_W * 3, PANEL_H), dpi=180)
    draw_panel_A(axes[0], cov)
    draw_panel_B(axes[1], ser)
    draw_panel_C(axes[2], addf)
    _format_axes(axes)
    fig.tight_layout()

    for pth in (F_OUT_PNG, F_OUT_PDF, F_OUT_SVG):
        fig.savefig(pth, bbox_inches="tight")
    print(f"[6402] wrote: {F_OUT_PNG}")
    print("[6402] done")

