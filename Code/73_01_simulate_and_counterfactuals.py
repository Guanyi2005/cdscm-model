
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ===== paths & files =====
IN_INPUTS = Path("outputs/71_run/71_observed_clean.csv")     
IN_ANCH   = Path("outputs/72_run/7201_anchors_A.csv")        
IN_PARAM  = Path("72_00_params.yaml")                      
OUT_DIR   = Path("outputs/73_run")
F_B       = OUT_DIR / "7301_inst_B.csv"                     
F_C       = OUT_DIR / "7301_inst_C.csv"                     
F_D       = OUT_DIR / "7301_inst_D.csv"                   
F_E       = OUT_DIR / "7301_cumu_E.csv"                     
F_ANCHREP = OUT_DIR / "7301_anchor_filter_report.csv"

# ===== params =====
COLS = ["A","B","C","D","E"]
LAG_PRE_TARGET  = 6      
LAG_POST_TARGET = 24     
RNG_SEED = 7301

# ===== utils =====
def need(df: pd.DataFrame, cols, name: str):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise RuntimeError(f"missing {miss} in {name}")

def load_params_yaml(fp: Path):

    p = yaml.safe_load(fp.read_text(encoding="utf-8"))
    if "effects" not in p:
        raise RuntimeError("params must contain 'effects'")
    eff = p["effects"]
    kap = {"B":{}, "C":{}, "D":{}, "E":{}}
    for k in ["B","C","D","E"]:
        if k not in eff:
            raise RuntimeError(f"missing effects.{k}")
        for parent in ["A","B","C","D"]:
            key = f"kappa_{parent}"
            kap[k][parent] = float(eff[k].get(key, 0.0))
    return kap

def parents_of(node: str):
    if node == "B": return ["A"]
    if node == "C": return ["B"]
    if node == "D": return ["C"]
    if node == "E": return ["A","B","C","D"]
    raise ValueError(f"invalid node {node}")

def step_update(prev: dict, a_tminus1: float, kap: dict, drop_edge=None, noneE=False):

    nxt = prev.copy()
    nxt["A"] = a_tminus1
    for node in ["B","C","D"]:
        infl = 0.0
        for u in parents_of(node):
            if drop_edge == (u, node):
                continue
            infl += kap[node].get(u, 0.0) * (prev[u] - 0.0)
        nxt[node] = infl
    infl_E = 0.0
    for u in parents_of("E"):
        if noneE or drop_edge == (u, "E"):
            continue
        infl_E += kap["E"].get(u, 0.0) * (prev[u] - 0.0)
    nxt["E"] = infl_E
    return nxt

def simulate_window(inputs_idx: pd.DataFrame, kap: dict, anchor_t: int,
                    lag_pre: int, lag_post: int, drop_edge=None, noneE=False):

    if (anchor_t - 1) not in inputs_idx.index:
        raise KeyError("a-1 not in inputs index")
    s_prev = inputs_idx.loc[anchor_t - 1, COLS].to_dict()

    lags = list(range(-lag_pre, 0))
    B = [inputs_idx.loc[anchor_t + h, "B"] for h in lags] if lag_pre > 0 else []
    C = [inputs_idx.loc[anchor_t + h, "C"] for h in lags] if lag_pre > 0 else []
    D = [inputs_idx.loc[anchor_t + h, "D"] for h in lags] if lag_pre > 0 else []
    E = [inputs_idx.loc[anchor_t + h, "E"] for h in lags] if lag_pre > 0 else []


    for h in range(0, lag_post + 1):
        t_obs_prev = anchor_t + h - 1
        if t_obs_prev not in inputs_idx.index:
            raise KeyError("prev step missing in inputs index during recursion")
        a_tm1 = float(inputs_idx.loc[t_obs_prev, "A"])
        s_cur = step_update(s_prev, a_tm1, kap, drop_edge=drop_edge, noneE=noneE)
        B.append(s_cur["B"]); C.append(s_cur["C"]); D.append(s_cur["D"]); E.append(s_cur["E"])
        s_prev = s_cur

    lag_all = list(range(-lag_pre, lag_post + 1))
    return np.array(lag_all, dtype=int), np.array(B), np.array(C), np.array(D), np.array(E)

def align_mean(inputs_idx: pd.DataFrame, kap: dict, anchors: list,
               lag_pre: int, lag_post: int, drop_edge=None, noneE=False, desc: str = ""):

    acc_B, acc_C, acc_D, acc_E, n = None, None, None, None, 0
    for a in tqdm(anchors, desc=desc, ncols=80):
        lags, B, C, D, E = simulate_window(inputs_idx, kap, a, lag_pre, lag_post,
                                           drop_edge=drop_edge, noneE=noneE)
        if acc_B is None:
            acc_B = B.astype(float); acc_C = C.astype(float)
            acc_D = D.astype(float); acc_E = E.astype(float)
        else:
            acc_B += B; acc_C += C; acc_D += D; acc_E += E
        n += 1
    if n == 0:
        raise RuntimeError("no anchors to align")
    return lags, acc_B / n, acc_C / n, acc_D / n, acc_E / n

# ===== main =====
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


    df = pd.read_csv(IN_INPUTS, usecols=["t","A","B","C","D","E"])
    need(df, ["t"] + COLS, IN_INPUTS.name)
    df["t"] = df["t"].astype(int)
    inputs = df.set_index("t").sort_index()
    t_min, t_max = int(inputs.index.min()), int(inputs.index.max())
    print(f"[73_01] t_index range: {t_min} … {t_max}")


    anch = pd.read_csv(IN_ANCH, usecols=["t","anchor_t"])
    need(anch, ["t","anchor_t"], IN_ANCH.name)
    A_raw = anch.loc[anch["anchor_t"] == 1, "t"].astype(int).tolist()
    print(f"[73_01] anchors raw: {len(A_raw)} min= {min(A_raw) if A_raw else 'NA'} max= {max(A_raw) if A_raw else 'NA'}")


    need_min = t_min + LAG_PRE_TARGET
    need_max = t_max - LAG_POST_TARGET
    print(f"[73_01] window need anchors in [ {need_min} , {need_max} ]")
    kept, reasons = [], []
    for a in A_raw:
        ok_prev = ((a - 1) in inputs.index)
        ok_win  = (a >= need_min) and (a <= need_max)
        if ok_prev and ok_win:
            kept.append(a)
        else:
            if not ok_prev: reasons.append((a, "no_prev_state(a-1)"))
            if not ok_win:  reasons.append((a, "not_enough_pre_or_post_window"))
    pd.DataFrame(reasons, columns=["anchor","reason"]).to_csv(F_ANCHREP, index=False)
    print(f"[73_01] anchors kept: {len(kept)}")
    if len(kept) == 0:
        raise RuntimeError(
            "no anchors after strict coverage filtering; see outputs/73_run/7301_anchor_filter_report.csv for reasons"
        )

    kap = load_params_yaml(IN_PARAM)


    lag, B_full, C_full, D_full, E_full = align_mean(
        inputs, kap, kept, LAG_PRE_TARGET, LAG_POST_TARGET, drop_edge=None, noneE=False,
        desc="[73_01] full"
    )

    lag, B_noAtoB, _, _, _ = align_mean(
        inputs, kap, kept, LAG_PRE_TARGET, LAG_POST_TARGET, drop_edge=("A","B"), noneE=False,
        desc="[73_01] NoA->B"
    )

    lag, _, C_noBtoC, _, _ = align_mean(
        inputs, kap, kept, LAG_PRE_TARGET, LAG_POST_TARGET, drop_edge=("B","C"), noneE=False,
        desc="[73_01] NoB->C"
    )

    lag, _, _, D_noCtoD, _ = align_mean(
        inputs, kap, kept, LAG_PRE_TARGET, LAG_POST_TARGET, drop_edge=("C","D"), noneE=False,
        desc="[73_01] NoC->D"
    )

    lag, _, _, _, E_noA  = align_mean(inputs, kap, kept, LAG_PRE_TARGET, LAG_POST_TARGET, drop_edge=("A","E"), noneE=False, desc="[73_01] NoA->E")
    lag, _, _, _, E_noB  = align_mean(inputs, kap, kept, LAG_PRE_TARGET, LAG_POST_TARGET, drop_edge=("B","E"), noneE=False, desc="[73_01] NoB->E")
    lag, _, _, _, E_noC  = align_mean(inputs, kap, kept, LAG_PRE_TARGET, LAG_POST_TARGET, drop_edge=("C","E"), noneE=False, desc="[73_01] NoC->E")
    lag, _, _, _, E_noD  = align_mean(inputs, kap, kept, LAG_PRE_TARGET, LAG_POST_TARGET, drop_edge=("D","E"), noneE=False, desc="[73_01] NoD->E")
    lag, _, _, _, E_none = align_mean(inputs, kap, kept, LAG_PRE_TARGET, LAG_POST_TARGET, drop_edge=None,      noneE=True,  desc="[73_01] NoneE")

    pd.DataFrame({
        "lag": lag, "Full_main": B_full, "NoAtoB": B_noAtoB
    }).to_csv(F_B, index=False)

    pd.DataFrame({
        "lag": lag, "Full_main": C_full, "NoBtoC": C_noBtoC
    }).to_csv(F_C, index=False)

    pd.DataFrame({
        "lag": lag, "Full_main": D_full, "NoCtoD": D_noCtoD
    }).to_csv(F_D, index=False)

    def _cumu(x):
        arr = np.asarray(x, dtype=float)
        out = arr.copy()
        for i in range(1, len(out)):
            out[i] = out[i-1] + out[i]
        return out

    pd.DataFrame({
        "lag": lag,
        "Full_main": _cumu(E_full),
        "NoAtoE":    _cumu(E_noA),
        "NoBtoE":    _cumu(E_noB),
        "NoCtoE":    _cumu(E_noC),
        "NoDtoE":    _cumu(E_noD),
        "NoneE":     _cumu(E_none)
    }).to_csv(F_E, index=False)

    print(f"[73_01] wrote:\n  {F_B}\n  {F_C}\n  {F_D}\n  {F_E}\n  {F_ANCHREP}")

if __name__ == "__main__":
    main()
