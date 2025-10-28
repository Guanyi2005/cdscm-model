
import os
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

# ===== paths & files =====
PARAMS_YAML = "62_00_params.yaml"
INPUT_OBS   = "outputs/62_run/62_observed.csv"
OUT63_DIR     = "outputs/63_run"

OUT_FULL    = os.path.join(OUT63_DIR, "631_do__Full_main.csv")
OUT_NOA2B   = os.path.join(OUT63_DIR, "631_do__NoAtoB.csv")
OUT_NOB2C   = os.path.join(OUT63_DIR, "631_do__NoBtoC.csv")
OUT_NOC2D   = os.path.join(OUT63_DIR, "631_do__NoCtoD.csv")
OUT_NONE    = os.path.join(OUT63_DIR, "631_do__None.csv")

OUT_NOA2E   = os.path.join(OUT63_DIR, "632_do__NoAtoE.csv")
OUT_NOB2E   = os.path.join(OUT63_DIR, "632_do__NoBtoE.csv")
OUT_NOC2E   = os.path.join(OUT63_DIR, "632_do__NoCtoE.csv")
OUT_NOD2E   = os.path.join(OUT63_DIR, "632_do__NoDtoE.csv")
OUT_NONE_E  = os.path.join(OUT63_DIR, "632_do__NoneE.csv")

# ===== params =====
REQ_COLS = ["stay_id", "t", "A_low"]   
PROGRESS_UNIT = "stay"


RUN_SPECS = {

    "Full_main": {},
    "NoAtoB":    {"B": ["kappa_A"]},
    "NoBtoC":    {"C": ["kappa_B"]},
    "NoCtoD":    {"D": ["kappa_C"]},
    "None": {
        "B": ["kappa_A"],
        "C": ["kappa_B", "kappa_A"],
        "D": ["kappa_C", "kappa_B", "kappa_A"],
        "E": ["kappa_D", "kappa_C", "kappa_B", "kappa_A"],
    },

    "NoAtoE":    {"E": ["kappa_A"]},
    "NoBtoE":    {"E": ["kappa_B"]},
    "NoCtoE":    {"E": ["kappa_C"]},
    "NoDtoE":    {"E": ["kappa_D"]},
    "NoneE":     {"E": ["kappa_A", "kappa_B", "kappa_C", "kappa_D"]},
}

def _err(msg: str):
    raise RuntimeError(msg)

def load_params(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "effects" not in cfg:
        _err("params missing 'effects' section")
    eff = cfg["effects"]
    for node in ["A", "B", "C", "D", "E"]:
        if node not in eff:
            _err(f"effects missing node '{node}'")
    required = {
        "A": ["baseline"],
        "B": ["baseline", "kappa_A"],
        "C": ["baseline", "kappa_B", "kappa_A"],
        "D": ["baseline", "kappa_C", "kappa_B", "kappa_A"],
        "E": ["baseline", "kappa_D", "kappa_C", "kappa_B", "kappa_A"],
    }
    for node, keys in required.items():
        for k in keys:
            if k not in eff[node]:
                _err(f"effects.{node} missing key '{k}'")
    return eff

def apply_run_overrides(effects_base: dict, run_tag: str) -> dict:
    eff = {k: dict(v) for k, v in effects_base.items()}
    spec = RUN_SPECS.get(run_tag, {})
    for node, zero_keys in spec.items():
        for k in zero_keys:
            if k in eff[node]:
                eff[node][k] = 0.0
    return eff

def load_observed(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        _err(f"missing input file: {csv_path}")
    need = [
        "stay_id", "t",
        "A_low",
        "A_value",          
        "B_sum_value",      
        "C_value",          
        "D_value"           
    ]
    df_all = pd.read_csv(csv_path)
    miss = [c for c in need if c not in df_all.columns]
    if miss:
        _err(f"inputs missing columns {miss} in {csv_path}")
    df = df_all[need].copy()
    df["stay_id"] = df["stay_id"].astype(int)
    df["t"] = df["t"].astype(int)
    df["A_low"] = df["A_low"].astype(int)
    df = df.sort_values(["stay_id", "t"]).reset_index(drop=True)
    return df

def simulate_doA0_one_stay(stay_df, effects, cut_edges=None, use_local_baseline=False, baseline_window=3):

    if cut_edges is None:
        cut_edges = set()

    cols_need = ["t", "A_value", "B_sum_value", "C_value", "D_value"]
    miss = [c for c in cols_need if c not in stay_df.columns]
    if miss:
        raise KeyError(f"simulate_doA0_one_stay missing columns: {miss}")

    sdf = stay_df.sort_values("t").reset_index(drop=True).copy()
    t_series = sdf["t"].tolist()

    B0 = effects["B0"]; C0 = effects["C0"]; D0 = effects["D0"]; E0 = effects["E0"]

    kA_B = 0.0 if "A->B" in cut_edges else effects["kA_B"]

    kB_C = 0.0 if "B->C" in cut_edges else effects["kB_C"]
    kA_C = 0.0 if "A->C" in cut_edges else effects.get("kA_C", 0.0)

    kC_D = 0.0 if "C->D" in cut_edges else effects["kC_D"]
    kB_D = 0.0 if "B->D" in cut_edges else effects.get("kB_D", 0.0)
    kA_D = 0.0 if "A->D" in cut_edges else effects.get("kA_D", 0.0)

    kD_E = 0.0 if "D->E" in cut_edges else effects["kD_E"]
    kC_E = 0.0 if "C->E" in cut_edges else effects.get("kC_E", 0.0)
    kB_E = 0.0 if "B->E" in cut_edges else effects.get("kB_E", 0.0)
    kA_E = 0.0 if "A->E" in cut_edges else effects.get("kA_E", 0.0)

    if use_local_baseline:
        w = max(1, min(baseline_window, len(sdf)))
        B0 = float(sdf["B_sum_value"].iloc[:w].mean())
        C0 = float(sdf["C_value"].iloc[:w].mean())
        D0 = float(sdf["D_value"].iloc[:w].mean())
        E0 = float(sdf["D_value"].iloc[:w].mean()) if "E0" not in effects else E0

    B = [None] * len(sdf)
    C = [None] * len(sdf)
    D = [None] * len(sdf)
    E = [None] * len(sdf)

    B[0] = B0
    C[0] = C0
    D[0] = D0
    E[0] = E0

    for i in range(1, len(sdf)):
        A_prev = 0.0
        B_prev, C_prev, D_prev = B[i-1], C[i-1], D[i-1]

        B[i] = B0 + kA_B * A_prev
        C[i] = C0 + kB_C * B_prev + kA_C * A_prev
        D[i] = D0 + kC_D * C_prev + kB_D * B_prev + kA_D * A_prev
        E[i] = E0 + kD_E * D_prev + kC_E * C_prev + kB_E * B_prev + kA_E * A_prev

    out = pd.DataFrame({
        "t": t_series,
        "B_do": B,
        "C_do": C,
        "D_do": D,
        "E_do": E,
    })
    return out


def run_tag_and_write(df_obs: pd.DataFrame, run_tag: str, effects_base: dict, out_path: str):
    eff_nested = apply_run_overrides(effects_base, run_tag)
    eff_flat = flatten_effects(eff_nested)

    stays = df_obs["stay_id"].unique().tolist()
    rows = []
    pbar = tqdm(total=len(stays), desc=f" {run_tag}", unit=PROGRESS_UNIT)
    for sid in stays:
        sub = df_obs[df_obs["stay_id"] == sid]
        sim = simulate_doA0_one_stay(sub, eff_flat)
        sim = sim.rename(columns={
            "B_do": "B_hat_doA0",
            "C_do": "C_hat_doA0",
            "D_do": "D_hat_doA0",
            "E_do": "E_hat_doA0",
        })
        sim.insert(0, "stay_id", int(sid))
        sim.insert(1, "run_tag", run_tag)
        rows.append(sim)
        pbar.update(1)
    pbar.close()

    out = pd.concat(rows, ignore_index=True)
    out = out[["stay_id", "t", "run_tag", "B_hat_doA0", "C_hat_doA0", "D_hat_doA0", "E_hat_doA0"]]
    out.to_csv(out_path, index=False)

def flatten_effects(eff_nested: dict) -> dict:

    f = {}
    f["B0"] = eff_nested["B"]["baseline"]
    f["C0"] = eff_nested["C"]["baseline"]
    f["D0"] = eff_nested["D"]["baseline"]
    f["E0"] = eff_nested["E"]["baseline"]

    f["kA_B"] = eff_nested["B"].get("kappa_A", 0.0)

    f["kB_C"] = eff_nested["C"].get("kappa_B", 0.0)
    f["kA_C"] = eff_nested["C"].get("kappa_A", 0.0)

    f["kC_D"] = eff_nested["D"].get("kappa_C", 0.0)
    f["kB_D"] = eff_nested["D"].get("kappa_B", 0.0)
    f["kA_D"] = eff_nested["D"].get("kappa_A", 0.0)

    f["kD_E"] = eff_nested["E"].get("kappa_D", 0.0)
    f["kC_E"] = eff_nested["E"].get("kappa_C", 0.0)
    f["kB_E"] = eff_nested["E"].get("kappa_B", 0.0)
    f["kA_E"] = eff_nested["E"].get("kappa_A", 0.0)
    return f

def main():
    for d in [OUT63_DIR, OUT63_DIR]:
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
    effects_base = load_params(PARAMS_YAML)
    df_obs = load_observed(INPUT_OBS)

    run_tag_and_write(df_obs, "Full_main", effects_base, OUT_FULL)
    run_tag_and_write(df_obs, "NoAtoB",   effects_base, OUT_NOA2B)
    run_tag_and_write(df_obs, "NoBtoC",   effects_base, OUT_NOB2C)
    run_tag_and_write(df_obs, "NoCtoD",   effects_base, OUT_NOC2D)
    run_tag_and_write(df_obs, "None",     effects_base, OUT_NONE)

    run_tag_and_write(df_obs, "NoAtoE",   effects_base, OUT_NOA2E)
    run_tag_and_write(df_obs, "NoBtoE",   effects_base, OUT_NOB2E)
    run_tag_and_write(df_obs, "NoCtoE",   effects_base, OUT_NOC2E)
    run_tag_and_write(df_obs, "NoDtoE",   effects_base, OUT_NOD2E)
    run_tag_and_write(df_obs, "NoneE",    effects_base, OUT_NONE_E)

if __name__ == "__main__":
    main()
