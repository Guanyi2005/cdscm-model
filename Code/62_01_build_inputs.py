
# ===== paths & files =====
import os, time, yaml
import pandas as pd

PARAMS = "62_00_params.yaml"

# ===== params =====
def load_params(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(p):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def log(msg):
    print(f"[6201] {msg}", flush=True)

# ===== common helpers =====
def _need_cols(df: pd.DataFrame, cols):
    miss = [c for c in cols if c not in df.columns]
    if miss: raise KeyError(f"missing columns: {miss}")

# ===== main =====
def main():
    tic = time.time()
    cfg = load_params(PARAMS)

    paths   = cfg["paths"]
    cols    = cfg["columns"]              
    thr     = cfg["thresholds"]
    in_csv  = paths["input_from_61"]
    out_inp = paths["v62_inputs"]
    out_cmp = paths["v62_compare"]

    ensure_dir(os.path.dirname(out_inp))
    ensure_dir(os.path.dirname(out_cmp))


    log(f"read: {in_csv}")
    df = pd.read_csv(in_csv)


    REQ = {
        "stay_id":   "stay_id",
        "hour":      "hour_from_t0",
        "A":         "a_map_mmhg",
        "B1":        "b_cate_mcgkgmin",
        "B2":        "b_vp_unitshour",
        "C":         "c_urine_mlkgh",
        "D":         "d_creat_mgdl",
        "E":         "e_rrt_on",
    }
    _need_cols(df, list(REQ.values()))

    std = pd.DataFrame({
        cols["stay_id"]:   df[REQ["stay_id"]].astype(int),
        cols["t"]:         pd.to_numeric(df[REQ["hour"]], errors="coerce").astype(int),
        cols["A_value"]:   pd.to_numeric(df[REQ["A"]], errors="coerce"),
        cols["B1_value"]:  pd.to_numeric(df[REQ["B1"]], errors="coerce"),
        cols["B2_value"]:  pd.to_numeric(df[REQ["B2"]], errors="coerce"),
        cols["C_value"]:   pd.to_numeric(df[REQ["C"]], errors="coerce"),
        cols["D_value"]:   pd.to_numeric(df[REQ["D"]], errors="coerce"),
        cols["E_on"]:      pd.to_numeric(df[REQ["E"]], errors="coerce").fillna(0).astype(int),
    })

    std = std.sort_values([cols["stay_id"], cols["t"]]).reset_index(drop=True)

    std[cols["B_sum_value"]] = std[cols["B1_value"]].fillna(0.0) + std[cols["B2_value"]].fillna(0.0)
    std[cols["A_prev"]]      = std.groupby(cols["stay_id"], sort=False)[cols["A_value"]].shift(1)
    if "E_prev" in cols:
        std[cols["E_prev"]] = std.groupby(cols["stay_id"], sort=False)[cols["E_on"]].shift(1).fillna(0).astype(int)

    A_low_thr = float(thr["A_low"])
    C_low_thr = float(thr["C_low"])
    D_hi_thr  = float(thr["D_high"])

    std[cols["A_low_flag"]]  = (std[cols["A_value"]] < A_low_thr).astype(int)
    std[cols["C_low_flag"]]  = (std[cols["C_value"]] < C_low_thr).astype(int)
    std[cols["D_high_flag"]] = (std[cols["D_value"]] > D_hi_thr).astype(int)
    std[cols["B_on_flag"]]   = ((std[cols["B1_value"]].fillna(0) > 0) | (std[cols["B2_value"]].fillna(0) > 0)).astype(int)

    std.to_csv(out_inp, index=False)
    log(f"wrote: {out_inp}")
    std.to_csv(out_cmp, index=False)
    log(f"wrote: {out_cmp}")
    log(f"done, elapsed={time.time()-tic:.1f}s")

if __name__ == "__main__":
    main()
