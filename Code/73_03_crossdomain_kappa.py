
# ===== paths & files =====
from pathlib import Path
import yaml, math, csv

IN_ICU_YAML   = Path("62_00_params.yaml")    
IN_MACRO_YAML = Path("72_00_params.yaml")    
OUT_DIR       = Path("outputs/73_run")
F_OUT_CSV     = OUT_DIR / "73_crossdomain_kappa.csv"

# ===== params =====
EDGES = [
    ("A→B", "B", "kappa_A"),
    ("B→C", "C", "kappa_B"),
    ("C→D", "D", "kappa_C"),
    ("D→E", "E", "kappa_D"),
]

# ===== common helpers =====
def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def need_effects_has_keys(effects: dict):
    required = {
        "B": ["kappa_A"],
        "C": ["kappa_B"],
        "D": ["kappa_C"],
        "E": ["kappa_D"],
    }
    for node, keys in required.items():
        if node not in effects:
            raise KeyError(f"effects missing node '{node}'")
        for k in keys:
            if k not in effects[node]:
                raise KeyError(f"effects.{node} missing key '{k}'")

def spearman_r(xs: list[float], ys: list[float]) -> float:

    def rankdata(a: list[float]) -> list[float]:
        n = len(a)
        idx = sorted(range(n), key=lambda i: (a[i], i))
        ranks = [0.0]*n
        i = 0
        while i < n:
            j = i
            while j+1 < n and a[idx[j+1]] == a[idx[i]]:
                j += 1
            r = (i + j + 2) / 2.0 
            for k in range(i, j+1):
                ranks[idx[k]] = r
            i = j + 1
        return ranks
    if len(xs) != len(ys) or len(xs) == 0:
        return float("nan")
    rx, ry = rankdata(xs), rankdata(ys)
    mx = sum(rx)/len(rx); my = sum(ry)/len(ry)
    vx = sum((r-mx)**2 for r in rx); vy = sum((r-my)**2 for r in ry)
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    cov = sum((rx[i]-mx)*(ry[i]-my) for i in range(len(rx)))
    return cov / math.sqrt(vx*vy)

# ===== main =====
def main():
    print("== Cross-domain κ agreement (ICU vs Macro) ==")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] read ICU params")
    p_ic = load_yaml(IN_ICU_YAML)
    eff_ic = p_ic["effects"]
    need_effects_has_keys(eff_ic)

    print("[2/3] read Macro params")
    p_ma = load_yaml(IN_MACRO_YAML)
    eff_ma = p_ma["effects"]
    need_effects_has_keys(eff_ma)

    print("[3/3] collect kappas and compute metrics")
    rows = []
    xs, ys = [], []
    agree = 0
    for edge_lbl, child, kname in EDGES:
        v_ic = float(eff_ic[child][kname])
        v_ma = float(eff_ma[child][kname])
        s_ok = (1 if (v_ic == 0.0 and v_ma == 0.0) else (1 if (v_ic > 0 and v_ma > 0) or (v_ic < 0 and v_ma < 0) else 0))
        rows.append((edge_lbl, v_ic, v_ma, bool(s_ok)))
        xs.append(v_ic); ys.append(v_ma)
        agree += s_ok

    rho = spearman_r(xs, ys)
    sign_rate = agree / len(rows) if rows else float("nan")

    with F_OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["edge","ICU","Macro","sign_agree"])
        for r in rows: w.writerow(r)

    print(f"[OK] wrote: {F_OUT_CSV}")
    print(f"Spearman ρ = {rho:.3f}")
    print(f"Sign agreement = {100.0*sign_rate:.1f}%")

if __name__ == "__main__":
    main()
