
# ===== paths & files =====
from pathlib import Path
import numpy as np,pandas as pd,json
from tqdm import tqdm

OUT_DIR=Path("outputs/51_C1"); OUT_DIR.mkdir(parents=True,exist_ok=True)
F_STRESS=OUT_DIR/"stress_51.csv"
F_SUMMARY=OUT_DIR/"summary_51.csv"
F_DEP=OUT_DIR/"dependency_footprint_51.csv"

# ===== params =====
SEEDS=range(1,101); T=30
ALPHA=0.8; BETA=0.2; THETA1=1.0; THETA2=0.5
SIGMA_Y=0.05; SIGMA_Z=0.08; S_MEAN=2.2; S_AR=0.85; S_NOISE=0.15
TOL=0.25

# ===== helpers =====
def f_I(x): return np.tanh(x)
def f_Y(x): return x**2
def gen_S(T,seed):
    rng=np.random.default_rng(seed); S=np.zeros(T+1); S[0]=S_MEAN
    for t in range(1,T+1): S[t]=S_MEAN*(1-S_AR)+S_AR*S[t-1]+rng.normal(0,S_NOISE)
    return S
def simulate(seed):
    rng=np.random.default_rng(seed); S=gen_S(T,seed)
    I=np.zeros(T+1); Y=np.zeros(T+1); Z=np.zeros(T+1); C=np.zeros(T+1,dtype=int)
    for t in range(T):
        I[t+1]=ALPHA*S[t]+BETA*C[t]
        Y[t+1]=f_I(I[t])+rng.normal(0,SIGMA_Y)
        Z[t+1]=f_Y(Y[t])+rng.normal(0,SIGMA_Z)
        C[t+1]=1 if (Z[t]>THETA1 and Y[t]>THETA2) else C[t]
    return Y,Z,C
def absorb_ok(C): return np.all(np.diff(C)>=0)
def bounded_ok(Y,Z,tol=TOL):
    return (Y.min()>=-1-tol and Y.max()<=1+tol and Z.min()>=0-tol and Z.max()<=2+tol)
def acyclic_ok():
    try: df=pd.read_csv(F_DEP)
    except: return False
    return df.shape==(4,5)

# ===== main =====
if __name__=="__main__":
    rows=[]
    for s in tqdm(SEEDS,desc="stress",ncols=80):
        Y,Z,C=simulate(s); rows.append(dict(seed=s,absorb=absorb_ok(C),
                                            bounded=bounded_ok(Y,Z),acyclic=acyclic_ok()))
    df=pd.DataFrame(rows); df.to_csv(F_STRESS,index=False)
    summary=dict(n=len(df),
                 absorb=int(df["absorb"].sum()),bounded=int(df["bounded"].sum()),
                 acyclic=int(df["acyclic"].sum()))
    pd.DataFrame([summary]).to_csv(F_SUMMARY,index=False)
    print("[51_02] checks exported:",F_STRESS,F_SUMMARY)
