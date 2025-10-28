
# ===== paths & files =====
from pathlib import Path
import numpy as np, pandas as pd, json, hashlib, platform
from datetime import datetime

OUT_DIR = Path("outputs/51_C1"); OUT_DIR.mkdir(parents=True, exist_ok=True)
F_SERIES = OUT_DIR/"series_51.csv"
F_DEP    = OUT_DIR/"dependency_footprint_51.csv"
F_REPVAL = OUT_DIR/"repvals_51.csv"
F_CONFIG = OUT_DIR/"config_51.json"
F_ENV    = OUT_DIR/"env_51.json"

# ===== params =====
SEED=1; T=30
ALPHA=0.8; BETA=0.2; THETA1=1.0; THETA2=0.5
SIGMA_Y=0.05; SIGMA_Z=0.08
S_MEAN=2.2; S_AR=0.85; S_NOISE=0.15
REP_T=(0,5,10,15,20,25,30)

# ===== helpers =====
def f_I(x): return np.tanh(x)
def f_Y(x): return x**2

def gen_S(T,seed=SEED,mean=S_MEAN,ar=S_AR,noise_std=S_NOISE):
    rng=np.random.default_rng(seed); S=np.zeros(T+1)
    S[0]=mean
    for t in range(1,T+1):
        S[t]=mean*(1-ar)+ar*S[t-1]+rng.normal(0.0,noise_std)
    return S

def simulate(T=T,seed=SEED):
    rng=np.random.default_rng(seed)
    S=gen_S(T,seed)
    I=np.zeros(T+1); Y=np.zeros(T+1); Z=np.zeros(T+1); C=np.zeros(T+1,dtype=int)
    for t in range(T):
        I[t+1]=ALPHA*S[t]+BETA*C[t]
        Y[t+1]=f_I(I[t])+rng.normal(0.0,SIGMA_Y)
        Z[t+1]=f_Y(Y[t])+rng.normal(0.0,SIGMA_Z)
        C[t+1]=1 if (Z[t]>THETA1 and Y[t]>THETA2) else C[t]
    return pd.DataFrame(dict(t=np.arange(T+1),S=S,I=I,Y=Y,Z=Z,C=C))

def dep_matrix():
    rows=["I(t+1)","Y(t+1)","Z(t+1)","C(t+1)"]
    cols=["S(t)","I(t)","Y(t)","Z(t)","C(t)"]
    M=np.array([[1,0,0,0,1],[0,1,0,0,0],[0,0,1,0,0],[0,0,1,1,1]],dtype=int)
    return pd.DataFrame(M,index=rows,columns=cols)

def repvals(df,t_points=REP_T):
    rec=[]
    for t in t_points:
        row=df[df["t"]==t].iloc[0]
        rec.append(dict(t=int(t),I=row["I"],Y=row["Y"],Z=row["Z"],C=int(row["C"])))
    return pd.DataFrame(rec)

# ===== main =====
if __name__=="__main__":
    df=simulate(); df.to_csv(F_SERIES,index=False)
    dep_matrix().to_csv(F_DEP,index=False)
    repvals(df).to_csv(F_REPVAL,index=False)
    json.dump(dict(seed=SEED,T=T,alpha=ALPHA,beta=BETA,theta1=THETA1,theta2=THETA2),
              open(F_CONFIG,"w",encoding="utf-8"),indent=2,ensure_ascii=False)
    json.dump(dict(python=platform.python_version(),
                   platform=platform.platform(),
                   sha256=hashlib.sha256(open(__file__,"rb").read()).hexdigest()),
              open(F_ENV,"w",encoding="utf-8"),indent=2,ensure_ascii=False)
    print("[51_01] data exported:",F_SERIES,F_DEP,F_REPVAL)
