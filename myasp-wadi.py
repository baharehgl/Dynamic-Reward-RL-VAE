#!/usr/bin/env python3
# wadi_rl_equal_al.py
# =========================================================
# RL + VAE + Label-Propagation + Active-Learning on WADI
# with K equal validation slices and three AL budgets.
# =========================================================
import os, random, numpy as np, pandas as pd, tensorflow as tf
from collections import namedtuple
tf.compat.v1.disable_eager_execution()

from tensorflow.keras import layers, models, losses
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from env_wadi import EnvTimeSeriesWaDi

# ─── paths / GPU ────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
WA_DI    = os.path.join(BASE, "WaDi")
SENSOR_C = os.path.join(WA_DI, "WADI_14days_new.csv")
LABEL_C  = os.path.join(WA_DI, "WADI_attackdataLABLE.csv")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# ─── hyper-parameters ---------------------------------------------------
EPISODES   = 30
N_STEPS    = 25
BATCH_SIZE = 128
DISCOUNT   = 0.5
TN,TP,FP,FN= 1,10,-1,-10

NUM_LP     = 200          # pseudo-labels per episode
K_SLICES   = 5            # equal-slice validation
MAX_VAE_SAMPLES = 200
VAE_EPOCHS = 2
VAE_BATCH  = 32

# ─── 1) load sensor CSV (numeric only) ---------------------------------
df = pd.read_csv(SENSOR_C, decimal='.')
df.columns = df.columns.str.strip()
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna(axis=1, how="all").dropna(axis=0, how="all").reset_index(drop=True)
if "Row" in df.columns: df = df.drop(columns=["Row"])
feature_cols = df.columns.tolist(); n_features=len(feature_cols); N_INPUT=n_features+1
print(f"{n_features} numeric features retained")

# ─── 2) VAE pre-training on normal windows -----------------------------
lbl_vec = pd.read_csv(LABEL_C, header=1)["Attack LABLE (1:No Attack, -1:Attack)"].astype(int)
norm_idx = np.where(lbl_vec.values[N_STEPS:]==1)[0] + N_STEPS
sample   = np.random.choice(norm_idx, min(MAX_VAE_SAMPLES,len(norm_idx)), replace=False)
Xs = StandardScaler().fit_transform(np.array(
     [df.iloc[i-N_STEPS+1:i+1][feature_cols].values.flatten() for i in sample],"float32"))

def build_vae(dim, hid=64, lat=10):
    x=layers.Input((dim,)); h=layers.Dense(hid,"relu")(x)
    mu=layers.Dense(lat)(h); lv=tf.clip_by_value(layers.Dense(lat)(h),-10,10)
    z=layers.Lambda(lambda t:t[0]+tf.exp(0.5*t[1])*tf.random.normal(tf.shape(t[0])))((mu,lv))
    out=layers.Dense(dim,"sigmoid")(layers.Dense(hid,"relu")(z))
    vae=models.Model(x,out)
    vae.add_loss(tf.reduce_mean(losses.mse(x,out)*dim
               -0.5*tf.reduce_sum(1+lv-tf.square(mu)-tf.exp(lv),1)))
    vae.compile("adam"); return vae
vae = build_vae(N_STEPS*n_features); vae.fit(Xs, epochs=VAE_EPOCHS, batch_size=VAE_BATCH, verbose=0)

# ─── 3) state / reward helpers -----------------------------------------
def make_state(ts,c):
    if c<N_STEPS:return None
    W=ts[feature_cols].iloc[c-N_STEPS+1:c+1].values.astype("float32")
    return np.stack([np.concatenate([W,np.zeros((N_STEPS,1))],1),
                     np.concatenate([W,np.ones ((N_STEPS,1))],1)])
def reward_fn(ts,c,a,coef):
    if c<N_STEPS:return [0,0]
    win=ts[feature_cols].iloc[c-N_STEPS+1:c+1].values.flatten()[None]
    pen=coef*np.mean((vae.predict(win)-win)**2)
    base=[TN,FP] if ts['label'].iat[c]==0 else [FN,TP]
    return [base[0]+pen, base[1]+pen]

# ─── 4) LSTM-Q network (TF v1) -----------------------------------------
class QNet:
    def __init__(self,scope): self.sc=scope
    def build(self):
        with tf.compat.v1.variable_scope(self.sc):
            self.S=tf.compat.v1.placeholder(tf.float32,[None,N_STEPS,N_INPUT])
            self.T=tf.compat.v1.placeholder(tf.float32,[None,2])
            seq=tf.compat.v1.unstack(self.S,N_STEPS,1)
            out,_=tf.compat.v1.nn.static_rnn(tf.compat.v1.nn.rnn_cell.LSTMCell(128),seq,dtype=tf.float32)
            self.Q=layers.Dense(2)(out[-1])
            self.trn=tf.compat.v1.train.AdamOptimizer(3e-4).minimize(tf.reduce_mean(tf.square(self.Q-self.T)))
    def predict(self,x,ss): return ss.run(self.Q,{self.S:x})
    def update (self,x,y,ss): ss.run(self.trn,{self.S:x,self.T:y})
def copy_params(ss,src,dst):
    sv=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,src.sc)
    dv=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,dst.sc)
    for s,d in zip(sorted(sv,key=lambda v:v.name),sorted(dv,key=lambda v:v.name)): ss.run(d.assign(s))

# ─── 5) training (LP + AL) ---------------------------------------------
def train(sess, NUM_AL):
    env=EnvTimeSeriesWaDi(SENSOR_C,LABEL_C,N_STEPS); env.statefnc=make_state
    ql,qt=QNet("ql"),QNet("qt"); ql.build(); qt.build()
    sess.run(tf.compat.v1.global_variables_initializer())
    memory, coef = [], 20.0; T=namedtuple("T","s r ns d")

    for ep in range(1, EPISODES+1):
        # -------- LP + AL on previous labels ----------------------------
        env.reset(); labs=np.array(env.timeseries['label'].iloc[N_STEPS:])
        if np.any(labs!=-1):
            W=np.array([w for w in env.states_list if w is not None]); flat=W.reshape(W.shape[0],-1)
            lp=LabelSpreading(kernel='knn',n_neighbors=10).fit(flat,labs)
            uncert=1-lp.label_distributions_.max(1); idx=np.argsort(-uncert)
            # AL queries
            for i in idx[:NUM_AL]:
                env.timeseries['label'].iat[i+N_STEPS]=env.timeseries['anomaly'].iat[i+N_STEPS]
            # LP pseudo
            for i in idx[NUM_AL:NUM_AL+NUM_LP]:
                env.timeseries['label'].iat[i+N_STEPS]=lp.transduction_[i]

        env.rewardfnc=lambda ts,c,a,cf=coef: reward_fn(ts,c,a,cf)
        s,done=env.reset(),False; eps=max(0.1,1-ep/EPISODES)
        while not done:
            a=np.random.choice(2) if random.random()<eps else np.argmax(ql.predict([s],sess)[0])
            raw,r,done,_=env.step(a); ns=raw[a] if raw.ndim>2 else raw
            memory.append(T(s,r,ns,done)); s=ns
        # replay
        for _ in range(5):
            batch=random.sample(memory,min(BATCH_SIZE,len(memory)))
            S,R,NS,_=map(np.array,zip(*batch))
            qn=qt.predict(NS,sess); tgt=R+DISCOUNT*np.repeat(qn.max(1,keepdims=True),2,1)
            ql.update(S,tgt.astype("float32"),sess)
        copy_params(sess,ql,qt)
        coef=max(min(coef+0.001*np.sum(R[:,0]),100),0.1)
        print(f"[train] AL={NUM_AL} ep{ep:02}/{EPISODES} labels={np.sum(env.timeseries['label']!=-1)} coef={coef:.2f}")
    return ql

# ─── 6) equal-slice validation (SMD-style artefacts) -------------------
def run_slice(env,sess,est,prefix):
    s,done=env.reset(),False; P,G,val=[],[],[]
    while not done:
        a=np.argmax(est.predict([s],sess)[0]); P.append(a); G.append(env.timeseries['anomaly'].iat[env.timeseries_curser])
        val.append(s[-1][0]); s,_,done,_=env.step(a); s=s[a] if s.ndim>2 else s
    pr,rc,f1,_=precision_recall_fscore_support(G,P,average='binary',zero_division=0)
    au=average_precision_score(G,P)
    np.savetxt(f"{prefix}.txt",[pr,rc,f1,au],fmt="%.6f")
    fig,ax=plt.subplots(4,sharex=True,figsize=(8,6))
    ax[0].plot(val); ax[0].set_title("Time Series")
    ax[1].plot(P,'g');ax[1].set_title("Predictions")
    ax[2].plot(G,'r');ax[2].set_title("Ground Truth")
    ax[3].plot([au]*len(val),'m');ax[3].set_title("AU-PR")
    plt.tight_layout(); plt.savefig(f"{prefix}.png"); plt.close(fig)
    return f1,au
def validate(sess,est,out_dir,K=K_SLICES):
    os.makedirs(out_dir,exist_ok=True)
    base_ts=EnvTimeSeriesWaDi(SENSOR_C,LABEL_C,N_STEPS).timeseries_repo[0]
    seg=len(base_ts)//K; f1s,aus=[]
    for i in range(K):
        env=EnvTimeSeriesWaDi(SENSOR_C,LABEL_C,N_STEPS); env.statefnc=make_state
        env.timeseries_repo[0]=base_ts.iloc[i*seg:(i+1)*seg]
        f1,au=run_slice(env,sess,est,f"{out_dir}/slice_{i}"); f1s.append(f1); aus.append(au)
        print(f"[val] {out_dir} slice {i+1}/{K} F1={f1:.3f} AUPR={au:.3f}")
    print(f"[val] {out_dir} mean F1={np.mean(f1s):.3f} mean AUPR={np.mean(aus):.3f}")

# ─── 7) main loop over three AL budgets ---------------------------------
if __name__=="__main__":
    for AL_budget in [1000, 5000, 10000]:
        print(f"\n=== Active-Learning budget {AL_budget} ===")
        tf.compat.v1.reset_default_graph(); sess=tf.compat.v1.Session()
        tf.compat.v1.keras.backend.set_session(sess)
        estimator=train(sess, NUM_AL=AL_budget)
        validate(sess, estimator, out_dir=f"validation_AL{AL_budget}", K=K_SLICES)
