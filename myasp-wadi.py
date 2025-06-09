#!/usr/bin/env python3
import os, random, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import layers, models, losses
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from sklearn.ensemble import IsolationForest
from collections import namedtuple
from env_wadi import EnvTimeSeriesWaDi

# ─────────────────────────────────────────────────────────────────────────────
# 1) PATHS & HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(__file__)
SENSOR_CSV = os.path.join(BASE, "WaDi", "WADI_14days_new.csv")
LABEL_CSV  = os.path.join(BASE, "WaDi", "WADI_attackdataLABLE.csv")

N_STEPS    = 25      # length of sliding window
LATENT     = 10      # VAE latent dim
HIDDEN     = 64      # VAE hidden dim
MAX_SAMP   = 200     # # windows to pretrain VAE on
EPISODES   = 10      # RL episodes per AL budget
BATCH_SIZE = 128     # RL replay‐batch
AL_BUDGETS = [1000,5000,10000]
K_SLICES   = 3       # equal‐slice validation
LR         = 3e-4    # Q‐network learning rate
TN,TP,FP,FN = 1,10,-1,-10

print("=== WaDI: VAE-guided RL + Dynamic Reward + Active Learning ===\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2) LOAD & ALIGN SENSOR + LABEL
# ─────────────────────────────────────────────────────────────────────────────
df = (
    pd.read_csv(SENSOR_CSV, decimal=".")
      .apply(pd.to_numeric, errors="coerce")
      .dropna(axis=1, how="all")
      .dropna(axis=0, how="all")
      .reset_index(drop=True)
)
if "Row" in df.columns:
    df.drop(columns="Row", inplace=True)

raw_lbl = pd.read_csv(LABEL_CSV, header=1)["Attack LABLE (1:No Attack, -1:Attack)"]\
            .astype(int).values
anomaly = np.where(raw_lbl==-1,1,0)

L = min(len(df), len(anomaly))
df     = df.iloc[:L].reset_index(drop=True)
anomaly= anomaly[:L]

df["anomaly"] = anomaly
df["label"]   = anomaly.copy()   # we'll relabel with LP/AL later

feature_cols = df.columns.drop(["anomaly","label"]).tolist()
n_feats      = len(feature_cols)

print(f"[DATA] aligned {L} timesteps × {n_feats} numeric features.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 3) BUILD & PRETRAIN VAE (EAGER)
# ─────────────────────────────────────────────────────────────────────────────
print("[VAE] sampling up to",MAX_SAMP,"normal windows…")
norm_idx = np.where(anomaly[N_STEPS:]==0)[0] + N_STEPS
samp_idx = np.random.choice(norm_idx, min(MAX_SAMP,len(norm_idx)), replace=False)

Xw = np.stack([
    df.iloc[i-N_STEPS+1:i+1][feature_cols].values.flatten()
    for i in samp_idx
], axis=0).astype("float32")
scaler = StandardScaler().fit(Xw)
Xw_s   = scaler.transform(Xw)
print(f"[VAE] train tensor = {Xw_s.shape}\n")

def build_vae(input_dim, hidden=HIDDEN, latent=LATENT):
    x_in = layers.Input((input_dim,))
    h    = layers.Dense(hidden,activation="relu")(x_in)
    mu   = layers.Dense(latent)(h)
    lv   = tf.clip_by_value(layers.Dense(latent)(h), -10, 10)
    z    = layers.Lambda(lambda t: t[0] + tf.exp(0.5*t[1])*tf.random.normal(tf.shape(t[0])))([mu,lv])
    decoder_h = layers.Dense(hidden,activation="relu")
    x_out    = layers.Dense(input_dim,activation="sigmoid")
    dec      = x_out(decoder_h(z))
    vae      = models.Model(x_in, dec)
    recon    = losses.mse(x_in, dec)*input_dim
    kl       = -0.5*tf.reduce_sum(1+lv-tf.square(mu)-tf.exp(lv),axis=1)
    vae.add_loss(tf.reduce_mean(recon+kl))
    vae.compile(optimizer="adam")
    return vae

vae = build_vae(N_STEPS*n_feats)
vae.fit(Xw_s, epochs=2, batch_size=32, verbose=1)
print("[VAE] pretraining complete\n")

# compute per-step penalty:
print("[VAE] computing per-step reconstruction error…")
all_w = np.stack([
    df.iloc[i-N_STEPS+1:i+1][feature_cols].values.flatten()
    for i in range(N_STEPS-1, len(df))
], axis=0).astype("float32")
all_s = scaler.transform(all_w)
errs  = []
for i in range(0, len(all_s), 256):
    chunk = all_s[i:i+256]
    pred  = vae.predict(chunk, verbose=0)
    errs.append(((pred-chunk)**2).mean(axis=1))
penalty_array = np.concatenate([np.zeros(N_STEPS-1), np.concatenate(errs)])
print(f"[VAE] penalty_array ready, length={len(penalty_array)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 4) TF-1 LSTM Q-NETWORK
# ─────────────────────────────────────────────────────────────────────────────
class QNet:
    def __init__(self,scope):
        self.sc = scope
    def build(self):
        with tf.compat.v1.variable_scope(self.sc):
            self.S = tf.compat.v1.placeholder(tf.float32,[None,N_STEPS,n_feats+1],"S")
            self.T = tf.compat.v1.placeholder(tf.float32,[None,2],"T")
            seq   = tf.compat.v1.unstack(self.S,N_STEPS,1)
            out,_ = tf.compat.v1.nn.static_rnn(
                tf.compat.v1.nn.rnn_cell.LSTMCell(64), seq, dtype=tf.float32)
            self.Q = layers.Dense(2)(out[-1])
            self.train = tf.compat.v1.train.AdamOptimizer(LR)\
                         .minimize(tf.reduce_mean(tf.square(self.Q-self.T)))
    def predict(self,x,sess):
        return sess.run(self.Q,{self.S:x})
    def update(self,x,y,sess):
        sess.run(self.train,{self.S:x, self.T:y})

def copy_params(sess,src,dst):
    sv = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, src.sc)
    dv = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, dst.sc)
    for a,b in zip(sorted(sv,key=lambda v:v.name), sorted(dv,key=lambda v:v.name)):
        sess.run(b.assign(a))


# ─────────────────────────────────────────────────────────────────────────────
# 5) STATE & REWARD + DYNAMIC COEF
# ─────────────────────────────────────────────────────────────────────────────
def make_state(ts,c,prev=None,act=None):
    if c< N_STEPS: return None
    W = ts[feature_cols].iloc[c-N_STEPS+1:c+1].values.astype("float32")
    a0= np.concatenate([W, np.zeros((N_STEPS,1),dtype="float32")],1)
    a1= np.concatenate([W, np.ones ((N_STEPS,1),dtype="float32")],1)
    return np.stack([a0,a1],0)

def reward_fn(ts,c,a,coef):
    if c< N_STEPS: return [0,0]
    pen = coef * penalty_array[c]
    lbl = ts["label"].iat[c]
    base = [TN,FP] if lbl==0 else [FN,TP]
    return [base[0]+pen, base[1]+pen]

def update_coef(coef,rew,α=1e-3,lo=0.1,hi=10):
    nc = coef + α*( - rew )
    return float(np.clip(nc, lo, hi))


# ─────────────────────────────────────────────────────────────────────────────
# 6) TRAIN & VALIDATE (equal slices)
# ─────────────────────────────────────────────────────────────────────────────
def train_and_validate(AL_budget):
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    ql,qt = QNet("ql"),QNet("qt")
    ql.build(); qt.build()
    sess.run(tf.compat.v1.global_variables_initializer())
    from tensorflow.compat.v1.keras import backend as K
    K.set_session(sess)

    env = EnvTimeSeriesWaDi(SENSOR_CSV,LABEL_CSV,N_STEPS)
    env.statefnc = make_state

    Transition = namedtuple("T","s r ns d")
    memory,coef = [], 1.0

    # warmup isolation forest
    env.reset()
    W0=[]
    for i in range(N_STEPS,len(env.timeseries)):
        if env.timeseries["label"].iat[i]==0:
            W0.append(env.timeseries[feature_cols]\
                      .iloc[i-N_STEPS+1:i+1].values.flatten())
    if W0:
        IsolationForest(contamination=0.01)\
          .fit(np.array(W0[:MAX_SAMP],dtype="float32"))

    for ep in range(1,EPISODES+1):
        # LABEL PROPAGATION + ACTIVE LEARNING
        env.reset()
        labs = np.array(env.timeseries["label"].iloc[N_STEPS:])
        if (labs!=-1).any():
            Warr = np.stack([s for s in env.states_list if s is not None])
            flat = Warr.reshape(len(Warr),-1)
            lp   = LabelSpreading(n_neighbors=10).fit(flat,labs)
            uncert = 1-lp.label_distributions_.max(1)
            idx    = np.argsort(-uncert)
            # first AL_budget => reveal true
            for i in idx[:AL_budget]:
                env.timeseries["label"].iat[i+N_STEPS]=env.timeseries["anomaly"].iat[i+N_STEPS]
            # next NUM_LP => LP labels
            for i in idx[AL_budget:AL_budget+500]:
                env.timeseries["label"].iat[i+N_STEPS]=lp.transduction_[i]

        # ROLLOUT
        env.rewardfnc = lambda t,c,a: reward_fn(t,c,a,coef)
        s,done = env.reset(),False
        ep_rew = 0.0
        while not done:
            if random.random()<0.1:  # small epsilon
                a = random.choice([0,1])
            else:
                a = int(np.argmax(ql.predict([s],sess)[0]))
            raw,r,done,_ = env.step(a)
            ns = raw[a] if raw.ndim>2 else raw
            memory.append(Transition(s,r,ns,done))
            ep_rew += r[a]
            s = ns

        # REPLAY UPDATES
        for _ in range(5):
            batch = random.sample(memory, min(BATCH_SIZE,len(memory)))
            S,R,NS,_ = map(np.array, zip(*batch))
            qn = qt.predict(NS,sess)
            tgt= R + 0.5*np.repeat(qn.max(1,keepdims=True),2,1)
            ql.update(S,tgt.astype("float32"),sess)
        copy_params(sess,ql,qt)

        coef = update_coef(coef,ep_rew)
        print(f"[train AL={AL_budget}] ep{ep:02}/{EPISODES}  coef={coef:.2f}  reward={ep_rew:.2f}")

    # EQUAL‐SLICE VALIDATION
    TS0 = EnvTimeSeriesWaDi(SENSOR_CSV,LABEL_CSV,N_STEPS).timeseries_repo[0]
    seg = len(TS0)//K_SLICES
    f1s,aucs=[],[]
    outd = f"exp_AL{AL_budget}"
    os.makedirs(outd,exist_ok=True)

    for i in range(K_SLICES):
        envv = EnvTimeSeriesWaDi(SENSOR_CSV,LABEL_CSV,N_STEPS)
        envv.statefnc = make_state
        envv.timeseries_repo[0] = TS0.iloc[i*seg:(i+1)*seg].reset_index(drop=True)

        s,done=[],False
        X,Y,V=[],[],[]
        s,done=envv.reset(),False
        while not done:
            a = int(np.argmax(ql.predict([s],sess)[0]))
            X.append(a)
            Y.append(envv.timeseries["anomaly"].iat[envv.timeseries_curser])
            V.append(s[-1][0])
            raw,_,done,_=envv.step(a)
            s= raw[a] if raw.ndim>2 else raw

        p,r,f1,_ = precision_recall_fscore_support(Y,X,average="binary",zero_division=0)
        au       = average_precision_score(Y,X)
        f1s.append(f1); aucs.append(au)

        # save metrics + figure
        prefix = os.path.join(outd,f"slice_{i}")
        np.savetxt(prefix+".txt",[p,r,f1,au],fmt="%.3f")
        fig,ax=plt.subplots(4,1,figsize=(6,8),sharex=True)
        ax[0].plot(V);    ax[0].set_title("Series")
        ax[1].plot(X,'g');ax[1].set_title("Pred")
        ax[2].plot(Y,'r');ax[2].set_title("Truth")
        ax[3].plot([au]*len(V),'m');ax[3].set_title("AUPR")
        fig.savefig(prefix+".png"); plt.close(fig)

        print(f"[val AL={AL_budget}] slice {i+1}/{K_SLICES}  F1={f1:.3f}  AU={au:.3f}")

    print(f"[val AL={AL_budget}] mean F1={np.mean(f1s):.3f} mean AUPR={np.mean(aucs):.3f}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 7) RUN ALL AL BUDGETS
# ─────────────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    for AL in AL_BUDGETS:
        print(f"\n=== ACTIVE LEARNING BUDGET: {AL} ===")
        train_and_validate(AL)
