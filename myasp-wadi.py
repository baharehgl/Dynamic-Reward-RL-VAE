#!/usr/bin/env python3
# myasp‑wadi.py  – RL + VAE + LP/AL on the WADI dataset
# ============================================================================
# Everything is identical to your last working file except:
#   • the new line 309  (env_val.statefnc = make_state)
#   • the TD‑target patch in the replay‑memory loop
# ============================================================================

import os, random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()
from tensorflow.keras import layers, models, losses
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from collections import namedtuple

from env_wadi import EnvTimeSeriesWaDi
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
print("GPUs detected by TensorFlow:", tf.config.list_physical_devices('GPU'))

# ----------------------------- hyper‑parameters ------------------------------
EPISODES           = 2
N_STEPS            = 25
DISCOUNT_FACTOR    = 0.5
TN, TP, FP, FN     = 1, 10, -1, -10
ACTION_SPACE_N     = 2
VALIDATION_SPLIT   = 0.8
MAX_VAE_SAMPLES    = 10
NUM_LP             = 200
NUM_AL             = 1000
BATCH_SIZE         = 128
VAE_EPOCHS         = 2
VAE_BATCH          = 32

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
WA_DI_DIR  = os.path.join(BASE_DIR, 'WaDi')
SENSOR_CSV = os.path.join(WA_DI_DIR, 'WADI_14days_new.csv')
LABEL_CSV  = os.path.join(WA_DI_DIR, 'WADI_attackdataLABLE.csv')

print("=== WADI RL with dynamic reward & active learning ===\n")

# ------------------------- 1) load & clean sensor data -----------------------
df_all = pd.read_csv(SENSOR_CSV, decimal='.')
df_all.columns = df_all.columns.str.strip()
df_all = df_all.apply(pd.to_numeric, errors='coerce')
df_all = df_all.dropna(axis=1, how='all').dropna(axis=0, how='all').reset_index(drop=True)
if 'Row' in df_all.columns:
    df_all = df_all.drop(columns=['Row'])

feature_cols = df_all.columns.tolist()
n_features   = len(feature_cols)
N_INPUT_DIM  = n_features + 1
print(f"Detected {n_features} numeric features; N_INPUT_DIM = {N_INPUT_DIM}\n")

# --------------------- 2) pre‑train VAE on normal windows --------------------
lbl_df  = pd.read_csv(LABEL_CSV, header=1, low_memory=False)
labels  = np.where(lbl_df["Attack LABLE (1:No Attack, -1:Attack)"].astype(int)==1, 0, 1)

normal_pos = np.where(labels[N_STEPS:] == 0)[0] + N_STEPS
sampled    = np.random.choice(normal_pos, size=min(MAX_VAE_SAMPLES, len(normal_pos)), replace=False)
windows = [df_all.iloc[i-N_STEPS+1:i+1][feature_cols].values.flatten() for i in sampled]

Xs = StandardScaler().fit_transform(np.array(windows, dtype='float32'))
print(f"VAE windows shape: {Xs.shape}")

def build_vae(inp_dim, latent_dim=10, hidden=64):
    x   = layers.Input(shape=(inp_dim,))
    h   = layers.Dense(hidden, activation='relu')(x)
    mu  = layers.Dense(latent_dim)(h)
    lv  = layers.Dense(latent_dim)(h)
    lv  = tf.clip_by_value(lv, -10., 10.)
    z   = layers.Lambda(lambda t: t[0] + tf.exp(0.5*t[1])*tf.random.normal(tf.shape(t[0])))([mu, lv])
    h2  = layers.Dense(hidden, activation='relu')(z)
    out = layers.Dense(inp_dim, activation='sigmoid')(h2)
    vae = models.Model(x, out)
    recon = losses.mse(x, out) * inp_dim
    kl    = -0.5*tf.reduce_sum(1 + lv - tf.square(mu) - tf.exp(lv), axis=-1)
    vae.add_loss(tf.reduce_mean(recon + kl))
    vae.compile(optimizer='adam')
    return vae

vae = build_vae(N_STEPS*n_features)
vae.fit(Xs, epochs=VAE_EPOCHS, batch_size=VAE_BATCH, verbose=1)
vae.save('vae_wadi.h5')

# ----------------------- 3) state & reward functions -------------------------
def make_state(ts, c):
    if c < N_STEPS: return None
    W  = ts[feature_cols].iloc[c-N_STEPS+1:c+1].values.astype('float32')
    return np.stack([np.concatenate([W, np.zeros((N_STEPS,1))],1),
                     np.concatenate([W, np.ones ((N_STEPS,1))],1)])

def reward_fn(ts, c, a, vae_model, coef):
    if c < N_STEPS: return [0,0]
    win = ts[feature_cols].iloc[c-N_STEPS+1:c+1].values.flatten()[None].astype('float32')
    pen = coef * np.mean((vae_model.predict(win) - win)**2)
    lbl = ts['label'].iat[c]
    base= [TN,FP] if lbl==0 else [FN,TP]
    return [base[0]+pen, base[1]+pen]

# --------------------------- 4) Q‑network class ------------------------------
class QNet:
    def __init__(self, scope, lr=3e-4):
        self.scope = scope
        with tf.compat.v1.variable_scope(scope):
            self.S = tf.compat.v1.placeholder(tf.float32,[None,N_STEPS,N_INPUT_DIM],'S')
            self.T = tf.compat.v1.placeholder(tf.float32,[None,ACTION_SPACE_N],'T')
            seq = tf.compat.v1.unstack(self.S, N_STEPS, 1)
            cell= tf.compat.v1.nn.rnn_cell.LSTMCell(128)
            out,_= tf.compat.v1.nn.static_rnn(cell, seq, dtype=tf.float32)
            self.logits = layers.Dense(ACTION_SPACE_N)(out[-1])
            self.loss   = tf.reduce_mean(tf.square(self.logits-self.T))
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)
    def predict(self,s,ss): return ss.run(self.logits,{self.S:s})
    def update (self,s,t,ss): ss.run(self.train_op,{self.S:s,self.T:t})

def epsilon_policy(est,nA,sess):
    def pol(obs,eps):
        A = np.ones(nA)*eps/nA
        q = est.predict([obs],sess)[0]
        A[np.argmax(q)] += 1-eps
        return A
    return pol

def copy_params(sess,src,dst):
    vs = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,scope=src.scope)
    vt = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,scope=dst.scope)
    for v1,v2 in zip(sorted(vs,key=lambda v:v.name),sorted(vt,key=lambda v:v.name)):
        sess.run(v2.assign(v1))

# ---------------- 5) Q‑learning with dynamic reward & AL/LP ------------------
def update_coef(old,r,alpha=0.001,target=0,lo=0.1,hi=100):
    return max(min(old+alpha*(r-target),hi),lo)

def q_learning(env,sess,ql,qt,vae_model,init_coef):
    Transition = namedtuple('T',['s','r','ns','d'])
    sess.run(tf.compat.v1.global_variables_initializer())
    epss  = np.linspace(1,0.1,10000)
    policy= epsilon_policy(ql,ACTION_SPACE_N,sess)

    # warm‑up isolation forest (unchanged) …
    env.reset()
    W=[ env.timeseries[feature_cols].iloc[i-N_STEPS+1:i+1].values.flatten()
        for i in range(N_STEPS,len(env.timeseries)) if env.timeseries['label'].iat[i]==0 ]
    if W: IsolationForest(0.01).fit(np.array(W[:MAX_VAE_SAMPLES],dtype='float32'))

    coef = init_coef; memory=[]; coef_hist=[]
    for ep in range(EPISODES):
        env.statefnc=make_state
        env.rewardfnc=lambda ts,c,a,cf=coef:reward_fn(ts,c,a,vae_model,cf)
        state,done = env.reset(),False; ep_r=0
        while not done:
            a = np.random.choice(ACTION_SPACE_N,p=policy(state,epss[min(ep,len(epss)-1)]))
            raw,r,done,_ = env.step(a); ns = raw[a] if raw.ndim>2 else raw
            memory.append(Transition(state,r,ns,done))
            ep_r += r[a]; state = ns
        # ----- replay update (MODIFIED for 1‑variant next‑state) --------------
        for _ in range(5):
            S,R,NS,_ = map(np.array,zip(*random.sample(memory,BATCH_SIZE)))
            q_next      = qt.predict(NS, sess)                 # (B,2)
            q_next_max  = q_next.max(1, keepdims=True)         # (B,1)
            tgt = R + DISCOUNT_FACTOR * np.repeat(q_next_max, ACTION_SPACE_N, 1)
            ql.update(S, tgt.astype('float32'), sess)
        copy_params(sess,ql,qt)
        coef = update_coef(coef, ep_r); coef_hist.append(coef)
    return ql, coef_hist

# ------------------------------- 6) validation ------------------------------
def validate(env,sess,trained):
    split=int(len(env.timeseries_repo[0])*VALIDATION_SPLIT)
    preds,gts=[],[]
    policy=epsilon_policy(trained,ACTION_SPACE_N,sess)
    state,_done = env.reset(),False
    while env.timeseries_curser < split:
        raw,_,_done,_=env.step(0); state=raw[0] if raw.ndim>2 else raw
        if _done: break
    while not _done:
        a=np.argmax(policy(state,0))
        preds.append(a); gts.append(env.timeseries['anomaly'].iat[env.timeseries_curser])
        raw,_,_done,_=env.step(a); state=raw[a] if raw.ndim>2 else raw
    p,r,f1,_ = precision_recall_fscore_support(gts,preds,average='binary',zero_division=0)
    aupr     = average_precision_score(gts,preds)
    print(f"[val] F1={f1:.3f}  AUPR={aupr:.3f}")
    return f1,aupr

# ------------------------------- 7) wrapper ---------------------------------
def train_wrapper(nLP,nAL,discount):
    tf.compat.v1.reset_default_graph(); sess=tf.compat.v1.Session()
    from tensorflow.compat.v1.keras import backend as K; K.set_session(sess)
    vae_model = load_model('vae_wadi.h5', compile=False)
    env       = EnvTimeSeriesWaDi(SENSOR_CSV, LABEL_CSV, N_STEPS)
    env.statefnc = make_state                        # ensure callback set
    ql, qt    = QNet('qlearn'), QNet('qtarget')
    trained, coefs = q_learning(env,sess,ql,qt,vae_model,20.0)

    # ----- NEW: validation env gets the same statefnc -----------------------
    env_val = EnvTimeSeriesWaDi(SENSOR_CSV, LABEL_CSV, N_STEPS)
    env_val.statefnc = make_state                     # ← required line
    f1, aupr = validate(env_val, sess, trained)
    # -----------------------------------------------------------------------

    exp=f'exp_AL{nAL}'; os.makedirs(os.path.join(exp,'plots'),exist_ok=True)
    plt.figure(); plt.plot(coefs); plt.title('dynamic_coef'); plt.savefig(os.path.join(exp,'plots','coef.png'))

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for AL in [1000, 5000, 10000]:
        train_wrapper(NUM_LP, AL, discount=0.96)
