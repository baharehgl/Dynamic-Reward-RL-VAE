#!/usr/bin/env python3
# wadi_rl_equal_al.py
import os, random, numpy as np, pandas as pd, tensorflow as tf
from collections import namedtuple
tf.compat.v1.disable_eager_execution()

from tensorflow.keras import layers, models, losses
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from env_wadi import EnvTimeSeriesWaDi

# ─── paths & GPU ───────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
WA_DI = os.path.join(BASE, "WaDi")
SENSOR, LABEL = os.path.join(WA_DI,"WADI_14days_new.csv"), os.path.join(WA_DI,"WADI_attackdataLABLE.csv")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# ─── hyperparameters ───────────────────────────────
EPISODES, N_STEPS, BATCH_SIZE = 30, 25, 128
DISCOUNT       = 0.5
TN,TP,FP,FN    = 1,10,-1,-10
NUM_LP, K_SLICES = 200, 5
MAX_VAE_SAMPLES  = 200
VAE_EPOCHS, VAE_BATCH = 2, 32

# ─── load numeric columns ──────────────────────────
df = pd.read_csv(SENSOR, decimal='.')
df.columns = df.columns.str.strip()
df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1,how="all")\
        .dropna(axis=0,how="all").reset_index(drop=True)
if "Row" in df.columns: df = df.drop(columns=["Row"])
feature_cols = df.columns.tolist()
n_features   = len(feature_cols)
N_INPUT       = n_features + 1

# ─── Q-network definition (TF1 style) ─────────────
class QNet:
    def __init__(self, scope):
        self.sc = scope
    def build(self):
        with tf.compat.v1.variable_scope(self.sc):
            self.S = tf.compat.v1.placeholder(tf.float32, [None, N_STEPS, N_INPUT])
            self.T = tf.compat.v1.placeholder(tf.float32, [None, 2])
            seq = tf.compat.v1.unstack(self.S, N_STEPS, 1)
            out,_ = tf.compat.v1.nn.static_rnn(
                tf.compat.v1.nn.rnn_cell.LSTMCell(128), seq, dtype=tf.float32)
            self.Q = layers.Dense(2)(out[-1])
            self.trn = tf.compat.v1.train.AdamOptimizer(3e-4).minimize(
                tf.reduce_mean(tf.square(self.Q - self.T)))
    def predict(self, x, sess): return sess.run(self.Q, {self.S: x})
    def update (self, x, y, sess): sess.run(self.trn, {self.S: x, self.T: y})

def copy_params(sess, src, dst):
    sv = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, src.sc)
    dv = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, dst.sc)
    for s, d in zip(sorted(sv, key=lambda v:v.name), sorted(dv, key=lambda v:v.name)):
        sess.run(d.assign(s))

# ─── state & reward helpers ────────────────────────
def make_state(ts, c):
    if c < N_STEPS: return None
    W = ts[feature_cols].iloc[c-N_STEPS+1:c+1].values.astype("float32")
    return np.stack([np.concatenate([W, np.zeros((N_STEPS,1))],1),
                     np.concatenate([W, np.ones ((N_STEPS,1))],1)])
def reward_fn(ts, c, a, coef, vae):
    if c < N_STEPS: return [0, 0]
    win = ts[feature_cols].iloc[c-N_STEPS+1:c+1].values.flatten()[None]
    pen = coef * np.mean((vae.predict(win) - win)**2)
    base = [TN,FP] if ts['label'].iat[c]==0 else [FN,TP]
    return [base[0] + pen, base[1] + pen]

# ─── build & pre-train VAE inside the session ───────
def build_vae(orig_dim, hid=64, lat=10):
    x_in = layers.Input((orig_dim,))
    h    = layers.Dense(hid, activation='relu')(x_in)
    z_mu = layers.Dense(lat)(h)
    z_lv = tf.clip_by_value(layers.Dense(lat)(h), -10, 10)
    z    = layers.Lambda(lambda t: t[0] + tf.exp(0.5*t[1]) *
                         tf.random.normal(tf.shape(t[0])))([z_mu, z_lv])
    enc  = models.Model(x_in, [z_mu, z_lv, z], name="encoder")
    z_in = layers.Input((lat,))
    d_h  = layers.Dense(hid, activation='relu')(z_in)
    x_out= layers.Dense(orig_dim, activation='sigmoid')(d_h)
    dec  = models.Model(z_in, x_out, name="decoder")
    recon = dec(z)
    vae = models.Model(x_in, recon, name="vae")
    rl  = losses.mse(x_in, recon) * orig_dim
    kl  = -0.5 * tf.reduce_sum(1 + z_lv - tf.square(z_mu) - tf.exp(z_lv), axis=1)
    vae.add_loss(tf.reduce_mean(rl + kl))
    vae.compile(optimizer='adam')
    return vae, enc, dec

# ─── full training function ──────────────────────────
def train_and_validate(AL_budget):
    # — TF1 graph + session
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    K.set_session(sess)

    # — Q-networks
    ql, qt = QNet("ql"), QNet("qt")
    ql.build(); qt.build()
    sess.run(tf.compat.v1.global_variables_initializer())

    # — prepare environment
    env = EnvTimeSeriesWaDi(SENSOR_C, LABEL_C, N_STEPS)
    env.statefnc = make_state

    # — prepare VAE data (normal windows)
    lbl_vec = pd.read_csv(LABEL_C, header=1)["Attack LABLE (1:No Attack, -1:Attack)"].astype(int)
    norm_idx= np.where(lbl_vec.values[N_STEPS:]==1)[0] + N_STEPS
    sample  = np.random.choice(norm_idx, min(MAX_VAE_SAMPLES, len(norm_idx)), replace=False)
    windows = [df.iloc[i-N_STEPS+1:i+1][feature_cols].values.flatten() for i in sample]
    Xs      = StandardScaler().fit_transform(np.array(windows, "float32"))

    # — build & train VAE
    vae, encoder, decoder = build_vae(N_STEPS * n_features)
    print(f"\n[VAE] training on {len(Xs)} windows...")
    vae.fit(Xs, epochs=VAE_EPOCHS, batch_size=VAE_BATCH, verbose=1)

    # — warm-up IsolationForest (optional)
    env.reset()
    W = []
    for i in range(N_STEPS, len(env.timeseries)):
        if env.timeseries['label'].iat[i] == 0:
            W.append(env.timeseries[feature_cols].iloc[i-N_STEPS+1:i+1].values.flatten())
    if W:
        IsolationForest(contamination=0.01).fit(np.array(W[:MAX_VAE_SAMPLES], "float32"))

    # — replay memory
    Transition = namedtuple("T", "s r ns d")
    memory, coef = [], 20.0

    # — main RL loop with LP + AL
    for ep in range(1, EPISODES+1):
        # LP + AL
        env.reset()  # builds states_list
        labs = np.array(env.timeseries['label'].iloc[N_STEPS:])
        if np.any(labs != -1):
            W = np.array([w for w in env.states_list if w is not None])
            flat = W.reshape(W.shape[0], -1)
            lp = LabelSpreading(kernel='knn', n_neighbors=10).fit(flat, labs)
            uncert = 1 - lp.label_distributions_.max(axis=1)
            idx    = np.argsort(-uncert)
            # AL queries
            for i in idx[:AL_budget]:
                env.timeseries['label'].iat[i+N_STEPS] = env.timeseries['anomaly'].iat[i+N_STEPS]
            # LP pseudo-labels
            for i in idx[AL_budget:AL_budget+NUM_LP]:
                env.timeseries['label'].iat[i+N_STEPS] = lp.transduction_[i]

        # rollout
        env.rewardfnc = lambda ts,c,a,cf=coef: reward_fn(ts,c,a,cf,vae)
        s, done = env.reset(), False
        eps      = max(0.1, 1 - ep/EPISODES)
        while not done:
            if random.random() < eps:
                a = random.choice([0,1])
            else:
                a = np.argmax(ql.predict([s], sess)[0])
            raw, r, done, _ = env.step(a)
            ns = raw[a] if raw.ndim>2 else raw
            memory.append(Transition(s, r, ns, done))
            s = ns

        # replay updates
        for _ in range(5):
            batch = random.sample(memory, min(BATCH_SIZE, len(memory)))
            S, R, NS, _ = map(np.array, zip(*batch))
            qn = qt.predict(NS, sess)
            tgt = R + DISCOUNT * np.repeat(qn.max(1,keepdims=True), 2, 1)
            ql.update(S, tgt.astype("float32"), sess)
        copy_params(sess, ql, qt)

        # update coef
        coef = max(min(coef + 0.001 * np.sum(R[:,0]), 100), 0.1)
        print(f"[train AL={AL_budget}] ep {ep:02}/{EPISODES}  coef={coef:.2f}")

    # — validation on K equal slices (SMD style)
    full_ts = EnvTimeSeriesWaDi(SENSOR_C, LABEL_C, N_STEPS).timeseries_repo[0]
    seg = len(full_ts) // K_SLICES
    os.makedirs(f"validation_AL{AL_budget}", exist_ok=True)
    f1s, aus = [], []

    for i in range(K_SLICES):
        env_val = EnvTimeSeriesWaDi(SENSOR_C, LABEL_C, N_STEPS)
        env_val.statefnc = make_state
        env_val.timeseries_repo[0] = full_ts.iloc[i*seg:(i+1)*seg]

        s, done = env_val.reset(), False
        preds, gts, vals = [], [], []
        while not done:
            a = np.argmax(ql.predict([s], sess)[0])
            preds.append(a)
            gts.append(env_val.timeseries['anomaly'].iat[env_val.timeseries_curser])
            vals.append(s[-1][0])
            nxt,_,done,_ = env_val.step(a)
            s = nxt[a] if nxt.ndim>2 else nxt

        p, r, f1, _ = precision_recall_fscore_support(gts, preds, average='binary', zero_division=0)
        aupr = average_precision_score(gts, preds)
        f1s.append(f1); aus.append(aupr)

        prefix = f"validation_AL{AL_budget}/slice_{i}"
        np.savetxt(prefix+".txt", [p,r,f1,aupr], fmt="%.6f")
        fig, ax = plt.subplots(4, sharex=True, figsize=(8,6))
        ax[0].plot(vals); ax[0].set_title("Time Series")
        ax[1].plot(preds,'g'); ax[1].set_title("Predictions")
        ax[2].plot(gts,'r'); ax[2].set_title("Ground Truth")
        ax[3].plot([aupr]*len(vals),'m'); ax[3].set_title("AU-PR")
        plt.tight_layout(); plt.savefig(prefix+".png"); plt.close(fig)

        print(f"[val AL={AL_budget}] slice {i+1}/{K_SLICES}  F1={f1:.3f}  AUPR={aupr:.3f}")

    print(f"[val AL={AL_budget}] mean F1={np.mean(f1s):.3f}  mean AUPR={np.mean(aus):.3f}")

# ─── driver: three budgets ──────────────────────────────
if __name__ == "__main__":
    for AL_budget in [1000, 5000, 10000]:
        print(f"\n=== Active-Learning budget: {AL_budget} ===")
        train_and_validate(AL_budget)
