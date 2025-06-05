#!/usr/bin/env python3
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

# ─── 1) PRE-TRAIN VAE IN EAGER MODE ─────────────────────────────────
from tensorflow.keras import layers, models, losses
from sklearn.preprocessing import StandardScaler

# file paths
BASE   = os.path.dirname(__file__)
WA_DI  = os.path.join(BASE, "WaDi")
SENSOR = os.path.join(WA_DI, "WADI_14days_new.csv")
LABEL  = os.path.join(WA_DI, "WADI_attackdataLABLE.csv")

# ─── Load & clean sensor data ────────────────────────────────────────
df = pd.read_csv(SENSOR, decimal='.')
df.columns = df.columns.str.strip()

# Convert non-numeric → NaN, then drop columns that are ALL NaN, then drop rows that are ALL NaN
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(axis=1, how='all')   # only drop a column if every entry is NaN
df = df.dropna(axis=0, how='all').reset_index(drop=True)
if 'Row' in df.columns:
    df = df.drop(columns=['Row'])

feature_cols = df.columns.tolist()
n_features   = len(feature_cols)
N_STEPS       = 25

# ─── Load attack labels and align length ─────────────────────────────
lbl_all = pd.read_csv(LABEL, header=1)["Attack LABLE (1:No Attack, -1:Attack)"].astype(int).values
lbl_all = lbl_all[: len(df)]  # trim to match cleaned df length

# Identify “normal” timesteps (label == 1 means “no attack”)
norm_idx = np.where(lbl_all[N_STEPS:] == 1)[0] + N_STEPS

# ─── Sample up to MAX_VAE_SAMPLES normal windows ──────────────────────
MAX_VAE_SAMPLES = 200
sample_idx      = np.random.choice(
    norm_idx,
    size=min(MAX_VAE_SAMPLES, len(norm_idx)),
    replace=False
)

print(f"[VAE] sampling {len(sample_idx)} normal windows of length {N_STEPS}")

windows = np.array([
    df.iloc[i - N_STEPS + 1 : i + 1][feature_cols].values.flatten()
    for i in sample_idx
], dtype='float32')

# Fit one StandardScaler on THESE sampled windows—and keep it for both training & penalty calculation
scaler = StandardScaler().fit(windows)
Xs = scaler.transform(windows)
print(f"[VAE] training tensor shape = {Xs.shape}")

# ─── Define VAE (eager) ──────────────────────────────────────────────
def build_vae(input_dim, hidden=64, latent=10):
    # Encoder
    x_in = layers.Input((input_dim,))
    h    = layers.Dense(hidden, activation='relu')(x_in)
    z_mu = layers.Dense(latent)(h)
    z_lv = tf.clip_by_value(layers.Dense(latent)(h), -10.0, 10.0)
    z    = layers.Lambda(
        lambda t: t[0] + tf.exp(0.5 * t[1]) * tf.random.normal(tf.shape(t[0]))
    )([z_mu, z_lv])
    encoder = models.Model(x_in, [z_mu, z_lv, z], name="encoder")

    # Decoder
    z_in  = layers.Input((latent,))
    dh    = layers.Dense(hidden, activation='relu')(z_in)
    x_out = layers.Dense(input_dim, activation='sigmoid')(dh)
    decoder = models.Model(z_in, x_out, name="decoder")

    # VAE = encoder → decoder
    recon = decoder(z)
    vae   = models.Model(x_in, recon, name="vae")
    rl = losses.mse(x_in, recon) * input_dim
    kl = -0.5 * tf.reduce_sum(1 + z_lv - tf.square(z_mu) - tf.exp(z_lv), axis=1)
    vae.add_loss(tf.reduce_mean(rl + kl))
    vae.compile(optimizer="adam")
    return vae, encoder, decoder

vae, encoder, decoder = build_vae(N_STEPS * n_features)
vae.summary()

# Train the VAE (just 2 epochs here)
vae.fit(Xs, epochs=2, batch_size=32, verbose=1)
print("[VAE] pretraining complete")

# ─── Compute per-step reconstruction error (penalty_array) ───────────
print("[VAE] computing per-step reconstruction error")

all_windows = np.array([
    df.iloc[i - N_STEPS + 1 : i + 1][feature_cols].values.flatten()
    for i in range(N_STEPS - 1, len(df))
], dtype='float32')

# Use *the exact same* scaler that we fitted on “windows”
all_windows_scaled = scaler.transform(all_windows)

# Predict in reasonably small batches to avoid OOM
batch_size = 64
errs = []
for start in range(0, len(all_windows_scaled), batch_size):
    chunk = all_windows_scaled[start : start + batch_size]
    pred  = vae.predict(chunk, batch_size=chunk.shape[0], verbose=0)
    errs.append(np.mean((pred - chunk) ** 2, axis=1))

recon_err = np.concatenate(errs, axis=0)
# Prepend zeros for the first (N_STEPS−1) positions
penalty_array = np.concatenate([np.zeros(N_STEPS - 1), recon_err])
print(f"[VAE] penalty_array ready, length = {len(penalty_array)}")
print(f"[VAE] raw penalty stats: min={np.nanmin(penalty_array):.4f}, "
      f"max={np.nanmax(penalty_array):.4f}, mean={np.nanmean(penalty_array):.4f}")

# ─── 2) RL TRAINING & VALIDATION IN TF-1 GRAPH ────────────────────────
tf.compat.v1.disable_eager_execution()
K = tf.compat.v1.keras.backend

from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from env_wadi import EnvTimeSeriesWaDi

# Hyperparameters
EPISODES   = 10
BATCH_SIZE = 64
DISCOUNT   = 0.5
TN,TP,FP,FN= 1,10,-1,-10
NUM_LP     = 200
K_SLICES   = 5

# ─── Q-Network (TF1-style) ───────────────────────────────────────────
class QNet:
    def __init__(self, scope):
        self.sc = scope

    def build(self):
        with tf.compat.v1.variable_scope(self.sc):
            # Input shape: [batch, N_STEPS, n_features+1]
            self.S = tf.compat.v1.placeholder(tf.float32, [None, N_STEPS, n_features + 1])
            self.T = tf.compat.v1.placeholder(tf.float32, [None, 2])
            seq, _ = tf.compat.v1.nn.static_rnn(
                tf.compat.v1.nn.rnn_cell.LSTMCell(64),
                tf.compat.v1.unstack(self.S, N_STEPS, axis=1),
                dtype=tf.float32
            )
            self.Q = layers.Dense(2)(seq[-1])
            self.loss = tf.reduce_mean(tf.square(self.Q - self.T))
            self.train_op = tf.compat.v1.train.AdamOptimizer(3e-4).minimize(self.loss)

    def predict(self, x, sess):
        return sess.run(self.Q, {self.S: x})

    def update(self, x, y, sess):
        sess.run(self.train_op, {self.S: x, self.T: y})

def copy_params(sess, src: QNet, dst: QNet):
    sv = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, src.sc)
    dv = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, dst.sc)
    for s_var, d_var in zip(sorted(sv, key=lambda v: v.name), sorted(dv, key=lambda v: v.name)):
        sess.run(d_var.assign(s_var))

# ─── State & Reward using “penalty_array” ────────────────────────────
def make_state(ts, c):
    if c < N_STEPS:
        return None
    W = ts[feature_cols].iloc[c - N_STEPS + 1 : c + 1].values.astype('float32')
    a0 = np.concatenate([W, np.zeros((N_STEPS, 1), dtype='float32')], axis=1)
    a1 = np.concatenate([W, np.ones ((N_STEPS, 1), dtype='float32')], axis=1)
    return np.stack([a0, a1], axis=0)

def reward_fn(ts, c, a, coef):
    if c < N_STEPS:
        return [0, 0]
    pen = coef * float(penalty_array[c])
    lbl = ts['label'].iat[c]
    base = [TN, FP] if lbl == 0 else [FN, TP]
    return [base[0] + pen, base[1] + pen]

# ─── Full train + validate ──────────────────────────────────────────
def train_and_validate(AL_budget):
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    K.set_session(sess)

    # Build Q-networks and initialize
    ql = QNet("ql"); ql.build()
    qt = QNet("qt"); qt.build()
    sess.run(tf.compat.v1.global_variables_initializer())

    # Initialize the WaDi environment
    env = EnvTimeSeriesWaDi(SENSOR, LABEL, N_STEPS)
    env.statefnc = make_state

    from collections import namedtuple
    Transition = namedtuple("T", ["s", "r", "ns", "d"])
    memory, coef = [], 20.0

    # Optional warm-up IsolationForest
    W0 = []
    env.reset()
    for i in range(N_STEPS, len(env.timeseries)):
        if env.timeseries['label'].iat[i] == 0:
            W0.append(env.timeseries[feature_cols].iloc[i - N_STEPS + 1 : i + 1].values.flatten())
    if W0:
        from sklearn.ensemble import IsolationForest
        IsolationForest(contamination=0.01).fit(np.array(W0[:MAX_VAE_SAMPLES], dtype='float32'))

    # ─── Training episodes ─────────────────────────────────────────────
    for ep in range(1, EPISODES + 1):
        # 1) Label-Propagation + Active Learning before rollout
        env.reset()
        labs = np.array(env.timeseries['label'].iloc[N_STEPS:])
        if np.any(labs != -1):
            Warr = np.array([w for w in env.states_list if w is not None])
            flat = Warr.reshape(Warr.shape[0], -1)
            lp   = LabelSpreading(kernel='knn', n_neighbors=10).fit(flat, labs)
            uncert = 1.0 - lp.label_distributions_.max(axis=1)
            idx    = np.argsort(-uncert)
            # Top AL_budget → assign true anomaly label
            for i in idx[:AL_budget]:
                env.timeseries['label'].iat[i + N_STEPS] = env.timeseries['anomaly'].iat[i + N_STEPS]
            # Next NUM_LP → assign LP’s pseudo‐label
            for i in idx[AL_budget : AL_budget + NUM_LP]:
                env.timeseries['label'].iat[i + N_STEPS] = lp.transduction_[i]

        # 2) Rollout with ε-greedy
        env.rewardfnc = lambda ts, c, a: reward_fn(ts, c, a, coef)
        s, done = env.reset(), False
        eps = max(0.1, 1.0 - ep / EPISODES)
        ep_reward = 0.0

        while not done:
            if random.random() < eps:
                a = random.choice([0, 1])
            else:
                a = np.argmax(ql.predict([s], sess)[0])

            raw, r, done, _ = env.step(a)
            ns = raw[a] if raw.ndim > 2 else raw
            memory.append(Transition(s, r, ns, done))
            ep_reward += float(r[a])
            s = ns

        # 3) Replay‐memory updates
        for _ in range(5):
            batch = random.sample(memory, min(BATCH_SIZE, len(memory)))
            S_batch, R_batch, NS_batch, _ = map(np.array, zip(*batch))
            qn = qt.predict(NS_batch, sess)
            qn_max = np.max(qn, axis=1, keepdims=True)
            tgt = R_batch + DISCOUNT * np.repeat(qn_max, 2, axis=1)
            ql.update(S_batch, tgt.astype('float32'), sess)

        copy_params(sess, ql, qt)

        # 4) Update dynamic‐reward coefficient
        coef = max(min(coef + 0.001 * ep_reward, 10.0), 0.1)
        print(f"[train AL={AL_budget}] ep{ep:02d}/{EPISODES} coef={coef:.2f} reward={ep_reward:.2f}")

    # ─── Equal-slice validation ───────────────────────────────────────
    base_ts = EnvTimeSeriesWaDi(SENSOR, LABEL, N_STEPS).timeseries_repo[0]
    seg     = len(base_ts) // K_SLICES
    outdir  = f"validation_AL{AL_budget}"
    os.makedirs(outdir, exist_ok=True)
    f1s, aus = [], []

    for i in range(K_SLICES):
        envv = EnvTimeSeriesWaDi(SENSOR, LABEL, N_STEPS)
        envv.statefnc           = make_state
        envv.timeseries_repo[0] = base_ts.iloc[i * seg : (i + 1) * seg].reset_index(drop=True)

        s, done = envv.reset(), False
        P, G, V = [], [], []

        while not done:
            a = np.argmax(ql.predict([s], sess)[0])
            P.append(a)
            G.append(envv.timeseries['anomaly'].iat[envv.timeseries_curser])
            V.append(s[-1][0])

            nxt, _, done, _ = envv.step(a)
            s = nxt[a] if nxt.ndim > 2 else nxt

        p, r, f1, _ = precision_recall_fscore_support(G, P, average='binary', zero_division=0)
        au          = average_precision_score(G, P)
        f1s.append(f1); aus.append(au)

        prefix = f"{outdir}/slice_{i}"
        np.savetxt(prefix + ".txt", [p, r, f1, au], fmt="%.6f")
        fig, ax = plt.subplots(4, sharex=True, figsize=(8, 6))
        ax[0].plot(V);      ax[0].set_title("Time Series")
        ax[1].plot(P, 'g'); ax[1].set_title("Predictions")
        ax[2].plot(G, 'r'); ax[2].set_title("Ground Truth")
        ax[3].plot([au] * len(V), 'm'); ax[3].set_title("AUPR")
        plt.tight_layout(); plt.savefig(prefix + ".png"); plt.close(fig)

        print(f"[val AL={AL_budget}] slice{i+1}/{K_SLICES}  F1={f1:.3f}  AU={au:.3f}")

    print(f"[val AL={AL_budget}] mean F1={np.mean(f1s):.3f}  mean AUPR={np.mean(aus):.3f}\n")


# ─── DRIVER LOOP ─────────────────────────────────────────────────────
if __name__ == "__main__":
    for AL in [1000, 5000, 10000]:
        print(f"\n=== ACTIVE LEARNING BUDGET: {AL} ===")
        train_and_validate(AL)
