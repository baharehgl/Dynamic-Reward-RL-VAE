#!/usr/bin/env python3
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

# ─── 1) PRETRAIN VAE IN EAGER MODE ────────────────────────────────────
from tensorflow.keras import layers, models, losses
from sklearn.preprocessing import StandardScaler

# ─── Paths ────────────────────────────────────────────────────────────
BASE   = os.path.dirname(__file__)
WA_DI  = os.path.join(BASE, "WaDi")
SENSOR = os.path.join(WA_DI, "WADI_14days_new.csv")
LABEL  = os.path.join(WA_DI, "WADI_attackdataLABLE.csv")

# ─── A) Load & clean the sensor CSV ───────────────────────────────────
df = pd.read_csv(SENSOR, decimal='.')
df.columns = df.columns.str.strip()

# Convert everything to numeric; drop any column that becomes all-NaN, then drop any row that becomes all-NaN.
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(axis=1, how='all')
df = df.dropna(axis=0, how='all').reset_index(drop=True)

# Drop an optional “Row” column if present
if 'Row' in df.columns:
    df = df.drop(columns=['Row'])

print(f"[DATA] After cleaning, {len(df.columns)} numeric columns remain, {len(df)} rows total.")

# ─── B) Load the attack‐label CSV (might be shorter) ─────────────────
lbl_df   = pd.read_csv(LABEL, header=1, low_memory=False)
raw_lbl  = lbl_df["Attack LABLE (1:No Attack, -1:Attack)"].astype(int).values
anomaly_full = np.where(raw_lbl == 1, 0, 1)  # 1→0 normal, -1→1 anomaly

# Truncate sensor df and label array to the same minimum length
min_len = min(len(df), len(anomaly_full))
df       = df.iloc[:min_len].reset_index(drop=True)
anomaly  = anomaly_full[:min_len]
print(f"[DATA] Truncated both data and labels to length = {min_len}.")

# ─── C) Build the full timeseries DataFrame for RL ───────────────────
TS = df.copy()
TS["anomaly"] = anomaly
TS["label"]   = -1
print(f"[DATA] Final TS.shape = {TS.shape} (features + anomaly + label).")

# Keep a list of feature‐column names (all numeric columns)
feature_cols_full = df.columns.tolist()

# ─── D) Sample “normal” windows (anomaly==0) for VAE pretraining ─────
N_STEPS         = 25
MAX_VAE_SAMPLES = 200

normal_idx = np.where(TS["anomaly"].values[N_STEPS:] == 0)[0] + N_STEPS
sample_idx = np.random.choice(
    normal_idx,
    size=min(MAX_VAE_SAMPLES, len(normal_idx)),
    replace=False
)
print(f"[VAE] Sampling {len(sample_idx)} normal windows of length {N_STEPS}.")

# Build a (num_samples × (N_STEPS·n_features)) array
windows = np.array([
    TS.iloc[i - N_STEPS + 1 : i + 1][feature_cols_full].values.flatten()
    for i in sample_idx
], dtype="float32")

# Drop any column that has zero variance across these sampled windows
col_std = windows.std(axis=0)
zero_std_cols = np.where(col_std == 0)[0]
if len(zero_std_cols) > 0:
    print(f"[VAE] Dropping {len(zero_std_cols)} constant columns.")
    keep_mask = col_std > 0
    windows = windows[:, keep_mask]
    feature_cols = [feature_cols_full[i] for i in range(len(feature_cols_full)) if keep_mask[i]]
    n_features   = len(feature_cols)
    print(f"[VAE] Remaining features after drop = {n_features}")
else:
    feature_cols = feature_cols_full.copy()
    n_features   = len(feature_cols_full)
    print(f"[VAE] Using all {n_features} features (none were constant).")

# Standardize those sampled windows
scaler = StandardScaler().fit(windows)
Xs     = scaler.transform(windows)
print(f"[VAE] Training tensor shape = {Xs.shape}")

# ─── E) Build & pretrain the VAE (TF‐2/Eager) ─────────────────────────
def build_vae(input_dim, hidden=64, latent=10):
    x_in = layers.Input((input_dim,))
    h    = layers.Dense(hidden, activation="relu")(x_in)
    z_mu = layers.Dense(latent)(h)
    z_lv = tf.clip_by_value(layers.Dense(latent)(h), -10.0, 10.0)
    z    = layers.Lambda(
        lambda t: t[0] + tf.exp(0.5 * t[1]) * tf.random.normal(tf.shape(t[0]))
    )([z_mu, z_lv])
    encoder = models.Model(x_in, [z_mu, z_lv, z], name="encoder")

    z_in  = layers.Input((latent,))
    dh    = layers.Dense(hidden, activation="relu")(z_in)
    x_out = layers.Dense(input_dim, activation="sigmoid")(dh)
    decoder = models.Model(z_in, x_out, name="decoder")

    recon = decoder(z)
    vae   = models.Model(x_in, recon, name="vae")
    rl = losses.mse(x_in, recon) * input_dim
    kl = -0.5 * tf.reduce_sum(1 + z_lv - tf.square(z_mu) - tf.exp(z_lv), axis=1)
    vae.add_loss(tf.reduce_mean(rl + kl))
    vae.compile(optimizer="adam")
    return vae, encoder, decoder

vae, encoder, decoder = build_vae(N_STEPS * n_features)
vae.summary()

vae.fit(Xs, epochs=2, batch_size=32, verbose=1)
print("[VAE] Pretraining complete")

# ─── F) Compute the per‐step reconstruction error array ────────────────
print("[VAE] Computing per‐step reconstruction error")
all_windows = np.array([
    TS.iloc[i - N_STEPS + 1 : i + 1][feature_cols].values.flatten()
    for i in range(N_STEPS - 1, len(TS))
], dtype="float32")

all_windows_scaled = scaler.transform(all_windows)
batch_size = 64
errs = []
for start in range(0, len(all_windows_scaled), batch_size):
    chunk = all_windows_scaled[start : start + batch_size]
    pred  = vae.predict(chunk, batch_size=chunk.shape[0], verbose=0)
    errs.append(np.mean((pred - chunk) ** 2, axis=1))

recon_err = np.concatenate(errs, axis=0)
penalty_array = np.concatenate([np.zeros(N_STEPS - 1), recon_err])
print(f"[VAE] Penalty array length = {len(penalty_array)}  ")
print(f"[VAE] raw penalty stats: min={np.nanmin(penalty_array):.4e}, "
      f"max={np.nanmax(penalty_array):.4e}, mean={np.nanmean(penalty_array):.4e}")

# ─── 2) RL TRAINING & VALIDATION IN TF‐1 GRAPH ─────────────────────────
tf.compat.v1.disable_eager_execution()
K = tf.compat.v1.keras.backend

from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Environment “wrapper” over TS DataFrame ──────────────────────────
class EnvWaDi:
    """
    At reset(), cursor t=N_STEPS.  Each step(action) → t+1.
    Returns next state = two (N_STEPS×(n_features+1)) arrays (one for action=0, one for action=1),
    reward = [r0, r1], done flag.
    """
    def __init__(self, timeseries_df, statefnc, rewardfnc):
        self.timeseries     = timeseries_df.copy().reset_index(drop=True)
        self.N              = len(timeseries_df)
        self.statefnc      = statefnc
        self.rewardfnc     = rewardfnc
        self.t0             = N_STEPS
        self.action_space_n = 2

    def reset(self):
        self.t = self.t0
        # Precompute states_list (flattened “action=0” slices) for LP/AL
        self.states_list = []
        for c in range(self.t0, self.N):
            s = self.statefnc(self.timeseries, c)
            if s is not None:
                # store only the action=0 window, flattened
                self.states_list.append(s[0].flatten())
        return self.statefnc(self.timeseries, self.t)

    def step(self, action):
        r = self.rewardfnc(self.timeseries, self.t, action)
        self.t += 1
        done = int(self.t >= self.N)
        if not done:
            s_next = self.statefnc(self.timeseries, self.t)
        else:
            # If done, repeat last valid state so shapes match:
            s_next = self.statefnc(self.timeseries, self.N - 1)
        return s_next, r, done, {}

# ─── Q‐Network (TF‐1 style) ────────────────────────────────────────────
class QNet:
    def __init__(self, scope):
        self.sc = scope

    def build(self):
        with tf.compat.v1.variable_scope(self.sc):
            self.S = tf.compat.v1.placeholder(tf.float32, [None, N_STEPS, n_features + 1])
            self.T = tf.compat.v1.placeholder(tf.float32, [None, 2])
            seq, _ = tf.compat.v1.nn.static_rnn(
                tf.compat.v1.nn.rnn_cell.LSTMCell(64),
                tf.compat.v1.unstack(self.S, N_STEPS, axis=1),
                dtype=tf.float32
            )
            self.Q    = layers.Dense(2)(seq[-1])
            self.loss = tf.reduce_mean(tf.square(self.Q - self.T))
            self.train = tf.compat.v1.train.AdamOptimizer(3e-4).minimize(self.loss)

    def predict(self, x, sess):
        return sess.run(self.Q, {self.S: x})

    def update(self, x, y, sess):
        sess.run(self.train, {self.S: x, self.T: y})

def copy_params(sess, src: QNet, dst: QNet):
    sv = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, src.sc)
    dv = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, dst.sc)
    for s_var, d_var in zip(sorted(sv, key=lambda v: v.name), sorted(dv, key=lambda v: v.name)):
        sess.run(d_var.assign(s_var))

# ─── State & Reward using penalty_array ───────────────────────────────
def make_state(ts_df, c):
    if c < N_STEPS:
        return None
    W = ts_df[feature_cols].iloc[c - N_STEPS + 1 : c + 1].values.astype("float32")
    a0 = np.concatenate([W, np.zeros((N_STEPS, 1), dtype="float32")], axis=1)
    a1 = np.concatenate([W, np.ones ((N_STEPS, 1), dtype="float32")], axis=1)
    return np.stack([a0, a1], axis=0)

def reward_fn(ts_df, c, a, coef):
    if c < N_STEPS:
        return [0.0, 0.0]
    pen = coef * float(penalty_array[c])
    lbl = ts_df["label"].iat[c]
    base = [TN, FP] if lbl == 0 else [FN, TP]
    return [base[0] + pen, base[1] + pen]

# ─── Full train + validate ─────────────────────────────────────────────
def train_and_validate(AL_budget):
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    K.set_session(sess)

    # Build Q‐networks
    ql = QNet("ql"); ql.build()
    qt = QNet("qt"); qt.build()
    sess.run(tf.compat.v1.global_variables_initializer())

    env = EnvWaDi(TS, make_state, lambda ts, cc, aa: [0.0, 0.0])
    from collections import namedtuple
    Transition = namedtuple("T", ["s", "r", "ns", "d"])
    memory, coef = [], 20.0

    # Optional warm‐up IsolationForest on known‐normal segments
    W0 = []
    env.reset()
    for ii in range(N_STEPS, len(env.timeseries)):
        if env.timeseries["label"].iat[ii] == 0:
            W0.append(env.timeseries[feature_cols].iloc[ii - N_STEPS + 1 : ii + 1].values.flatten())
    if W0:
        from sklearn.ensemble import IsolationForest
        IsolationForest(contamination=0.01).fit(np.array(W0[:MAX_VAE_SAMPLES], dtype="float32"))

    # ─── RL EPISODES ───────────────────────────────────────────────────
    for ep in range(1, EPISODES + 1):
        # 1) Label‐Propagation + Active Learning before rollout
        env.reset()
        labs = np.array(env.timeseries["label"].iloc[N_STEPS :])
        if np.any(labs != -1):
            Warr = np.array(env.states_list)  # each element = flattened “action=0” window
            flat = Warr.reshape(Warr.shape[0], -1)
            lp   = LabelSpreading(kernel="knn", n_neighbors=10).fit(flat, labs)
            uncert = 1.0 - lp.label_distributions_.max(axis=1)
            idx    = np.argsort(-uncert)
            for i in idx[:AL_budget]:
                env.timeseries["label"].iat[i + N_STEPS] = env.timeseries["anomaly"].iat[i + N_STEPS]
            for i in idx[AL_budget : AL_budget + NUM_LP]:
                env.timeseries["label"].iat[i + N_STEPS] = lp.transduction_[i]

        # 2) Rollout with ε‐greedy, storing in memory
        env.rewardfnc = lambda ts, cc, aa: reward_fn(ts, cc, aa, coef)
        s, done = env.reset(), False
        eps     = max(0.1, 1.0 - float(ep) / EPISODES)
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

        # 3) Replay updates (5 gradient steps)
        for _ in range(5):
            batch = random.sample(memory, min(BATCH_SIZE, len(memory)))
            S_batch, R_batch, NS_batch, _ = map(np.array, zip(*batch))

            qn     = qt.predict(NS_batch, sess)
            qn_max = np.max(qn, axis=1, keepdims=True)
            tgt    = R_batch + DISCOUNT * np.repeat(qn_max, 2, axis=1)
            ql.update(S_batch, tgt.astype("float32"), sess)

        copy_params(sess, ql, qt)

        # 4) Update the dynamic‐reward coefficient
        coef = max(min(coef + 0.001 * ep_reward, 10.0), 0.1)
        print(f"[train AL={AL_budget}] ep{ep:02d}/{EPISODES}  coef={coef:.2f}  reward={ep_reward:.2f}")

    # ─── Equal‐slice validation ────────────────────────────────────────
    base_ts = TS.copy()
    seg     = len(base_ts) // K_SLICES
    outdir  = f"validation_AL{AL_budget}"
    os.makedirs(outdir, exist_ok=True)
    f1s, aus = [], []

    for i in range(K_SLICES):
        chunk_df = base_ts.iloc[i * seg : (i + 1) * seg].reset_index(drop=True)
        envv = EnvWaDi(chunk_df, make_state, lambda ts, cc, aa: [0.0, 0.0])
        s, done = envv.reset(), False
        P, G, V = [], [], []

        while not done:
            a = np.argmax(ql.predict([s], sess)[0])
            P.append(a)
            G.append(envv.timeseries["anomaly"].iat[envv.t])
            V.append(s[-1][0])
            nxt, _, done, _ = envv.step(a)
            s = nxt[a] if nxt.ndim > 2 else nxt

        p, r, f1, _ = precision_recall_fscore_support(G, P, average="binary", zero_division=0)
        au          = average_precision_score(G, P)
        f1s.append(f1); aus.append(au)

        prefix = f"{outdir}/slice_{i}"
        np.savetxt(prefix + ".txt", [p, r, f1, au], fmt="%.6f")
        fig, ax = plt.subplots(4, sharex=True, figsize=(8, 6))
        ax[0].plot(V);      ax[0].set_title("Time Series")
        ax[1].plot(P, "g"); ax[1].set_title("Predictions")
        ax[2].plot(G, "r"); ax[2].set_title("Ground Truth")
        ax[3].plot([au] * len(V), "m"); ax[3].set_title("AUPR")
        plt.tight_layout(); plt.savefig(prefix + ".png"); plt.close(fig)

        print(f"[val AL={AL_budget}] slice {i+1}/{K_SLICES}   F1={f1:.3f}   AU={au:.3f}")

    print(f"[val AL={AL_budget}] mean F1={np.mean(f1s):.3f}   mean AUPR={np.mean(aus):.3f}\n")


# ─── DRIVER LOOP ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # (You can lower these if you want shorter run time)
    EPISODES   = 10
    BATCH_SIZE = 64
    DISCOUNT   = 0.5
    TN, TP, FP, FN = 1, 10, -1, -10
    NUM_LP     = 200
    K_SLICES   = 3

    for AL in [1000, 5000, 10000]:
        print(f"\n=== ACTIVE LEARNING BUDGET: {AL} ===")
        train_and_validate(AL)
