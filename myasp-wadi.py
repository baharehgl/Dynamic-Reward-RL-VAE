
import os, random
import numpy as np
import pandas as pd
import tensorflow as tf

# ───────────────────────────────────────────────────────────────────────────────
# A) PRETRAIN VAE (EAGER MODE)
# ───────────────────────────────────────────────────────────────────────────────
from tensorflow.keras import layers, models, losses
from sklearn.preprocessing import StandardScaler

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(__file__)
WA_DI  = os.path.join(BASE, "WaDi")
SENSOR = os.path.join(WA_DI, "WADI_14days_new.csv")
LABEL  = os.path.join(WA_DI, "WADI_attackdataLABLE.csv")

# ─── 1) Load + clean sensor data ───────────────────────────────────────────────
df = pd.read_csv(SENSOR, decimal='.')
df.columns = df.columns.str.strip()

# Convert all columns to numeric, drop any all-NaN row/column
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna(axis=1, how='all').dropna(axis=0, how='all').reset_index(drop=True)

# Drop “Row” column if present
if 'Row' in df.columns:
    df = df.drop(columns=['Row'])

print(f"[DATA] Kept {len(df.columns)} numeric sensor columns; {len(df)} time‐steps total.")

# ─── 2) Load + align anomaly labels ─────────────────────────────────────────────
lbl_df       = pd.read_csv(LABEL, header=1, low_memory=False)
raw_lbl      = lbl_df["Attack LABLE (1:No Attack, -1:Attack)"].astype(int).values
#  1 → “No Attack” → label 0;  -1 → “Attack” → label 1
anomaly_full = np.where(raw_lbl == 1, 0, 1)

# Truncate both to the same length
min_len = min(len(df), len(anomaly_full))
df = df.iloc[:min_len].reset_index(drop=True)
anomaly = anomaly_full[:min_len]
print(f"[DATA] Truncated series + labels to length = {min_len}")

# Build a DataFrame TS containing all sensors + “anomaly” + “label”
TS = df.copy()
TS["anomaly"] = anomaly
TS["label"]   = -1   # “-1” means “unlabeled”

print(f"[DATA] TS.shape = {TS.shape} (sensors + anomaly + label)")

feature_cols_full = df.columns.tolist()
n_features_full   = len(feature_cols_full)

# ─── 3) Sample up to MAX_VAE_SAMPLES “normal” windows of length N_STEPS ───────
N_STEPS         = 25
MAX_VAE_SAMPLES = 200

# Find indices c where anomaly[c] == 0, but only c >= N_STEPS
normal_indices = np.where(TS["anomaly"].values[N_STEPS:] == 0)[0] + N_STEPS
sample_idx     = np.random.choice(
    normal_indices,
    size = min(MAX_VAE_SAMPLES, len(normal_indices)),
    replace=False
)
print(f"[VAE] Sampling {len(sample_idx)} normal windows of length {N_STEPS}")

# Build a (num_samples, 25, n_features_full) array
temp = np.array([
    TS.iloc[i - N_STEPS + 1 : i + 1][feature_cols_full].values
    for i in sample_idx
], dtype="float32")   # → shape = (num_samples, 25, n_features_full)

# Compute variance per‐sensor across all sampled windows (axes 0 and 1)
sensor_var = temp.var(axis=(0,1))   # shape = (n_features_full,)

# Keep only sensors whose variance > 0
keep_sensors = sensor_var > 0
feature_cols = [
    feature_cols_full[i]
    for i in range(n_features_full)
    if keep_sensors[i]
]
n_features = len(feature_cols)
print(f"[VAE] Dropped {n_features_full - n_features} zero‐var sensors; now n_features = {n_features}")

# Flatten each sampled window (only surviving sensors) into length (25 * n_features)
windows = np.array([
    TS.iloc[i - N_STEPS + 1 : i + 1][feature_cols].values.flatten()
    for i in sample_idx
], dtype="float32")  # → shape = (num_samples, 25*n_features)

INPUT_DIM = windows.shape[1]
print(f"[VAE] Flattened‐window shape = {INPUT_DIM}")

# Standardize the sampled windows
scaler = StandardScaler().fit(windows)
Xs     = scaler.transform(windows)
print(f"[VAE] Xs.shape = {Xs.shape}")

# ─── 4) Build & train the VAE ─────────────────────────────────────────────────
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
    # Loss = MSE_reconstruction * input_dim + KL
    rl = losses.mse(x_in, recon) * input_dim
    kl = -0.5 * tf.reduce_sum(1 + z_lv - tf.square(z_mu) - tf.exp(z_lv), axis=1)
    vae.add_loss(tf.reduce_mean(rl + kl))
    vae.compile(optimizer="adam")
    return vae, encoder, decoder

vae, encoder, decoder = build_vae(INPUT_DIM, hidden=64, latent=10)
vae.summary()

vae.fit(Xs, epochs=2, batch_size=32, verbose=1)
print("[VAE] Pretraining complete")

# ─── 5) Compute per‐step reconstruction error over the entire TS ────────────
print("[VAE] Computing per‐step reconstruction error...")
all_windows = np.array([
    TS.iloc[i - N_STEPS + 1 : i + 1][feature_cols].values.flatten()
    for i in range(N_STEPS - 1, len(TS))
], dtype="float32")  # → shape = (len(TS)-(N_STEPS-1), 25*n_features)

all_windows_scaled = scaler.transform(all_windows)
batch_size = 64
errs = []
for start in range(0, len(all_windows_scaled), batch_size):
    chunk = all_windows_scaled[start : start + batch_size]
    pred  = vae.predict(chunk, batch_size=chunk.shape[0], verbose=0)
    errs.append(np.mean((pred - chunk) ** 2, axis=1))

recon_err     = np.concatenate(errs, axis=0)
penalty_array = np.concatenate([np.zeros(N_STEPS - 1), recon_err])
print(f"[VAE] penalty_array ready (length = {len(penalty_array)})")
print(f"[VAE] stats: min={np.nanmin(penalty_array):.3e}, max={np.nanmax(penalty_array):.3e}, mean={np.nanmean(penalty_array):.3e}")

# ───────────────────────────────────────────────────────────────────────────────
# B) RL TRAINING & VALIDATION (TF‐1 GRAPH MODE)
# ───────────────────────────────────────────────────────────────────────────────
tf.compat.v1.disable_eager_execution()
K = tf.compat.v1.keras.backend

from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── 6) Minimal EnvWaDi ───────────────────────────────────────────────────────
class EnvWaDi:
    """
    An environment for WADI that keeps a current single “state” of shape (25, n_features+1).
    - reset() returns state_0: shape = (25, n_features+1)
    - step(a) returns (next_state, [r0, r1], done, {}), where next_state is the
      two‐branch array of shape (2, 25, n_features+1).  For RL‐updates, the caller picks
      raw[a] to obtain the single (25, n_features+1) next state.
    """
    def __init__(self, timeseries_df, statefnc, rewardfnc):
        self.timeseries     = timeseries_df.copy().reset_index(drop=True)
        self.N              = len(self.timeseries)
        self.statefnc       = statefnc
        self.rewardfnc      = rewardfnc
        self.t0             = N_STEPS
        self.action_space_n = 2

    def reset(self):
        self.t = self.t0
        # Precompute “flattened action=0” windows for LabelSpreading / AL
        self.states_list = []
        for c in range(self.t0, self.N):
            two_states = self.statefnc(self.timeseries, c)
            if two_states is not None:
                # store only the “action=0” branch as a 1‐D flattened vector
                flat0 = two_states[0].flatten()
                self.states_list.append(flat0)
        # Return the single “current” state (25, n_features+1) → two_states[0]
        return self.statefnc(self.timeseries, self.t)[0]

    def step(self, action):
        """
        Given an action ∈ {0,1} at time self.t:
          - Compute r = [r0, r1] via rewardfnc(self.timeseries, self.t, action).
          - Increment self.t ← self.t+1.
          - done = (self.t >= self.N).
          - If not done: return two_states = statefnc(self.timeseries, self.t), else
            return two_states = statefnc(self.timeseries, self.N−1) just to keep the shape consistent.
        """
        r = self.rewardfnc(self.timeseries, self.t, action)
        self.t += 1
        done = int(self.t >= self.N)
        if not done:
            two_states = self.statefnc(self.timeseries, self.t)
        else:
            two_states = self.statefnc(self.timeseries, self.N - 1)
        return two_states, r, done, {}

# ─── 7) Q‐Network (TF‐1 style) ────────────────────────────────────────────────
class QNet:
    def __init__(self, scope): self.sc = scope

    def build(self):
        with tf.compat.v1.variable_scope(self.sc):
            # placeholder “S” has shape [None, 25, n_features+1]
            self.S = tf.compat.v1.placeholder(
                tf.float32, [None, N_STEPS, n_features + 1], name="S"
            )
            # placeholder “T” has shape [None, 2]
            self.T = tf.compat.v1.placeholder(
                tf.float32, [None, 2], name="T"
            )
            # build an LSTMCell(64) over 25 time‐steps
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(64)
            seq, _ = tf.compat.v1.nn.static_rnn(
                cell, tf.compat.v1.unstack(self.S, N_STEPS, axis=1), dtype=tf.float32
            )
            # “seq[-1]” is the final LSTM output at t=25
            self.Q    = layers.Dense(2)(seq[-1])
            self.loss = tf.reduce_mean(tf.square(self.Q - self.T))
            self.train = tf.compat.v1.train.AdamOptimizer(3e-4).minimize(self.loss)

    def predict(self, x, sess):
        return sess.run(self.Q, {self.S: x})

    def update(self, x, y, sess):
        sess.run(self.train, {self.S: x, self.T: y})

def copy_params(sess, src: QNet, dst: QNet):
    sv = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, src.sc
    )
    dv = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, dst.sc
    )
    for s_var, d_var in zip(
        sorted(sv, key=lambda v: v.name),
        sorted(dv, key=lambda v: v.name)
    ):
        sess.run(d_var.assign(s_var))

# ─── 8) State & Reward functions using penalty_array ────────────────────────
def make_state(ts_df, c):
    if c < N_STEPS:
        return None
    W  = ts_df[feature_cols].iloc[c - N_STEPS + 1 : c + 1].values.astype("float32")
    a0 = np.concatenate([W, np.zeros((N_STEPS, 1), dtype="float32")], axis=1)
    a1 = np.concatenate([W, np.ones ((N_STEPS, 1), dtype="float32")], axis=1)
    # return a (2, 25, n_features+1) array
    return np.stack([a0, a1], axis=0)

def reward_fn(ts_df, c, a, coef):
    if c < N_STEPS:
        return [0.0, 0.0]
    pen = coef * float(penalty_array[c])
    lbl = ts_df["label"].iat[c]
    base = [TN, FP] if lbl == 0 else [FN, TP]
    return [base[0] + pen, base[1] + pen]

# ─── 9) Full train + equal-slice validate ──────────────────────────────────
def train_and_validate(AL_budget):
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    K.set_session(sess)

    # Build qlearn + qtarget and initialize
    ql = QNet("ql"); ql.build()
    qt = QNet("qt"); qt.build()
    sess.run(tf.compat.v1.global_variables_initializer())

    env = EnvWaDi(TS, make_state, lambda ts, cc, aa: [0.0, 0.0])
    from collections import namedtuple
    Transition = namedtuple("T", ["s", "r", "ns", "d"])
    memory, coef = [], 20.0

    # 9A) Optional IsolationForest warm-up on known-normal windows
    W0 = []
    env.reset()
    for ii in range(N_STEPS, len(env.timeseries)):
        if env.timeseries["label"].iat[ii] == 0:
            W0.append(
                env.timeseries[feature_cols]
                .iloc[ii - N_STEPS + 1 : ii + 1]
                .values.flatten()
            )
    if W0:
        from sklearn.ensemble import IsolationForest
        IsolationForest(contamination=0.01).fit(
            np.array(W0[:MAX_VAE_SAMPLES], dtype="float32")
        )

    # 9B) RL EPISODES
    for ep in range(1, EPISODES + 1):
        # Step (1): LabelSpreading + Active Learning
        env.reset()
        labs = np.array(env.timeseries["label"].iloc[N_STEPS :])
        if np.any(labs != -1):
            Warr = np.array(env.states_list)  # shape=(#windows, 25*(n_features+1))
            flat = Warr.reshape(Warr.shape[0], -1)
            lp   = LabelSpreading(kernel="knn", n_neighbors=10).fit(flat, labs)
            uncert = 1.0 - lp.label_distributions_.max(axis=1)
            idx    = np.argsort(-uncert)
            # Top AL_budget get true anomaly label
            for i in idx[:AL_budget]:
                env.timeseries["label"].iat[i + N_STEPS] = env.timeseries["anomaly"].iat[i + N_STEPS]
            # Next NUM_LP get LP’s pseudo-label
            for i in idx[AL_budget : AL_budget + NUM_LP]:
                env.timeseries["label"].iat[i + N_STEPS] = lp.transduction_[i]

        # Step (2): Rollout with ε‐greedy
        env.rewardfnc = lambda ts, cc, aa: reward_fn(ts, cc, aa, coef)
        s, done = env.reset(), False  # s has shape (25, n_features+1)
        eps     = max(0.1, 1.0 - float(ep) / EPISODES)
        ep_reward = 0.0

        while not done:
            # ε‐greedy over “two‐branch Q‐values”
            if random.random() < eps:
                a = random.choice([0, 1])
            else:
                # Evaluate Q‐value of taking action=0 vs action=1
                #  • s0 = s if we pretend action=0,  shape (25,nf+1)
                #  • s1 = s if we pretend action=1
                two_states = make_state(env.timeseries, env.t)
                s0 = two_states[0]   # shape (25, nf+1)
                s1 = two_states[1]   # shape (25, nf+1)
                # feed into Q‐net:
                q0 = ql.predict([s0], sess)[0]  # shape (2,) = [Q(s0,0), Q(s0,1)]
                q1 = ql.predict([s1], sess)[0]  # shape (2,) = [Q(s1,0), Q(s1,1)]
                # We want “Q(s, a=0)” = q0[0], “Q(s, a=1)” = q1[1]
                qa0 = q0[0]
                qa1 = q1[1]
                if qa1 > qa0:
                    a = 1
                else:
                    a = 0

            two_states, r, done, _ = env.step(a)
            ns = two_states[a] if not done else two_states[0]
            # ns has shape (25, nf+1)
            memory.append(Transition(s, r, ns, done))
            ep_reward += float(r[a])
            s = ns

        # Step (3): Replay updates (5 minibatches per episode)
        for _ in range(5):
            batch = random.sample(memory, min(BATCH_SIZE, len(memory)))
            S_batch, R_batch, NS_batch, _ = map(np.array, zip(*batch))
            # NS_batch has shape (B, 25, nf+1) but actually each NS was a single branch → we stacked them
            qn     = qt.predict(NS_batch, sess)         # shape (B, 2)
            qn_max = np.max(qn, axis=1, keepdims=True)  # shape (B, 1)
            tgt    = R_batch + DISCOUNT * np.repeat(qn_max, 2, axis=1)
            ql.update(S_batch, tgt.astype("float32"), sess)

        copy_params(sess, ql, qt)

        # Step (4): Update dynamic reward coefficient
        coef = max(min(coef + 0.001 * ep_reward, 10.0), 0.1)
        print(f"[train AL={AL_budget}] ep{ep:02d}/{EPISODES}  coef={coef:.2f}  reward={ep_reward:.2f}")

    # ─── 10) Equal‐slice validation ───────────────────────────────────────────
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
            # Deterministic (ε=0) greedy
            two_states = make_state(envv.timeseries, envv.t)
            s0 = two_states[0]; s1 = two_states[1]
            q0 = ql.predict([s0], sess)[0]; q1 = ql.predict([s1], sess)[0]
            qa0 = q0[0]; qa1 = q1[1]
            a  = 1 if qa1 > qa0 else 0

            P.append(a)
            G.append(envv.timeseries["anomaly"].iat[envv.t])
            V.append(s[-1][0])
            nxt, _, done, _ = envv.step(a)
            s = nxt[a] if not done else nxt[0]

        p, r, f1, _ = precision_recall_fscore_support(G, P, average="binary", zero_division=0)
        au          = average_precision_score(G, P)
        f1s.append(f1); aus.append(au)

        prefix = f"{outdir}/slice_{i}"
        np.savetxt(prefix + ".txt", [p, r, f1, au], fmt="%.6f")
        fig, ax = plt.subplots(4, sharex=True, figsize=(8,6))
        ax[0].plot(V);      ax[0].set_title("Time Series")
        ax[1].plot(P, "g"); ax[1].set_title("Predictions")
        ax[2].plot(G, "r"); ax[2].set_title("Ground Truth")
        ax[3].plot([au]*len(V), "m"); ax[3].set_title("AUPR")
        plt.tight_layout(); plt.savefig(prefix + ".png"); plt.close(fig)

        print(f"[val AL={AL_budget}] slice {i+1}/{K_SLICES}   F1={f1:.3f}   AU={au:.3f}")

    print(f"[val AL={AL_budget}] mean F1={np.mean(f1s):.3f}   mean AUPR={np.mean(aus):.3f}\n")


# ───────────────────────────────────────────────────────────────────────────────
# C) Driver loop
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Hyperparameters (reduce if it’s still too slow)
    EPISODES   = 10     # RL episodes per AL‐budget
    BATCH_SIZE = 64     # RL minibatch size
    DISCOUNT   = 0.5
    TN, TP, FP, FN = 1, 10, -1, -10
    NUM_LP     = 200    # how many pseudo‐labels per episode
    K_SLICES   = 3      # equal-slice cross‐validation folds

    for AL in [1000, 5000, 10000]:
        print(f"\n=== ACTIVE LEARNING BUDGET: {AL} ===")
        train_and_validate(AL)
