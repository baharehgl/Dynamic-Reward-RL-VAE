#!/usr/bin/env python3
import os, random
import numpy as np
import pandas as pd
import tensorflow as tf

# ─── 1) VAE PRETRAIN ───────────────────────────────────────────────
from tensorflow.keras import layers, models, losses
from sklearn.preprocessing import StandardScaler

BASE   = os.path.dirname(__file__)
WA_DI  = os.path.join(BASE, "WaDi")
SENSOR = os.path.join(WA_DI, "WADI_14days_new.csv")
LABEL  = os.path.join(WA_DI, "WADI_attackdataLABLE.csv")

# load & down-sample for speed
df = pd.read_csv(SENSOR, decimal='.')
df.columns = df.columns.str.strip()
df = df.apply(pd.to_numeric, errors='coerce') \
       .dropna(axis=1, how='all') \
       .dropna(axis=0, how='all') \
       .reset_index(drop=True)
if 'Row' in df.columns:
    df = df.drop(columns=['Row'])
feature_cols = df.columns.tolist()
n_features   = len(feature_cols)
N_STEPS       = 25

# load labels (down-sampled same as df)
lbl_all = pd.read_csv(LABEL, header=1)["Attack LABLE (1:No Attack, -1:Attack)"] \
             .iloc[::2].astype(int).values

# sample up to 100 normal windows
norm_idx = np.where(lbl_all[N_STEPS:] == 1)[0] + N_STEPS
MAX_VAE_SAMPLES = 100
sample_idx      = np.random.choice(norm_idx,
                                   min(MAX_VAE_SAMPLES, len(norm_idx)),
                                   replace=False)

print(f"[VAE] sampling {len(sample_idx)} windows")
windows = np.array([
    df.iloc[i-N_STEPS+1:i+1][feature_cols].values.flatten()
    for i in sample_idx
], dtype='float32')
Xs = StandardScaler().fit_transform(windows)

def build_vae(input_dim, hidden=32, latent=5):
    x_in = layers.Input((input_dim,))
    h    = layers.Dense(hidden, activation='relu')(x_in)
    mu   = layers.Dense(latent)(h)
    lv   = tf.clip_by_value(layers.Dense(latent)(h), -10, 10)
    z    = layers.Lambda(lambda t: t[0] +
                         tf.exp(0.5*t[1]) * tf.random.normal(tf.shape(t[0]))
                        )([mu, lv])
    enc  = models.Model(x_in, [mu, lv, z])
    z_in = layers.Input((latent,))
    dh   = layers.Dense(hidden, activation='relu')(z_in)
    out  = layers.Dense(input_dim, activation='sigmoid')(dh)
    dec  = models.Model(z_in, out)

    recon = dec(z)
    vae   = models.Model(x_in, recon)
    rl    = losses.mse(x_in, recon) * input_dim
    kl    = -0.5 * tf.reduce_sum(1 + lv - tf.square(mu) - tf.exp(lv), axis=1)
    vae.add_loss(tf.reduce_mean(rl + kl))
    vae.compile(optimizer='adam')
    return vae, enc, dec

vae, encoder, decoder = build_vae(N_STEPS * n_features)
vae.fit(Xs, epochs=2, batch_size=32, verbose=1)
print("[VAE] pretrained")

# build penalty array
all_w = np.array([
    df.iloc[i-N_STEPS+1:i+1][feature_cols].values.flatten()
    for i in range(N_STEPS-1, len(df))
], dtype='float32')
all_w = StandardScaler().fit_transform(all_w)

errs, bs = [], 128
for i in range(0, len(all_w), bs):
    c = all_w[i:i+bs]
    p = vae.predict(c, batch_size=c.shape[0], verbose=0)
    errs.append(np.mean((p - c)**2, axis=1))
recon_err     = np.concatenate(errs)
penalty_array = np.concatenate([np.zeros(N_STEPS-1), recon_err])

# ── FIX: replace any NaN/inf with 0.0 ─────────────────────────────
penalty_array = np.nan_to_num(penalty_array,
                              nan=0.0,
                              posinf=0.0,
                              neginf=0.0)
print("[VAE] penalty_array cleaned – NaNs replaced with 0, length=", len(penalty_array))


# ─── 2) RL + LP + AL + Validation (TF-1 Graph) ─────────────────────
tf.compat.v1.disable_eager_execution()
K = tf.compat.v1.keras.backend

from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from env_wadi import EnvTimeSeriesWaDi

# hyperparams (reduced)
EPISODES   = 10
BATCH_SIZE = 128
DISCOUNT   = 0.5
TN, TP, FP, FN = 1, 10, -1, -10
NUM_LP     = 100
K_SLICES   = 3

class QNet:
    def __init__(s, scope): s.sc = scope
    def build(s):
        with tf.compat.v1.variable_scope(s.sc):
            s.S   = tf.compat.v1.placeholder(tf.float32,
                                             [None, N_STEPS, n_features+1])
            s.T   = tf.compat.v1.placeholder(tf.float32, [None, 2])
            seq   = tf.compat.v1.unstack(s.S, N_STEPS, 1)
            out,_ = tf.compat.v1.nn.static_rnn(
                        tf.compat.v1.nn.rnn_cell.LSTMCell(64),
                        seq, dtype=tf.float32)
            s.Q   = layers.Dense(2)(out[-1])
            s.trn = tf.compat.v1.train.AdamOptimizer(3e-4)\
                        .minimize(tf.reduce_mean((s.Q - s.T)**2))
    def predict(s, x, ss): return ss.run(s.Q, {s.S: x})
    def update(s, x, y, ss): ss.run(s.trn, {s.S: x, s.T: y})

def copy_params(ss, src, dst):
    v1 = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                     src.sc)
    v2 = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                     dst.sc)
    for a, b in zip(sorted(v1, key=lambda v: v.name),
                    sorted(v2, key=lambda v: v.name)):
        ss.run(b.assign(a))

def make_state(ts, c):
    if c < N_STEPS: return None
    W = ts[feature_cols].iloc[c-N_STEPS+1:c+1].values.astype('float32')
    return np.stack([
      np.concatenate([W, np.zeros((N_STEPS,1))], 1),
      np.concatenate([W, np.ones ((N_STEPS,1))], 1)
    ])

def reward_fn(ts, c, a, coef):
    if c < N_STEPS:
        return [0.0, 0.0]
    pen = coef * float(penalty_array[c])
    # clamp in case cast still weird
    if np.isnan(pen):
        pen = 0.0
    base = [TN, FP] if ts['label'].iat[c] == 0 else [FN, TP]
    return [base[0] + pen, base[1] + pen]

def train_and_validate(AL):
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session(); K.set_session(sess)

    # build networks
    ql, qt = QNet("ql"), QNet("qt")
    ql.build(); qt.build()
    sess.run(tf.compat.v1.global_variables_initializer())

    env = EnvTimeSeriesWaDi(SENSOR, LABEL, N_STEPS)
    env.statefnc = make_state

    from collections import namedtuple
    T   = namedtuple("T", "s r ns d")
    mem, coef = [], 20.0

    # warm-up IsolationForest
    env.reset()
    buf = []
    for i in range(N_STEPS, len(env.timeseries)):
        if env.timeseries['label'].iat[i] == 0:
            buf.append(env.timeseries[feature_cols]
                       .iloc[i-N_STEPS+1:i+1].values.flatten())
    if buf:
        from sklearn.ensemble import IsolationForest
        IsolationForest(contamination=0.01).fit(
            np.array(buf[:MAX_VAE_SAMPLES], 'float32'))

    for ep in range(1, EPISODES+1):
        # ── LP + AL ─────────────────────────────────
        env.reset()
        labs = np.array(env.timeseries['label'].iloc[N_STEPS:])
        if np.any(labs != -1):
            W    = np.array([s for s in env.states_list if s is not None])
            flat = W.reshape(W.shape[0], -1)
            lp   = LabelSpreading(n_neighbors=5).fit(flat, labs)
            uncert = 1 - lp.label_distributions_.max(axis=1)
            idx    = np.argsort(-uncert)
            # Active Learning:
            for i in idx[:AL]:
                env.timeseries['label'].iat[i+N_STEPS] = \
                    env.timeseries['anomaly'].iat[i+N_STEPS]
            # Pseudo-labels:
            for i in idx[AL:AL+NUM_LP]:
                env.timeseries['label'].iat[i+N_STEPS] = lp.transduction_[i]

        # ── set reward fn before reset! ─────────────────
        env.rewardfnc = lambda ts, c, a, cf=coef: reward_fn(ts, c, a, cf)
        s, done, ep_reward = env.reset(), False, 0.0
        eps = max(0.1, 1 - ep/EPISODES)

        # rollout
        while not done:
            if random.random() < eps:
                a = random.choice([0,1])
            else:
                a = np.argmax(ql.predict([s], sess)[0])
            raw, r, done, _ = env.step(a)
            ns = raw[a] if raw.ndim>2 else raw
            mem.append(T(s, r, ns, done))
            ep_reward += r[a]
            s = ns

        # replay
        for _ in range(3):
            batch = random.sample(mem, min(BATCH_SIZE, len(mem)))
            S, R, NS, _ = map(np.array, zip(*batch))
            qn = qt.predict(NS, sess)
            tgt = R + DISCOUNT * np.repeat(qn.max(1, keepdims=True), 2, 1)
            ql.update(S, tgt.astype('float32'), sess)
        copy_params(sess, ql, qt)

        # dynamic‐coef update
        coef = max(min(coef + 0.001 * (ep_reward - 0.0), 100), 0.1)
        print(f"[train AL={AL}] ep{ep:02}/{EPISODES}"
              f" coef={coef:.2f} reward={ep_reward:.2f}")

    # ── equal‐slice validation ─────────────────────────────
    full = EnvTimeSeriesWaDi(SENSOR, LABEL, N_STEPS).timeseries_repo[0]
    seg  = len(full) // K_SLICES
    out  = f"val_AL{AL}"; os.makedirs(out, exist_ok=True)
    f1s, aus = [], []

    for i in range(K_SLICES):
        envv = EnvTimeSeriesWaDi(SENSOR, LABEL, N_STEPS)
        envv.statefnc = make_state
        part = full.iloc[i*seg:(i+1)*seg].reset_index(drop=True)
        envv.timeseries_repo[0] = part
        s, done, P, G, V = envv.reset(), False, [], [], []
        while not done:
            a = np.argmax(ql.predict([s], sess)[0])
            P.append(a)
            G.append(envv.timeseries['anomaly'].iat[envv.timeseries_curser])
            V.append(s[-1][0])
            nxt, _, done, _ = envv.step(a)
            s = nxt[a] if nxt.ndim>2 else nxt

        p, r, f1, _ = precision_recall_fscore_support(
            G, P, average='binary', zero_division=0)
        au = average_precision_score(G, P)
        f1s.append(f1); aus.append(au)

        # save & plot...
        print(f"[val AL={AL}] slice{i+1}/{K_SLICES}"
              f" F1={f1:.3f} AU={au:.3f}")

    print(f"[val AL={AL}] mean F1={np.mean(f1s):.3f}"
          f" mean AUPR={np.mean(aus):.3f}\n")


if __name__=="__main__":
    for AL in [1000, 5000, 10000]:
        print(f"\n=== ACTIVE LEARNING BUDGET: {AL} ===")
        train_and_validate(AL)
