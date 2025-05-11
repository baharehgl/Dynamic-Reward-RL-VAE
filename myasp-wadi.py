#!/usr/bin/env python3
import os, random, numpy as np, pandas as pd, tensorflow as tf
from collections import namedtuple
tf.compat.v1.disable_eager_execution()

from tensorflow.keras import layers, models, losses
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from env_wadi import EnvTimeSeriesWaDi

# ─── replace the backend import with TF-1.x compat backend ─────────────
K = tf.compat.v1.keras.backend

# ─── rest of your imports & hyperparams ─────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
WA_DI= os.path.join(BASE, "WaDi")
SENSOR_C = os.path.join(WA_DI, "WADI_14days_new.csv")
LABEL_C  = os.path.join(WA_DI, "WADI_attackdataLABLE.csv")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

EPISODES, N_STEPS, BATCH_SIZE = 30, 25, 128
DISCOUNT = 0.5
TN,TP,FP,FN = 1,10,-1,-10
NUM_LP, K_SLICES = 200, 5
MAX_VAE_SAMPLES, VAE_EPOCHS, VAE_BATCH = 200, 2, 32

# ─── 1) load numeric sensor cols ────────────────────────────────────────
df = pd.read_csv(SENSOR_C, decimal='.')
df.columns = df.columns.str.strip()
df = df.apply(pd.to_numeric, errors="coerce")\
       .dropna(axis=1, how="all").dropna(axis=0, how="all")\
       .reset_index(drop=True)
if "Row" in df.columns: df = df.drop(columns=["Row"])
feature_cols = df.columns.tolist(); n_features = len(feature_cols)
N_INPUT = n_features + 1

# ─── 2) VAE pretrain on normal windows ──────────────────────────────────
print("\n-- VAE pretraining --")
lbl = pd.read_csv(LABEL_C, header=1)["Attack LABLE (1:No Attack, -1:Attack)"].astype(int)
normal_idx = np.where(lbl.values[N_STEPS:]==1)[0] + N_STEPS
sample_idx = np.random.choice(normal_idx, min(MAX_VAE_SAMPLES, len(normal_idx)), replace=False)
windows = [df.iloc[i-N_STEPS+1:i+1][feature_cols].values.flatten() for i in sample_idx]
Xs = StandardScaler().fit_transform(np.array(windows, "float32"))
print("windows shape:", Xs.shape)

def build_vae(dim, hid=64, lat=10):
    x_in = layers.Input((dim,))
    h    = layers.Dense(hid, activation='relu')(x_in)
    mu   = layers.Dense(lat)(h)
    lv   = tf.clip_by_value(layers.Dense(lat)(h), -10, 10)
    z    = layers.Lambda(lambda t: t[0]+tf.exp(0.5*t[1])*tf.random.normal(tf.shape(t[0])))((mu,lv))
    enc  = models.Model(x_in, [mu, lv, z], name="encoder")
    z_in = layers.Input((lat,)); d_h = layers.Dense(hid,"relu")(z_in)
    x_out= layers.Dense(dim,"sigmoid")(d_h)
    dec  = models.Model(z_in, x_out, name="decoder")
    recon = dec(z); vae = models.Model(x_in, recon, name="vae")
    rl = losses.mse(x_in, recon)*dim
    kl = -0.5 * tf.reduce_sum(1 + lv - tf.square(mu) - tf.exp(lv), axis=1)
    vae.add_loss(tf.reduce_mean(rl + kl)); vae.compile("adam")
    return vae, enc, dec

vae, encoder, decoder = build_vae(N_STEPS * n_features)
encoder.summary(print_fn=lambda l: print(" ",l))
decoder.summary(print_fn=lambda l: print(" ",l))
vae.fit(Xs, epochs=VAE_EPOCHS, batch_size=VAE_BATCH, verbose=1)

# ─── 3) state & reward ───────────────────────────────────────────────
def make_state(ts, c):
    if c < N_STEPS: return None
    W = ts[feature_cols].iloc[c-N_STEPS+1:c+1].values.astype("float32")
    return np.stack([np.concatenate([W, np.zeros((N_STEPS,1))],1),
                     np.concatenate([W, np.ones ((N_STEPS,1))],1)])
def reward_fn(ts, c, a, coef, vae_model):
    if c < N_STEPS: return [0,0]
    w = ts[feature_cols].iloc[c-N_STEPS+1:c+1].values.flatten()[None]
    pen = coef * np.mean((vae_model.predict(w)-w)**2)
    base = [TN,FP] if ts['label'].iat[c]==0 else [FN,TP]
    return [base[0]+pen, base[1]+pen]

# ─── 4) Q-network ─────────────────────────────────────────────────────
class QNet:
    def __init__(s, sc): s.sc=sc
    def build(s):
        with tf.compat.v1.variable_scope(s.sc):
            s.S  = tf.compat.v1.placeholder(tf.float32, [None,N_STEPS,N_INPUT])
            s.T  = tf.compat.v1.placeholder(tf.float32, [None,2])
            seq  = tf.compat.v1.unstack(s.S, N_STEPS, 1)
            out,_= tf.compat.v1.nn.static_rnn(tf.compat.v1.nn.rnn_cell.LSTMCell(128), seq, dtype=tf.float32)
            s.Q  = layers.Dense(2)(out[-1])
            s.trn= tf.compat.v1.train.AdamOptimizer(3e-4).minimize(
                       tf.reduce_mean(tf.square(s.Q - s.T)))
    def predict(s, x, ss): return ss.run(s.Q, {s.S:x})
    def update (s, x, y, ss): ss.run(s.trn, {s.S:x, s.T:y})
def copy_params(ss, src, dst):
    sv = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, src.sc)
    dv = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, dst.sc)
    for a,b in zip(sorted(sv,key=lambda v:v.name), sorted(dv,key=lambda v:v.name)):
        ss.run(b.assign(a))

# ─── 5) full train + validate in one session ─────────────────────────
def train_and_validate(AL_budget):
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    K.set_session(sess)

    # build Q-nets
    ql, qt = QNet("ql"), QNet("qt")
    ql.build(); qt.build()
    sess.run(tf.compat.v1.global_variables_initializer())

    # env
    env = EnvTimeSeriesWaDi(SENSOR_C, LABEL_C, N_STEPS)
    env.statefnc = make_state

    # replay memory
    Transition = namedtuple("T","s r ns d")
    memory, coef = [], 20.0

    # warm-up iso-forest
    env.reset()
    W=[]
    for i in range(N_STEPS, len(env.timeseries)):
        if env.timeseries['label'].iat[i]==0:
            W.append(env.timeseries[feature_cols].iloc[i-N_STEPS+1:i+1].values.flatten())
    if W:
        IsolationForest(contamination=0.01).fit(np.array(W[:MAX_VAE_SAMPLES],"float32"))

    # episodes
    for ep in range(1, EPISODES+1):
        # LP + AL
        env.reset(); labs=np.array(env.timeseries['label'].iloc[N_STEPS:])
        if np.any(labs!=-1):
            W_arr=np.array([w for w in env.states_list if w is not None])
            flat=W_arr.reshape(W_arr.shape[0], -1)
            lp = LabelSpreading(kernel='knn', n_neighbors=10).fit(flat, labs)
            uncert=1-lp.label_distributions_.max(axis=1)
            idx=np.argsort(-uncert)
            for i in idx[:AL_budget]:
                env.timeseries['label'].iat[i+N_STEPS]=env.timeseries['anomaly'].iat[i+N_STEPS]
            for i in idx[AL_budget:AL_budget+NUM_LP]:
                env.timeseries['label'].iat[i+N_STEPS]=lp.transduction_[i]

        # rollout
        env.rewardfnc = lambda ts,c,a,cf=coef: reward_fn(ts,c,a,cf,vae)
        s,done = env.reset(), False
        eps = max(0.1, 1 - ep/EPISODES)
        while not done:
            if random.random() < eps:
                a = random.choice([0,1])
            else:
                a = np.argmax(ql.predict([s], sess)[0])
            raw, r, done, _ = env.step(a)
            ns = raw[a] if raw.ndim>2 else raw
            memory.append(Transition(s, r, ns, done))
            s = ns

        # replay
        for _ in range(5):
            batch = random.sample(memory, min(BATCH_SIZE, len(memory)))
            S, R, NS, _ = map(np.array, zip(*batch))
            qn = qt.predict(NS, sess)
            tgt = R + DISCOUNT * np.repeat(qn.max(1, keepdims=True), 2, 1)
            ql.update(S, tgt.astype("float32"), sess)
        copy_params(sess, ql, qt)

        # update coef
        coef = max(min(coef + 0.001*np.sum(R[:,0]), 100), 0.1)
        print(f"[train AL={AL_budget}] ep{ep:02}/{EPISODES} coef={coef:.2f}")

    # validation on equal slices
    full_ts = EnvTimeSeriesWaDi(SENSOR_C, LABEL_C, N_STEPS).timeseries_repo[0]
    seg = len(full_ts)//K_SLICES
    os.makedirs(f"validation_AL{AL_budget}", exist_ok=True)
    f1s, aus = [], []

    for i in range(K_SLICES):
        env_val = EnvTimeSeriesWaDi(SENSOR_C, LABEL_C, N_STEPS)
        env_val.statefnc = make_state
        env_val.timeseries_repo[0] = full_ts.iloc[i*seg:(i+1)*seg]
        s,done=[],False; s,done=env_val.reset(),False
        preds, gts, vals = [],[],[]
        while not done:
            a = np.argmax(ql.predict([s], sess)[0])
            preds.append(a)
            gts.append(env_val.timeseries['anomaly'].iat[env_val.timeseries_curser])
            vals.append(s[-1][0])
            nxt,_,done,_ = env_val.step(a)
            s = nxt[a] if nxt.ndim>2 else nxt

        p,r,f1,_=precision_recall_fscore_support(gts,preds,average='binary',zero_division=0)
        au=average_precision_score(gts,preds)
        f1s.append(f1); aus.append(au)

        prefix = f"validation_AL{AL_budget}/slice_{i}"
        np.savetxt(prefix+".txt",[p,r,f1,au],fmt="%.6f")
        fig,ax=plt.subplots(4,sharex=True,figsize=(8,6))
        ax[0].plot(vals); ax[0].set_title("TS")
        ax[1].plot(preds,'g');ax[1].set_title("Pred")
        ax[2].plot(gts,'r');ax[2].set_title("GT")
        ax[3].plot([au]*len(vals),'m');ax[3].set_title("AUPR")
        plt.tight_layout(); plt.savefig(prefix+".png"); plt.close(fig)

        print(f"[val AL={AL_budget}] slice {i+1}/{K_SLICES} F1={f1:.3f} AU={au:.3f}")

    print(f"[val AL={AL_budget}] mean F1={np.mean(f1s):.3f}  mean AUPR={np.mean(aus):.3f}")

# ─── driver ────────────────────────────────────────────────
if __name__ == "__main__":
    for AL in [1000, 5000, 10000]:
        print(f"\n=== RUN AL={AL} ===")
        train_and_validate(AL)
