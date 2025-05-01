
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Disable eager execution for TF1-style graph
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import layers, models, losses
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from collections import namedtuple

# Custom WADI environment
from env_wadi import EnvTimeSeriesWaDi

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
gpus = tf.config.list_physical_devices('GPU')
print("GPUs detected by TensorFlow:", gpus)

# ─── HYPERPARAMETERS & PATHS ──────────────────────────────────────────────────
EPISODES           = 2
N_STEPS            = 25
DISCOUNT_FACTOR    = 0.5
TN, TP, FP, FN     = 1, 10, -1, -10
ACTION_SPACE_N     = 2
VALIDATION_SPLIT   = 0.8
MAX_VAE_SAMPLES    = 10
NUM_LP             = 200    # Label-Propagation budget
NUM_AL             = 1000   # Active-Learning budget
BATCH_SIZE         = 128
VAE_EPOCHS         = 2
VAE_BATCH          = 32

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
WA_DI_DIR  = os.path.join(BASE_DIR, 'WaDi')
SENSOR_CSV = os.path.join(WA_DI_DIR, 'WADI_14days_new.csv')
LABEL_CSV  = os.path.join(WA_DI_DIR, 'WADI_attackdataLABLE.csv')

print("=== WADI RL with dynamic reward & active learning ===\n")

# ─── 1) Load sensor data and select numeric features ───────────────────────────
df_all = pd.read_csv(SENSOR_CSV)
# If there's an index or 'Row' column, drop it
if 'Row' in df_all.columns:
    df_all.drop(columns=['Row'], inplace=True)
# Select only numeric columns (skip dates, strings)
feature_cols = [c for c in df_all.columns if np.issubdtype(df_all[c].dtype, np.number)]
n_features   = len(feature_cols)
N_INPUT_DIM  = n_features + 1  # plus action bit
print(f"Detected {n_features} numeric features; N_INPUT_DIM = {N_INPUT_DIM}\n")

# ─── 2) Pretrain VAE on sampled normal windows ─────────────────────────────────
print("Pretraining VAE on sampled normal windows...")
lbl_df  = pd.read_csv(LABEL_CSV, header=1, low_memory=False)
raw_lbl = lbl_df["Attack LABLE (1:No Attack, -1:Attack)"].astype(int).values
labels  = np.where(raw_lbl == 1, 0, 1)
# Identify normal positions (exclude first N_STEPS)
normal_pos = np.where(labels[N_STEPS:] == 0)[0] + N_STEPS
# Sample up to MAX_VAE_SAMPLES
sampled    = np.random.choice(normal_pos, size=min(MAX_VAE_SAMPLES, len(normal_pos)), replace=False)
print(f"Sampling {len(sampled)} windows for VAE training...")
# Build sliding windows
windows = [df_all[feature_cols].iloc[i-N_STEPS+1:i+1].values.flatten() for i in sampled]
print(f"Built {len(windows)} windows each of length {N_STEPS * n_features}\n")
# Standardize
X  = np.array(windows, dtype='float32')
Xs = StandardScaler().fit_transform(X)
print(f"Standardized windows shape: {Xs.shape}\n")

# VAE definition
def build_vae(input_dim, latent_dim=10, hidden=64):
    inp = layers.Input(shape=(input_dim,))
    h   = layers.Dense(hidden, activation='relu')(inp)
    z_mean    = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
    z         = layers.Lambda(lambda args: args[0] + tf.exp(0.5*args[1]) * tf.random.normal(tf.shape(args[0])))([z_mean, z_log_var])
    h2  = layers.Dense(hidden, activation='relu')(z)
    out = layers.Dense(input_dim, activation='sigmoid')(h2)
    vae = models.Model(inp, out)
    recon = losses.mse(inp, out) * input_dim
    kl    = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae.add_loss(tf.reduce_mean(recon + kl))
    vae.compile(optimizer='adam', loss=None)
    return vae

vae = build_vae(N_STEPS * n_features)
print(f"Fitting VAE: epochs={VAE_EPOCHS}, batch_size={VAE_BATCH}...")
vae.fit(Xs, epochs=VAE_EPOCHS, batch_size=VAE_BATCH, verbose=1)
vae.save('vae_wadi.h5')
print("VAE pretraining complete; saved to vae_wadi.h5\n")

# ─── 3) State & Reward Functions ───────────────────────────────────────────────
def make_state(ts, c):
    if c < N_STEPS: return None
    W  = ts[feature_cols].iloc[c-N_STEPS+1:c+1].values.astype('float32')
    a0 = np.concatenate([W, np.zeros((N_STEPS,1),dtype='float32')], axis=1)
    a1 = np.concatenate([W, np.ones ((N_STEPS,1),dtype='float32')], axis=1)
    return np.stack([a0, a1], axis=0)

def reward_fn(ts, c, a, vae_model, coef):
    if c < N_STEPS: return [0,0]
    win = ts[feature_cols].iloc[c-N_STEPS+1:c+1].values.flatten().reshape(1,-1).astype('float32')
    pen = coef * np.mean((vae_model.predict(win) - win)**2)
    lbl = ts['label'].iat[c]
    base = [TN,FP] if lbl==0 else [FN,TP]
    return [base[0]+pen, base[1]+pen]

# ─── 4) Q-Network ───────────────────────────────────────────────────────────────
class QNet:
    def __init__(self, scope, lr=3e-4):
        self.scope = scope
        with tf.compat.v1.variable_scope(scope):
            self.S = tf.compat.v1.placeholder(tf.float32, [None, N_STEPS, N_INPUT_DIM], 'S')
            self.T = tf.compat.v1.placeholder(tf.float32, [None, ACTION_SPACE_N],    'T')
            seq = tf.compat.v1.unstack(self.S, N_STEPS, axis=1)
            cell= tf.compat.v1.nn.rnn_cell.LSTMCell(128)
            out,_= tf.compat.v1.nn.static_rnn(cell, seq, dtype=tf.float32)
            self.logits   = layers.Dense(ACTION_SPACE_N)(out[-1])
            self.loss     = tf.reduce_mean(tf.square(self.logits - self.T))
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)
    def predict(self, s, sess): return sess.run(self.logits, {self.S: s})
    def update(self, s, t, sess): sess.run(self.train_op, {self.S: s, self.T: t})

def epsilon_policy(est, nA, sess):
    def pol(obs, eps):
        A = np.ones(nA)*eps/nA
        q = est.predict([obs], sess)[0]
        b = np.argmax(q); A[b] += 1-eps
        return A
    return pol

def copy_params(sess, src, dst):
    vs = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=src.scope)
    vt = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=dst.scope)
    for s,t in zip(sorted(vs,key=lambda v:v.name), sorted(vt,key=lambda v:v.name)):
        sess.run(t.assign(s))

# ─── 5) Q-Learning with dynamic reward & Active Learning ───────────────────────
def update_coef(old, ep_reward, alpha=0.001, target=0.0, lo=0.1, hi=100.0):
    new = old + alpha*(ep_reward - target)
    return max(min(new, hi), lo)

def q_learning(env, sess, ql, qt, vae_model, init_coef):
    Transition = namedtuple('T',['s','r','ns','d'])
    sess.run(tf.compat.v1.global_variables_initializer())
    epss   = np.linspace(1.0, 0.1, 10000)
    policy = epsilon_policy(ql, ACTION_SPACE_N, sess)

    # Warm-up IsolationForest
    env.reset()
    W=[]
    for i in range(N_STEPS, len(env.timeseries)):
        if env.timeseries['label'].iat[i]==0:
            W.append(env.timeseries[feature_cols].iloc[i-N_STEPS+1:i+1].values.flatten())
    if len(W):
        IsolationForest(contamination=0.01).fit(np.array(W[:MAX_VAE_SAMPLES],dtype='float32'))

    dynamic_coef = init_coef
    memory       = []
    coef_history = []

    for ep in range(EPISODES):
        # Initialize uncertain list each episode
        uncert = []

        env.statefnc  = make_state
        env.rewardfnc = lambda ts,c,a,coef=dynamic_coef: reward_fn(ts,c,a,vae_model,coef)

        # Label Propagation
        labs = np.array(env.timeseries['label'].iloc[N_STEPS:])
        if np.any(labs != -1):
            arr  = np.array([s for s in env.states_list if s is not None])
            flat = arr.reshape(arr.shape[0],-1)
            lp   = LabelSpreading(kernel='knn', n_neighbors=10).fit(flat, labs)
            uncert = np.argsort(-np.max(lp.label_distributions_,axis=1))[:NUM_LP]
            for u in uncert:
                env.timeseries['label'].iat[u+N_STEPS] = lp.transduction_[u]

        # Active Learning
        for u in uncert[:NUM_AL]:
            env.timeseries['label'].iat[u+N_STEPS] = env.timeseries['anomaly'].iat[u+N_STEPS]

        # Rollout
        state, done = env.reset(), False
        ep_reward   = 0.0
        while not done:
            eps  = epss[min(ep, len(epss)-1)]
            probs= policy(state, eps)
            a    = np.random.choice(ACTION_SPACE_N, p=probs)
            raw, r, done, _ = env.step(a)
            ns            = raw[a] if raw.ndim>2 else raw
            memory.append(Transition(state, r, ns, done))
            ep_reward    += r[a]
            state         = ns

        # Training on replay memory
        for _ in range(5):
            batch = random.sample(memory, BATCH_SIZE)
            S,R,NS,_ = map(np.array, zip(*batch))
            q0 = qt.predict(NS[:,0], sess)
            q1 = qt.predict(NS[:,1], sess)
            tgt = R + DISCOUNT_FACTOR * np.stack((q0.max(1),q1.max(1)), axis=1)
            ql.update(S, tgt, sess)

        copy_params(sess, ql, qt)
        dynamic_coef = update_coef(dynamic_coef, ep_reward)
        coef_history.append(dynamic_coef)

    return ql, coef_history

# ─── 6) Validation ───────────────────────────────────────────────────────────────
def validate(env, sess, trained):
    split = int(len(env.timeseries_repo[0]) * VALIDATION_SPLIT)
    os.makedirs('validation', exist_ok=True)
    f1s, auprs = [], []
    with open('validation/perf.txt','w') as fout:
        for i in range(EPISODES):
            policy = epsilon_policy(trained, ACTION_SPACE_N, sess)
            state, done = env.reset(), False
            while env.timeseries_curser < split:
                raw,_,done,_ = env.step(0)
                state = raw[0] if raw.ndim>2 else raw
                if done: break
            preds, gts = [], []
            while not done:
                a = np.argmax(policy(state,0.0))
                preds.append(a)
                gts.append(env.timeseries['anomaly'].iat[env.timeseries_curser])
                raw,_,done,_ = env.step(a)
                state = raw[a] if raw.ndim>2 else raw
            p,r,f1,_ = precision_recall_fscore_support(gts,preds,average='binary',zero_division=0)
            aupr     = average_precision_score(gts,preds)
            fout.write(f"E{i+1}:P={p:.3f},R={r:.3f},F1={f1:.3f},AUPR={aupr:.3f}\n")
            f1s.append(f1); auprs.append(aupr)
    print(f"Validation complete: Avg F1={np.mean(f1s):.3f}, Avg AUPR={np.mean(auprs):.3f}")
    return np.mean(f1s), np.mean(auprs)

# ─── 7) Main wrapper ─────────────────────────────────────────────────────────────
def train_wrapper(nLP, nAL, discount):
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    from tensorflow.compat.v1.keras import backend as K; K.set_session(sess)

    vae_model = load_model('vae_wadi.h5', compile=False)
    env       = EnvTimeSeriesWaDi(SENSOR_CSV, LABEL_CSV, N_STEPS)
    ql, qt    = QNet('qlearn'), QNet('qtarget')

    trained, coefs = q_learning(env, sess, ql, qt, vae_model, init_coef=20.0)
    f1, aupr      = validate(EnvTimeSeriesWaDi(SENSOR_CSV, LABEL_CSV, N_STEPS), sess, trained)

    exp = f'exp_AL{nAL}'
    os.makedirs(os.path.join(exp,'plots'), exist_ok=True)
    plt.figure(); plt.plot(coefs); plt.title('dynamic_coef'); plt.savefig(os.path.join(exp,'plots','coef.png'))

if __name__ == "__main__":
    for AL in [1000, 5000, 10000]:
        train_wrapper(NUM_LP, AL, discount=0.96)

