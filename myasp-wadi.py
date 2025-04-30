import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import namedtuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from sklearn.semi_supervised import LabelSpreading
from sklearn.ensemble import IsolationForest
from tensorflow.keras import layers, models, losses
from tensorflow.keras.models import load_model

from env_wadi import EnvTimeSeriesWaDi

# disable eager so we can use tf.compat.v1.Session
tf.compat.v1.disable_eager_execution()

# ─── HYPERPARAMETERS ─────────────────────────────────────────────────────────────
EPISODES                  = 2
N_STEPS                   = 25
DISCOUNT_FACTOR           = 0.5
TN, TP, FP, FN            = 1, 10, -1, -10
ACTION_SPACE_N            = 2
VALIDATION_SEPARATE_RATIO = 0.8
MAX_WARMUP_SAMPLES        = 10000
NUM_LABELPROPAGATION      = 200

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
WA_DI_DIR  = os.path.join(BASE_DIR, 'WaDi')
SENSOR_CSV = os.path.join(WA_DI_DIR, 'WADI_14days_new.csv')
LABEL_CSV  = os.path.join(WA_DI_DIR, 'WADI_attackdataLABLE.csv')

# ─── DISCOVER FEATURE COLUMNS ────────────────────────────────────────────────────
# assume SENSOR_CSV has one header row listing all sensor features
_sensor_df   = pd.read_csv(SENSOR_CSV, nrows=1)
feature_cols = list(_sensor_df.columns)           # all sensor measurements
n_features   = len(feature_cols)
N_INPUT_DIM  = n_features + 1                     # plus action flag

# ─── VARIATIONAL AUTOENCODER ─────────────────────────────────────────────────────
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim   = tf.shape(z_mean)[1]
        eps   = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

def build_vae(original_dim, latent_dim=10, intermediate_dim=64):
    inp       = layers.Input(shape=(original_dim,))
    h         = layers.Dense(intermediate_dim, activation='relu')(inp)
    z_mean    = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
    z         = Sampling()([z_mean, z_log_var])
    h_dec     = layers.Dense(intermediate_dim, activation='relu')(z)
    out       = layers.Dense(original_dim, activation='sigmoid')(h_dec)

    vae       = models.Model(inp, out)
    recon     = losses.mse(inp, out) * original_dim
    kl        = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae.add_loss(tf.reduce_mean(recon + kl))
    vae.compile(optimizer='adam')
    return vae

# ─── STATE & REWARD FUNCTIONS ───────────────────────────────────────────────────
def RNNBinaryStateFuc(ts, c, prev=None, action=None):
    if c < N_STEPS:
        return None
    # get last N_STEPS sensor windows
    window = ts[feature_cols].iloc[c-N_STEPS+1:c+1].values.astype(np.float32)  # (N_STEPS, n_features)
    # append action bit to each time-step
    a0 = np.concatenate([window, np.zeros((N_STEPS,1),dtype=np.float32)], axis=1)
    a1 = np.concatenate([window, np.ones((N_STEPS,1), dtype=np.float32)], axis=1)
    return np.stack([a0, a1], axis=0)  # shape (2, N_STEPS, N_INPUT_DIM)

def RNNBinaryRewardFuc(ts, c, action, vae_model=None, coef=1.0, include_vae=True):
    if c < N_STEPS:
        return [0, 0]
    penalty = 0.0
    if include_vae and vae_model is not None:
        win = ts[feature_cols].iloc[c-N_STEPS+1:c+1].values.astype(np.float32)
        win_flat = win.flatten().reshape(1, -1)
        recon    = vae_model.predict(win_flat)
        penalty  = coef * np.mean((recon - win_flat) ** 2)
    lbl = ts['label'].iat[c]
    return [TN + penalty, FP + penalty] if lbl == 0 else [FN + penalty, TP + penalty]

def RNNBinaryRewardFucTest(ts, c, action):
    if c < N_STEPS:
        return [0, 0]
    lbl = ts['anomaly'].iat[c]
    return [TN, FP] if lbl == 0 else [FN, TP]

# ─── Q-NETWORK ───────────────────────────────────────────────────────────────────
class Q_Estimator_Nonlinear:
    def __init__(self, lr=3e-4, scope='q'):
        self.scope = scope
        with tf.compat.v1.variable_scope(scope):
            self.state  = tf.compat.v1.placeholder(tf.float32, [None, N_STEPS, N_INPUT_DIM], name='state')
            self.target = tf.compat.v1.placeholder(tf.float32, [None, ACTION_SPACE_N],    name='target')
            seq = tf.compat.v1.unstack(self.state, N_STEPS, axis=1)
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(N_HIDDEN_DIM)
            out, _ = tf.compat.v1.nn.static_rnn(cell, seq, dtype=tf.float32)
            self.logits   = layers.Dense(ACTION_SPACE_N)(out[-1])
            self.loss     = tf.reduce_mean(tf.square(self.logits - self.target))
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)

    def predict(self, state, sess):
        return sess.run(self.logits, {self.state: state})

    def update(self, state, target, sess):
        sess.run(self.train_op, {self.state: state, self.target: target})

# ─── POLICY & UTILITIES ─────────────────────────────────────────────────────────
def make_epsilon_greedy_policy(estimator, nA, sess):
    def policy_fn(obs, eps):
        A = np.ones(nA) * eps / nA
        q = estimator.predict([obs], sess)[0]
        b = np.argmax(q)
        A[b] += (1.0 - eps)
        return A
    return policy_fn

def safe_argmax(probs):
    return 0 if probs is None or len(probs)==0 else np.argmax(probs)

def safe_choice(probs):
    return 0 if probs is None or len(probs)==0 else np.random.choice(len(probs), p=probs)

def copy_model_parameters(sess, src, dest):
    sv = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=src.scope)
    dv = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=dest.scope)
    for s_var, d_var in zip(sorted(sv, key=lambda v: v.name), sorted(dv, key=lambda v: v.name)):
        sess.run(d_var.assign(s_var))

# ─── Q-LEARNING ─────────────────────────────────────────────────────────────────
def q_learning(env, sess, ql, qt,
               num_episodes, num_epochs,
               update_target_every,
               eps_start, eps_end, eps_steps,
               batch_size,
               coef, discount,
               num_LP, num_AL):
    T   = namedtuple('T', ['s','r','ns','d'])
    mem = []
    sess.run(tf.compat.v1.global_variables_initializer())
    epsilons = np.linspace(eps_start, eps_end, eps_steps)
    policy   = make_epsilon_greedy_policy(ql, ACTION_SPACE_N, sess)

    # ── Warm-up: train an IsolationForest on normal windows ─────────────
    windows = []
    # env.reset() populates env.timeseries
    env.reset()
    for i in range(N_STEPS, len(env.timeseries)):
        if env.timeseries['label'].iat[i] == 0:
            w = env.timeseries[feature_cols].iloc[i-N_STEPS+1:i+1].values.flatten()
            windows.append(w)
    if windows:
        wf = np.array(windows[:MAX_WARMUP_SAMPLES], dtype=np.float32)
        IsolationForest(contamination=0.01).fit(wf)

    # ── Main RL loop ───────────────────────────────────────────────────────
    for ep in range(num_episodes):
        # Label Spreading
        labs = np.array(env.timeseries['label'].iloc[N_STEPS:])
        if np.any(labs != -1):
            arr  = np.array([state for state in env.states_list if state is not None])
            flat = arr.reshape(arr.shape[0], -1)
            lp   = LabelSpreading(kernel='knn', n_neighbors=10)
            lp.fit(flat, labs)
            uncert = np.argsort(-np.max(lp.label_distributions_, axis=1))[:num_LP]
            for u in uncert:
                env.timeseries['label'].iat[u + N_STEPS] = lp.transduction_[u]

        # Active Learning
        if 'uncert' in locals():
            for u in uncert[:num_AL]:
                env.timeseries['label'].iat[u + N_STEPS] = env.timeseries['anomaly'].iat[u + N_STEPS]

        # rollout
        state, ep_r = env.reset(), 0
        while True:
            eps   = epsilons[min(ep, len(epsilons)-1)]
            probs = policy(state, eps)
            a     = safe_choice(probs)
            raw, r, done, _ = env.step(a)
            state = raw[a] if (isinstance(raw, np.ndarray) and raw.ndim>2) else raw
            ep_r += r[a]
            mem.append(T(state, r, raw, done))
            if done:
                break

        # train on minibatches
        for _ in range(num_epochs):
            batch = random.sample(mem, batch_size)
            S, R, NS, D = map(np.array, zip(*batch))
            # compute targets
            q0 = qt.predict(NS[:,0], sess)
            q1 = qt.predict(NS[:,1], sess)
            tgt = R + discount * np.stack((q0.max(1), q1.max(1)), axis=1)
            ql.update(S, tgt, sess)

        # periodically update target network
        if ep % update_target_every == 0:
            copy_model_parameters(sess, ql, qt)

# ─── VALIDATION ────────────────────────────────────────────────────────────────
def q_learning_validator(env, sess, trained, split_idx, record_dir, plot=True):
    os.makedirs(record_dir, exist_ok=True)
    with open(os.path.join(record_dir,'perf.txt'),'w') as f:
        all_f1, all_aupr = [], []
        for i in range(EPISODES):
            policy = make_epsilon_greedy_policy(trained, ACTION_SPACE_N, sess)
            state  = env.reset()
            # skip training portion
            while env.timeseries_curser < split_idx:
                raw, _, done, _ = env.step(0)
                state = raw[0] if (isinstance(raw,np.ndarray) and raw.ndim>2) else raw
                if done: break

            preds, gts, vals = [], [], []
            while True:
                probs = policy(state, 0.0)
                a     = safe_argmax(probs)
                preds.append(a)
                gts.append(env.timeseries['anomaly'].iat[env.timeseries_curser])
                vals.append(state[-1][0])
                raw, _, done, _ = env.step(a)
                state = raw[a] if (isinstance(raw,np.ndarray) and raw.ndim>2) else raw
                if done: break

            p, r, f1, _ = precision_recall_fscore_support(gts, preds, average='binary', zero_division=0)
            aupr         = average_precision_score(gts, preds)
            f.write(f"E{i+1}:P={p:.3f},R={r:.3f},F1={f1:.3f},AUPR={aupr:.3f}\n")
            all_f1.append(f1); all_aupr.append(aupr)

            if plot:
                fig, ax = plt.subplots(4,1,sharex=True)
                ax[0].plot(vals);        ax[0].set_title('Sensor Value')
                ax[1].plot(preds,'g-');  ax[1].set_title('Prediction')
                ax[2].plot(gts,'r-');    ax[2].set_title('Ground Truth')
                ax[3].plot([aupr]*len(vals)); ax[3].set_title('AUPR')
                fig.savefig(os.path.join(record_dir, f'v{i+1}.png'))
                plt.close(fig)

    return np.mean(all_f1), np.mean(all_aupr)

# ─── PRETRAIN VAE ───────────────────────────────────────────────────────────────
def pretrain_vae():
    tf.compat.v1.reset_default_graph()
    from tensorflow.compat.v1.keras import backend as K
    sess = tf.compat.v1.Session()
    K.set_session(sess)

    df_raw = pd.read_csv(SENSOR_CSV)
    raw_lbl= pd.read_csv(LABEL_CSV, header=1, low_memory=False)["Attack LABLE (1:No Attack, -1:Attack)"].astype(int).values
    labels = np.where(raw_lbl==1, 0, 1)
    L      = min(len(df_raw), len(labels))

    # build windows of normal data
    W = []
    for i in range(N_STEPS, L):
        if labels[i] == 0:
            win = df_raw[feature_cols].iloc[i-N_STEPS+1:i+1].values
            W.append(win.flatten())
    X  = np.array(W, dtype=np.float32)
    Xs = StandardScaler().fit_transform(X)

    vae = build_vae(original_dim=N_STEPS * n_features)
    vae.fit(Xs, epochs=2, batch_size=32, verbose=1)
    vae.save('vae_wadi.h5')
    sess.close()

# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    pretrain_vae()
    for AL in [1000, 5000, 10000]:
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        from tensorflow.compat.v1.keras import backend as K2
        K2.set_session(sess)

        vae_model = load_model('vae_wadi.h5', custom_objects={'Sampling': Sampling}, compile=False)
        env       = EnvTimeSeriesWaDi(SENSOR_CSV, LABEL_CSV, N_STEPS)
        env.statefnc  = RNNBinaryStateFuc
        env.rewardfnc = lambda ts,tc,a: RNNBinaryRewardFuc(ts,tc,a,vae_model,coef=20.0,include_vae=True)

        ql = Q_Estimator_Nonlinear(scope='qlearn')
        qt = Q_Estimator_Nonlinear(scope='qtarget')
        split_idx = int(len(env.timeseries_repo[0]) * VALIDATION_SEPARATE_RATIO)

        q_learning(env, sess, ql, qt,
                   num_episodes=EPISODES,
                   num_epochs=5,
                   update_target_every=1,
                   eps_start=1.0,
                   eps_end=0.1,
                   eps_steps=10000,
                   batch_size=128,
                   coef=20.0,
                   discount=DISCOUNT_FACTOR,
                   num_LP=NUM_LABELPROPAGATION,
                   num_AL=AL)

        exp_dir = f'exp_AL{AL}'
        val_dir = os.path.join(exp_dir, 'validation')
        f1, aupr = q_learning_validator(env, sess, ql, split_idx, val_dir, plot=True)
        print(f"AL={AL}: F1={f1:.3f}, AUPR={aupr:.3f}")

if __name__ == "__main__":
    main()
