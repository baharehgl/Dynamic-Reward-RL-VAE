# myasp-KPI.py

import os
import sys
import time
import random
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import precision_recall_fscore_support

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow.keras import layers, models, losses

# make sure we can import env_KPI.py from same folder
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from env_KPI import EnvKPI

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

# ----- Hyperparameters -----
EPISODES                  = 300
n_steps                   = 25
n_input_dim               = 2
n_hidden_dim              = 128
validation_separate_ratio = 0.9

TN_Value = 1; TP_Value = 5; FP_Value = -1; FN_Value = -5
NOT_ANOMALY = 0; ANOMALY = 1
action_space = [NOT_ANOMALY, ANOMALY]

# ----- VAE setup -----
def load_normal_data(data_path, n_steps):
    windows = []
    for fname in os.listdir(data_path):
        if not fname.endswith('.csv'): continue
        vals = pd.read_csv(os.path.join(data_path, fname))['value'].values
        if len(vals) < n_steps: continue
        for i in range(len(vals) - n_steps + 1):
            windows.append(vals[i:i + n_steps])
    arr = np.array(windows)
    return StandardScaler().fit_transform(arr)

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim   = tf.shape(z_mean)[1]
        eps   = tf.keras.backend.random_normal((batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

original_dim, latent_dim, intermediate_dim = n_steps, 10, 64

def build_vae(original_dim, latent_dim=2, intermediate_dim=64):
    x_in = layers.Input((original_dim,))
    h1   = layers.Dense(intermediate_dim, activation='relu')(x_in)
    h2   = layers.Dense(intermediate_dim, activation='relu')(h1)
    z_mean   = layers.Dense(latent_dim)(h2)
    z_log_var= layers.Dense(latent_dim)(h2)
    z_log_var= tf.clip_by_value(z_log_var, -10.0, 10.0)
    z        = Sampling()([z_mean, z_log_var])
    dec_h    = layers.Dense(intermediate_dim, activation='relu')(z)
    x_dec    = layers.Dense(original_dim, activation='sigmoid')(dec_h)

    vae = models.Model(x_in, x_dec)
    recon = losses.mse(x_in, x_dec) * original_dim
    kl    = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae.add_loss(tf.reduce_mean(recon + kl))
    vae.compile(optimizer='adam')
    return vae

# ----- State & Reward -----
def RNNBinaryStateFuc(ts, cursor, prev=None, action=None):
    if cursor == n_steps:
        st = [[ts['value'].iat[i], 0] for i in range(n_steps)]
        st.pop(0); st.append([ts['value'].iat[cursor], 1])
        return np.array(st, 'float32')
    if cursor > n_steps:
        s0 = np.concatenate((prev[1:], [[ts['value'].iat[cursor], 0]]))
        s1 = np.concatenate((prev[1:], [[ts['value'].iat[cursor], 1]]))
        return np.array([s0, s1], 'float32')
    return None

def RNNBinaryRewardFuc(ts, cursor, action, vae_model, dynamic_coef=1.0):
    if cursor < n_steps:
        return [0, 0]
    cur    = np.array([ts['value'].iloc[cursor-n_steps:cursor]])
    recon  = vae_model.predict(cur)
    err    = np.mean((recon - cur)**2)
    penalty= dynamic_coef * err
    lbl    = ts['label'].iat[cursor]
    if lbl == 0: return [TN_Value + penalty, FP_Value + penalty]
    if lbl == 1: return [FN_Value + penalty, TP_Value + penalty]
    return [0, 0]

def RNNBinaryRewardFucTest(ts, cursor, action=0):
    if cursor < n_steps:
        return [0, 0]
    an = ts['anomaly'].iat[cursor]
    return [TN_Value, FP_Value] if an == 0 else [FN_Value, TP_Value]

# ----- Q-estimator & helpers -----
class Q_Estimator_Nonlinear:
    def __init__(self, learning_rate=0.001, scope="Q"):
        self.scope = scope
        with tf.compat.v1.variable_scope(scope):
            self.state  = tf.compat.v1.placeholder(tf.float32, [None, n_steps, n_input_dim], name="state")
            self.target = tf.compat.v1.placeholder(tf.float32, [None, len(action_space)],    name="target")
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(n_hidden_dim)
            outs, _ = tf.compat.v1.nn.dynamic_rnn(cell, self.state, dtype=tf.float32)
            last = outs[:, -1, :]
            self.qvals = layers.Dense(len(action_space))(last)
            self.loss  = tf.reduce_mean(tf.square(self.qvals - self.target))
            self.train = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def predict(self, state, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        return sess.run(self.qvals, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        sess.run(self.train, {self.state: state, self.target: target})

def make_epsilon_greedy_policy(estimator, nA, sess):
    def policy_fn(obs, eps):
        A = np.ones(nA, dtype='float32') * (eps / nA)
        q = estimator.predict([obs], sess=sess)[0]
        A[np.argmax(q)] += (1.0 - eps)
        return A
    return policy_fn

def update_dynamic_coef_proportional(cur, rew, target=100.0, alpha=0.01, min_coef=0.1, max_coef=10.0):
    nc = cur + alpha * (target - rew)
    return max(min(nc, max_coef), min_coef)

# ----- Early-stopping training & validation -----
def train_with_validation(train_csv, test_csv,
                          num_episodes=300,
                          num_epoches=10,
                          validate_every=10,
                          patience=3,
                          num_LP=200,
                          num_AL=1000,
                          discount_factor=0.96):
    # Build full env and split indices
    env_full  = EnvKPI(train_csv, test_csv)
    total     = env_full.datasetsize
    train_cut = int(total * validation_separate_ratio)
    train_idx = list(range(train_cut))
    val_idx   = list(range(train_cut, total))

    # Wrapper reset to sample only from train_idx
    def train_reset(to_idx=None):
        i = random.choice(train_idx) if to_idx is None else to_idx
        return env_full.reset(to_idx=i)
    env_full.reset = train_reset
    env_full.statefnc = RNNBinaryStateFuc

    # Validation env
    env_val = EnvKPI(train_csv, test_csv)
    env_val.statefnc  = RNNBinaryStateFuc
    env_val.rewardfnc = RNNBinaryRewardFucTest

    # Pretrain VAE
    x_train = load_normal_data(os.path.join(current_dir, "normal-data"), n_steps)
    vae_model = build_vae(original_dim, latent_dim, intermediate_dim)
    vae_model.fit(x_train, epochs=5, batch_size=32, verbose=0)

    # TF session + Q nets
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sess)
    q_net   = Q_Estimator_Nonlinear(scope="qlearn")
    tgt_net = Q_Estimator_Nonlinear(scope="target")
    sess.run(tf.compat.v1.global_variables_initializer())

    best_val_f1 = 0.0
    no_improve  = 0
    dynamic_coef = 10.0
    history_rewards = []
    history_coefs   = []

    for ep in range(1, num_episodes+1):
        # TRAIN one episode
        env_full.rewardfnc = lambda ts,tc,a: RNNBinaryRewardFuc(ts,tc,a,vae_model,dynamic_coef)
        state = env_full.reset()
        env_full.states_list = env_full.get_states_list()
        episode_reward = 0.0
        replay_memory = []

        # collect transitions for 1 episode
        done = False
        while not done:
            probs = make_epsilon_greedy_policy(q_net, env_full.action_space_n, sess)(state, 1.0-ep/num_episodes)
            action = np.random.choice(len(probs), p=probs)
            nxt, r, done, _ = env_full.step(action)
            episode_reward += r[action]
            replay_memory.append((state, r, nxt, done))
            state = nxt[action]

        # TRAIN on collected transitions
        for _ in range(num_epoches):
            batch = random.sample(replay_memory, min(len(replay_memory), 64))
            states, rewards, next_states, dones = map(np.array, zip(*batch))
            # build targets
            q0 = tgt_net.predict(next_states, sess)
            # simply use max over actions
            max_next = np.max(q0, axis=1)
            targets = rewards + discount_factor * max_next[:, None]
            q_net.update(states, targets, sess)
        # periodically update target net
        if ep % 5 == 0:
            vars_q   = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "qlearn")
            vars_tgt = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "target")
            for vq, vt in zip(vars_q, vars_tgt):
                sess.run(vt.assign(vq))

        # record & update coef
        history_rewards.append(episode_reward)
        dynamic_coef = update_dynamic_coef_proportional(dynamic_coef, episode_reward,
                                                        target=0.0, alpha=0.001)
        history_coefs.append(dynamic_coef)

        # VALIDATION check
        if ep % validate_every == 0:
            f1s = []
            for vid in val_idx:
                st = env_val.reset(to_idx=vid)
                done = False
                preds, truths = [], []
                while not done:
                    act = np.argmax(make_epsilon_greedy_policy(q_net, env_val.action_space_n, sess)(st, 0.0))
                    preds.append(act)
                    truths.append(env_val.timeseries['anomaly'].iat[env_val.timeseries_cursor])
                    nxt, _, done, _ = env_val.step(act)
                    st = nxt[act]
                _,_,f1,_ = precision_recall_fscore_support(truths, preds, average='binary', zero_division=0)
                f1s.append(f1)
            val_f1 = np.mean(f1s)
            print(f"[Validation] Episode {ep}, F1 = {val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at episode {ep} (no improvement in {patience} checks)")
                    break

    # Plot training reward & coef curves
    os.makedirs(os.path.join(current_dir, "exp"), exist_ok=True)
    plt.figure(); plt.plot(history_rewards); plt.title("Reward"); plt.savefig("exp/reward.png"); plt.close()
    plt.figure(); plt.plot(history_coefs);   plt.title("Coef");   plt.savefig("exp/coef.png");   plt.close()

    print("Best validation F1:", best_val_f1)
    return history_rewards, history_coefs, best_val_f1

if __name__ == "__main__":
    train_csv = os.path.join(current_dir, "KPI_data", "train", "phase2_train.csv")
    test_csv  = os.path.join(current_dir, "KPI_data", "test",  "phase2_ground_truth.csv")

    # run with early stopping
    train_with_validation(train_csv, test_csv,
                          num_episodes=100,
                          num_epoches=5,
                          validate_every=10,
                          patience=2)
