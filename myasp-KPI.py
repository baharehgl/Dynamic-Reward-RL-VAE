# myasp-KPI.py

import os, sys, random, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.semi_supervised import LabelSpreading

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow.keras import layers, models, losses

# ensure env_KPI.py is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from env_KPI import EnvKPI

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

# --- Hyperparameters ---
EPISODES                  = 300
n_steps                   = 25
n_input_dim               = 2
n_hidden_dim              = 128
validation_separate_ratio = 0.9

TN_Value   = 1
TP_Value   = 5
FP_Value  = -1
FN_Value  = -5
NOT_ANOMALY = 0
ANOMALY     = 1
action_space= [NOT_ANOMALY, ANOMALY]

# --- VAE setup ---
def load_normal_data(path, n_steps):
    windows = []
    for f in os.listdir(path):
        if not f.endswith('.csv'): continue
        vals = pd.read_csv(os.path.join(path, f))['value'].values
        if len(vals) < n_steps: continue
        for i in range(len(vals)-n_steps+1):
            windows.append(vals[i:i+n_steps])
    arr = np.array(windows)
    return StandardScaler().fit_transform(arr)

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.keras.backend.random_normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5*z_log_var)*eps

original_dim, latent_dim, intermediate_dim = n_steps, 10, 64

def build_vae(original_dim, latent_dim=2, intermediate_dim=64):
    x = layers.Input((original_dim,))
    h = layers.Dense(intermediate_dim, activation='relu')(x)
    h = layers.Dense(intermediate_dim, activation='relu')(h)
    z_mean   = layers.Dense(latent_dim)(h)
    z_log_var= layers.Dense(latent_dim)(h)
    z_log_var= tf.clip_by_value(z_log_var, -10.0, 10.0)
    z        = Sampling()([z_mean, z_log_var])
    dec_h = layers.Dense(intermediate_dim, activation='relu')(z)
    x_dec = layers.Dense(original_dim, activation='sigmoid')(dec_h)

    vae = models.Model(x, x_dec)
    recon = losses.mse(x, x_dec)*original_dim
    kl    = -0.5 * tf.reduce_sum(1+z_log_var - tf.square(z_mean)-tf.exp(z_log_var), axis=-1)
    vae.add_loss(tf.reduce_mean(recon+kl))
    vae.compile(optimizer='adam')
    return vae

# --- State & reward ---
def RNNBinaryStateFuc(ts, cursor, prev=None, action=None):
    if cursor == n_steps:
        st = [[ts['value'].iat[i],0] for i in range(n_steps)]
        st.pop(0); st.append([ts['value'].iat[cursor],1])
        return np.array(st,'float32')
    if cursor > n_steps:
        s0 = np.concatenate((prev[1:], [[ts['value'].iat[cursor],0]]))
        s1 = np.concatenate((prev[1:], [[ts['value'].iat[cursor],1]]))
        return np.array([s0,s1],'float32')
    return None

def RNNBinaryRewardFuc(ts, cursor, action, vae_model, dynamic_coef=1.0):
    if cursor < n_steps:
        return [0,0]
    cur = np.array([ts['value'].iloc[cursor-n_steps:cursor]])
    recon = vae_model.predict(cur)
    err = np.mean((recon-cur)**2)
    pen = dynamic_coef*err
    lbl = ts['label'].iat[cursor]
    if lbl==0: return [TN_Value+pen, FP_Value+pen]
    if lbl==1: return [FN_Value+pen, TP_Value+pen]
    return [0,0]

def RNNBinaryRewardFucTest(ts, cursor, action=0):
    if cursor < n_steps: return [0,0]
    an = ts['anomaly'].iat[cursor]
    return [TN_Value, FP_Value] if an==0 else [FN_Value, TP_Value]

# --- Q-estimator & helpers ---
class Q_Estimator_Nonlinear:
    def __init__(self, lr=1e-3, scope="Q"):
        self.scope = scope
        with tf.compat.v1.variable_scope(scope):
            self.state = tf.compat.v1.placeholder(tf.float32, [None,n_steps,n_input_dim], name="state")
            self.target= tf.compat.v1.placeholder(tf.float32, [None,len(action_space)], name="target")
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(n_hidden_dim)
            outs, _ = tf.compat.v1.nn.dynamic_rnn(cell, self.state, dtype=tf.float32)
            last = outs[:, -1, :]
            self.qvals = layers.Dense(len(action_space))(last)
            self.loss = tf.reduce_mean(tf.square(self.qvals-self.target))
            self.train = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)
    def predict(self, s, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        return sess.run(self.qvals, {self.state:s})
    def update(self, s, t, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        sess.run(self.train, {self.state:s, self.target:t})

def make_epsilon_greedy_policy(estimator, nA, sess):
    def policy_fn(obs, eps):
        A = np.ones(nA)*(eps/nA)
        q = estimator.predict([obs], sess=sess)[0]
        A[np.argmax(q)] += (1-eps)
        return A
    return policy_fn

def update_dynamic_coef_proportional(cur, rew, target=100.0, alpha=0.01, min_coef=0.1, max_coef=10.0):
    nc = cur + alpha*(target-rew)
    return max(min(nc, max_coef), min_coef)

# --- Train with validation & early stopping ---
def train_with_validation(train_csv, test_csv,
                          num_episodes=300,
                          num_epoches=10,
                          validate_every=10,
                          patience=3,
                          discount_factor=0.96):
    # build full env
    env = EnvKPI(train_csv, test_csv)
    env.statefnc   = RNNBinaryStateFuc
    env_full_size  = env.datasetsize
    train_cut      = int(env_full_size * validation_separate_ratio)
    train_indices  = list(range(train_cut))
    val_indices    = list(range(train_cut, env_full_size))

    # separate test env
    env_test = EnvKPI(train_csv, test_csv)
    env_test.statefnc = RNNBinaryStateFuc
    env_test.rewardfnc= RNNBinaryRewardFucTest

    # pretrain VAE
    x_train = load_normal_data(os.path.join(current_dir,"normal-data"), n_steps)
    vae_model = build_vae(original_dim, latent_dim, intermediate_dim)
    vae_model.fit(x_train, epochs=5, batch_size=32, verbose=0)

    # TF session & networks
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sess)
    q_net   = Q_Estimator_Nonlinear(scope="Q")
    tgt_net = Q_Estimator_Nonlinear(scope="target")
    sess.run(tf.compat.v1.global_variables_initializer())

    best_val_f1 = 0.0
    no_improve  = 0
    dynamic_coef= 10.0
    history_r, history_c = [], []

    for ep in range(1, num_episodes+1):
        # pick a random train KPI
        idx = random.choice(train_indices)
        state = env.reset(to_idx=idx)
        env.rewardfnc = lambda ts,tc,a: RNNBinaryRewardFuc(ts,tc,a,vae_model,dynamic_coef)
        env.states_list = env.get_states_list()

        # run one episode
        done = False
        episode_reward = 0.0
        memory = []
        while not done:
            probs = make_epsilon_greedy_policy(q_net, env.action_space_n, sess)(state, max(0.1,1-ep/num_episodes))
            a = np.random.choice(len(probs), p=probs)
            nxt, r, done, _ = env.step(a)
            episode_reward += r[a]
            memory.append((state, r, nxt, done))
            state = nxt[a]

        # train on that memory
        for _ in range(num_epoches):
            batch = random.sample(memory, min(len(memory),64))
            states, rewards, next_states, dones = map(np.array, zip(*batch))
            q_next = tgt_net.predict(next_states, sess)
            max_n  = np.max(q_next, axis=1)
            targets= rewards + discount_factor * max_n[:,None]
            q_net.update(states, targets, sess)
        # update target network occasionally
        if ep % 5 == 0:
            vars_q = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "Q")
            vars_t = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "target")
            for vq, vt in zip(vars_q, vars_t):
                sess.run(vt.assign(vq))

        history_r.append(episode_reward)
        dynamic_coef = update_dynamic_coef_proportional(dynamic_coef, episode_reward,
                                                        target=0.0, alpha=0.001)
        history_c.append(dynamic_coef)

        # validation check
        if ep % validate_every == 0:
            f1s = []
            for vid in val_indices:
                st = env.reset(to_idx=vid)
                env.rewardfnc = RNNBinaryRewardFucTest
                done = False
                preds, truths = [], []
                while not done:
                    a = np.argmax(make_epsilon_greedy_policy(q_net, env.action_space_n, sess)(st,0))
                    preds.append(a)
                    truths.append(env.timeseries['anomaly'].iat[env.timeseries_cursor])
                    nxt, _, done, _ = env.step(a)
                    st = nxt[a]
                _,_,f1,_ = precision_recall_fscore_support(truths, preds,
                                    average='binary', zero_division=0)
                f1s.append(f1)
            val_f1 = np.mean(f1s)
            print(f"[Validation] ep {ep}, F1={val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at ep {ep}")
                    break

    # save plots
    os.makedirs(os.path.join(current_dir,"exp"), exist_ok=True)
    plt.figure(); plt.plot(history_r); plt.title("Reward"); plt.savefig("exp/reward.png"); plt.close()
    plt.figure(); plt.plot(history_c); plt.title("Coef");   plt.savefig("exp/coef.png");   plt.close()

    print("Best validation F1:", best_val_f1)
    return history_r, history_c, best_val_f1

if __name__ == "__main__":
    train_csv = os.path.join(current_dir,"KPI_data","train","phase2_train.csv")
    test_csv  = os.path.join(current_dir,"KPI_data","test", "phase2_ground_truth.csv")
    train_with_validation(train_csv, test_csv,
                          num_episodes=100,
                          num_epoches=5,
                          validate_every=10,
                          patience=2)
