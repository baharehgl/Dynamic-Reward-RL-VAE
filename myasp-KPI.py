# myasp-KPI.py

import os
import sys
import random
import itertools

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

# Ensure the KPI env is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from env_KPI import EnvKPI

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

# --------------------
# Hyperparameters
# --------------------
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

# --------------------
# State & Reward
# --------------------
def RNNBinaryStateFuc(ts, cursor, prev=None, action=None):
    if cursor < n_steps:
        return None
    if cursor == n_steps:
        st = [[ts['value'].iat[i],0] for i in range(n_steps)]
        st.pop(0)
        st.append([ts['value'].iat[cursor],1])
        return np.array(st,'float32')
    # cursor > n_steps
    s0 = np.concatenate((prev[1:], [[ts['value'].iat[cursor],0]]))
    s1 = np.concatenate((prev[1:], [[ts['value'].iat[cursor],1]]))
    return np.array([s0,s1],'float32')

def RNNBinaryRewardFuc(ts, cursor, action, vae_model, dynamic_coef=1.0):
    # only valid when cursor>=n_steps
    cur = np.array([ts['value'].iloc[cursor-n_steps:cursor]])
    recon = vae_model.predict(cur)
    err   = np.mean((recon-cur)**2)
    pen   = dynamic_coef * err
    lbl   = ts['label'].iat[cursor]
    if lbl == 0:
        return [TN_Value+pen, FP_Value+pen]
    else:
        return [FN_Value+pen, TP_Value+pen]

def RNNBinaryRewardFucTest(ts, cursor, action=0):
    an = ts['anomaly'].iat[cursor]
    if an == 0:
        return [TN_Value, FP_Value]
    else:
        return [FN_Value, TP_Value]

# --------------------
# VAE Setup
# --------------------
def load_normal_data(path, n_steps):
    windows = []
    for f in os.listdir(path):
        if not f.endswith('.csv'): continue
        vals = pd.read_csv(os.path.join(path,f))['value'].values
        if len(vals) < n_steps: continue
        for i in range(len(vals)-n_steps+1):
            windows.append(vals[i:i+n_steps])
    return StandardScaler().fit_transform(np.array(windows))

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.keras.backend.random_normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5*z_log_var)*eps

original_dim, latent_dim, intermediate_dim = n_steps, 10, 64

def build_vae():
    x_in = layers.Input((original_dim,))
    h = layers.Dense(intermediate_dim, activation='relu')(x_in)
    h = layers.Dense(intermediate_dim, activation='relu')(h)
    z_mean   = layers.Dense(latent_dim)(h)
    z_log_var= layers.Dense(latent_dim)(h)
    z_log_var= tf.clip_by_value(z_log_var, -10.0, 10.0)
    z        = Sampling()([z_mean, z_log_var])
    d = layers.Dense(intermediate_dim, activation='relu')(z)
    x_dec = layers.Dense(original_dim, activation='sigmoid')(d)

    vae = models.Model(x_in, x_dec)
    recon = losses.mse(x_in, x_dec)*original_dim
    kl    = -0.5*tf.reduce_sum(1+z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae.add_loss(tf.reduce_mean(recon+kl))
    vae.compile(optimizer='adam')
    return vae

# --------------------
# Q-Estimator
# --------------------
class Q_Estimator_Nonlinear:
    def __init__(self, lr, scope, sess):
        self.scope = scope
        self.sess  = sess
        with tf.compat.v1.variable_scope(scope):
            self.state  = tf.compat.v1.placeholder(tf.float32, [None,n_steps,n_input_dim], name="state")
            self.target = tf.compat.v1.placeholder(tf.float32, [None,len(action_space)],    name="target")
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(n_hidden_dim)
            outs,_ = tf.compat.v1.nn.dynamic_rnn(cell, self.state, dtype=tf.float32)
            last   = outs[:,-1,:]
            self.qvals = layers.Dense(len(action_space))(last)
            self.loss  = tf.reduce_mean(tf.square(self.qvals - self.target))
            self.train = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)

    def predict(self, s):
        return self.sess.run(self.qvals, {self.state:s})

    def update(self, s, t):
        self.sess.run(self.train, {self.state:s, self.target:t})

def make_epsilon_greedy_policy(estimator, nA):
    def policy_fn(obs, eps):
        A = np.ones(nA)*(eps/nA)
        q = estimator.predict([obs])[0]
        A[np.argmax(q)] += (1-eps)
        return A
    return policy_fn

def update_dynamic_coef_proportional(cur, rew, target=100.0, alpha=0.01, min_coef=0.1, max_coef=10.0):
    nc = cur + alpha*(target-rew)
    return max(min(nc, max_coef), min_coef)

# --------------------
# Active Learning
# --------------------
class active_learning:
    def __init__(self, env, N, estimator):
        self.env       = env
        self.N         = N
        self.estimator = estimator

    def get_samples(self):
        dists = []
        for st in self.env.states_list:
            q = self.estimator.predict([st])[0]
            dists.append(abs(q[0]-q[1]))
        order = np.argsort(dists)
        return order[:self.N].tolist()

# --------------------
# Train with validation
# --------------------
def train_with_validation(train_csv, test_csv,
                          num_episodes=100,
                          num_epoches=5,
                          validate_every=10,
                          patience=2,
                          num_AL=10,
                          discount_factor=0.96):

    # Build env & split indices
    env  = EnvKPI(train_csv, test_csv)
    env.statefnc = RNNBinaryStateFuc
    total = env.datasetsize
    cut   = int(total * validation_separate_ratio)
    train_idx = list(range(cut))
    val_idx   = list(range(cut, total))

    # Test env
    env_test = EnvKPI(train_csv, test_csv)
    env_test.statefnc  = RNNBinaryStateFuc
    env_test.rewardfnc = RNNBinaryRewardFucTest

    # Pretrain VAE
    vae = build_vae()
    x_tr= load_normal_data(os.path.join(current_dir,"normal-data"), n_steps)
    vae.fit(x_tr, epochs=5, batch_size=32, verbose=0)

    # TF session & nets
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sess)
    q_net   = Q_Estimator_Nonlinear(3e-4,    "q", sess)
    tgt_net = Q_Estimator_Nonlinear(3e-4, "target", sess)
    sess.run(tf.compat.v1.global_variables_initializer())

    best_val_f1 = 0.0
    no_imp      = 0
    dyn_coef    = 10.0
    hist_r, hist_c = [], []

    for ep in range(1, num_episodes+1):
        # choose train KPI
        ki = random.choice(train_idx)
        state = env.reset(to_idx=ki)
        env.rewardfnc = lambda ts,tc,a: RNNBinaryRewardFuc(ts,tc,a,vae,dyn_coef)
        env.states_list = env.get_states_list()

        # Active Learning
        al = active_learning(env, num_AL, q_net)
        samples = al.get_samples()
        for s in samples:
            pos = s + n_steps
            env.timeseries['label'].iat[pos] = env.timeseries['anomaly'].iat[pos]

        # Collect memory from RNN states
        memory = []
        for t, st in enumerate(env.states_list):
            probs = make_epsilon_greedy_policy(q_net, env.action_space_n)(st, max(0.1,1-ep/num_episodes))
            a     = np.random.choice(len(probs), p=probs)
            r     = env.rewardfnc(env.timeseries, t+n_steps, a)
            memory.append((st, r, st, False))

        # Train
        for _ in range(num_epoches):
            batch = random.sample(memory, min(len(memory),64))
            S, R, NS, D = map(np.array, zip(*batch))
            qn = tgt_net.predict(NS)
            mx = np.max(qn, axis=1)
            targets = R + discount_factor*mx[:,None]
            q_net.update(S, targets)

        # Sync
        if ep % 5 == 0:
            vq = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "q")
            vt = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "target")
            for x,y in zip(vq,vt):
                sess.run(y.assign(x))

        # Logging & coef
        total_r = sum(r[a] for (s,r,ns,d) in memory for a in [0])  # use first action's reward as proxy
        hist_r.append(total_r)
        dyn_coef = update_dynamic_coef_proportional(dyn_coef, total_r, target=0.0)
        hist_c.append(dyn_coef)

        # Validation
        if ep % validate_every == 0:
            f1s = []
            for vid in val_idx:
                env_test.reset(to_idx=vid)
                preds, truths = [], []
                for t, st in enumerate(env_test.get_states_list()):
                    a = np.argmax(make_epsilon_greedy_policy(q_net, env_test.action_space_n)(st,0))
                    preds.append(a)
                    truths.append(env_test.timeseries['anomaly'].iat[t+n_steps])
                _,_,f1,_ = precision_recall_fscore_support(truths, preds, average='binary', zero_division=0)
                f1s.append(f1)
            val_f1 = np.mean(f1s)
            print(f"[Val] Episode {ep}, F1 = {val_f1:.4f}")
            if val_f1 > best_val_f1:
                best_val_f1, no_imp = val_f1, 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    print(f"Early stopping at episode {ep}")
                    break

    # Save curves
    os.makedirs(os.path.join(current_dir,"exp"), exist_ok=True)
    plt.figure(); plt.plot(hist_r); plt.title("Reward"); plt.savefig("exp/reward.png"); plt.close()
    plt.figure(); plt.plot(hist_c); plt.title("Coef");   plt.savefig("exp/coef.png");   plt.close()

    print("Best validation F1:", best_val_f1)
    return hist_r, hist_c, best_val_f1

if __name__ == "__main__":
    train_csv = os.path.join(current_dir,"KPI_data","train","phase2_train.csv")
    test_csv  = os.path.join(current_dir,"KPI_data","test","phase2_ground_truth.csv")
    train_with_validation(train_csv, test_csv,
                          num_episodes=100,
                          num_epoches=5,
                          validate_every=10,
                          patience=2,
                          num_AL=10)
