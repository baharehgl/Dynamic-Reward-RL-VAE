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

TN_Value   = 1; TP_Value = 5; FP_Value = -1; FN_Value = -5
NOT_ANOMALY = 0; ANOMALY = 1
action_space= [NOT_ANOMALY, ANOMALY]

# --- VAE setup omitted for brevity (same as before) ---
# --- State & reward omitted for brevity (same) ---

# --- Q-estimator & helpers ---
class Q_Estimator_Nonlinear:
    def __init__(self, lr=1e-3, scope="Q", sess=None):
        self.scope = scope
        self.sess  = sess
        with tf.compat.v1.variable_scope(scope):
            self.state  = tf.compat.v1.placeholder(tf.float32, [None,n_steps,n_input_dim], name="state")
            self.target = tf.compat.v1.placeholder(tf.float32, [None,len(action_space)],    name="target")
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(n_hidden_dim)
            outs,_ = tf.compat.v1.nn.dynamic_rnn(cell, self.state, dtype=tf.float32)
            last   = outs[:,-1,:]
            self.qvals = layers.Dense(len(action_space))(last)
            self.loss  = tf.reduce_mean(tf.square(self.qvals-self.target))
            self.train = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)

    def predict(self, s, sess=None):
        sess = sess or self.sess
        return sess.run(self.qvals, {self.state:s})

    def update(self, s, t, sess=None):
        sess = sess or self.sess
        sess.run(self.train, {self.state:s, self.target:t})

def make_epsilon_greedy_policy(estimator, nA, sess):
    def policy_fn(obs, eps):
        A = np.ones(nA)*(eps/nA)
        q = estimator.predict([obs], sess=sess)[0]
        A[np.argmax(q)] += (1-eps)
        return A
    return policy_fn

# --- Active Learning ---
class active_learning:
    def __init__(self, env, N, estimator):
        self.env       = env
        self.N         = N
        self.estimator = estimator

    def get_samples(self):
        # use estimator.sess internally
        dists = []
        for st in self.env.states_list:
            q = self.estimator.predict([st])[0]
            dists.append(abs(q[0]-q[1]))
        order = np.argsort(dists)
        return order[:self.N].tolist()

# --- Training with validation & active learning ---
def train_with_validation(train_csv, test_csv,
                          num_episodes=100,
                          num_epoches=5,
                          validate_every=10,
                          patience=2,
                          num_AL=10,
                          discount_factor=0.96):

    # build and split env
    env = EnvKPI(train_csv, test_csv)
    env.statefnc = RNNBinaryStateFuc
    total = env.datasetsize
    cut   = int(total*validation_separate_ratio)
    train_idx = list(range(cut))
    val_idx   = list(range(cut, total))

    # test env
    env_test = EnvKPI(train_csv, test_csv)
    env_test.statefnc  = RNNBinaryStateFuc
    env_test.rewardfnc = RNNBinaryRewardFucTest

    # pretrain VAE
    xtr = load_normal_data(os.path.join(current_dir,"normal-data"), n_steps)
    vae = build_vae(original_dim, latent_dim, intermediate_dim)
    vae.fit(xtr, epochs=5, batch_size=32, verbose=0)

    # TF session & networks
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sess)
    q_net   = Q_Estimator_Nonlinear(scope="q",     lr=3e-4, sess=sess)
    tgt_net = Q_Estimator_Nonlinear(scope="target",lr=3e-4, sess=sess)
    sess.run(tf.compat.v1.global_variables_initializer())

    best_val_f1 = 0.0
    no_imp      = 0
    dyn_coef    = 10.0
    hist_r, hist_c = [], []

    for ep in range(1, num_episodes+1):
        # pick a train KPI
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

        # Build memory from RNN states
        memory = []
        for t, st in enumerate(env.states_list):
            probs = make_epsilon_greedy_policy(q_net, env.action_space_n, sess)(st, 1.0-ep/num_episodes)
            a = np.random.choice(len(probs), p=probs)
            r = env.rewardfnc(env.timeseries, t+n_steps, a)
            memory.append((st, r, st, False))

        # Train on memory
        for _ in range(num_epoches):
            batch = random.sample(memory, min(len(memory),64))
            S, R, NS, D = map(np.array, zip(*batch))
            qn = tgt_net.predict(NS)
            mx = np.max(qn, axis=1)
            targets = R + discount_factor*mx[:,None]
            q_net.update(S, targets)

        # Sync target net
        if ep % 5 == 0:
            vq = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "q")
            vt = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "target")
            for x,y in zip(vq,vt):
                sess.run(y.assign(x))

        # Logging & coef update
        reward_sum = sum(r[a] for (s,r,ns,d) in memory for a in [np.argmax(r)])
        hist_r.append(reward_sum)
        dyn_coef = update_dynamic_coef_proportional(dyn_coef, reward_sum, target=0.0, alpha=0.001)
        hist_c.append(dyn_coef)

        # Validation
        if ep % validate_every == 0:
            f1s = []
            for vid in val_idx:
                env_test.reset(to_idx=vid)
                preds, truths = [], []
                for t, st in enumerate(env_test.get_states_list()):
                    act = np.argmax(make_epsilon_greedy_policy(q_net, env_test.action_space_n, sess)(st,0))
                    preds.append(act)
                    truths.append(env_test.timeseries['anomaly'].iat[t+n_steps])
                _,_,f1,_ = precision_recall_fscore_support(truths, preds,
                                    average='binary', zero_division=0)
                f1s.append(f1)
            val_f1 = np.mean(f1s)
            print(f"[Val] ep {ep}, F1 {val_f1:.4f}")
            if val_f1 > best_val_f1:
                best_val_f1, no_imp = val_f1, 0
            else:
                no_imp += 1
                if no_imp >= patience:
                    print(f"Early stop at ep {ep}")
                    break

    # Save learning curves
    os.makedirs(os.path.join(current_dir,"exp"), exist_ok=True)
    plt.figure(); plt.plot(hist_r); plt.title("Reward"); plt.savefig("exp/reward.png")
    plt.figure(); plt.plot(hist_c); plt.title("Coef");   plt.savefig("exp/coef.png")

    print("Best Val F1:", best_val_f1)
    return hist_r, hist_c, best_val_f1

if __name__ == "__main__":
    train_csv = os.path.join(current_dir,"KPI_data","train","phase2_train.csv")
    test_csv  = os.path.join(current_dir,"KPI_data","test", "phase2_ground_truth.csv")
    train_with_validation(train_csv, test_csv,
                          num_episodes=100,
                          num_epoches=5,
                          validate_every=10,
                          patience=2,
                          num_AL=10)
