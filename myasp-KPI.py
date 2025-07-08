import os, sys, time, random, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# Disable eager for tf.placeholder()
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from tensorflow.keras import layers, models, losses
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.semi_supervised import LabelSpreading

# import our KPI env
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from env_KPI import EnvKPI

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

# Hyperparams
EPISODES                  = 3
n_steps                   = 25
n_input_dim               = 2
n_hidden_dim              = 128
validation_separate_ratio = 0.9

TN_Value = 1; TP_Value = 5; FP_Value = -1; FN_Value = -5
NOT_ANOMALY = 0; ANOMALY = 1
action_space = [NOT_ANOMALY, ANOMALY]

# --- VAE setup ---
def load_normal_data(data_path, n_steps):
    windows = []
    for fname in os.listdir(data_path):
        if not fname.endswith('.csv'): continue
        vals = pd.read_csv(os.path.join(data_path, fname))['value'].values
        if len(vals) < n_steps: continue
        for i in range(len(vals)-n_steps+1):
            windows.append(vals[i:i+n_steps])
    arr = np.array(windows)
    return StandardScaler().fit_transform(arr)

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]; dim = tf.shape(z_mean)[1]
        eps   = tf.keras.backend.random_normal((batch,dim))
        return z_mean + tf.exp(0.5*z_log_var)*eps

def build_vae(original_dim, latent_dim=2, intermediate_dim=64):
    x_in = layers.Input((original_dim,))
    h    = layers.Dense(intermediate_dim, activation='relu')(x_in)
    h    = layers.Dense(intermediate_dim, activation='relu')(h)
    z_mean   = layers.Dense(latent_dim)(h)
    z_log_var= layers.Dense(latent_dim)(h)
    z_log_var= tf.clip_by_value(z_log_var, -10.0, 10.0)
    z        = Sampling()([z_mean, z_log_var])
    dec_h    = layers.Dense(intermediate_dim, activation='relu')
    h_dec    = dec_h(z)
    x_dec    = layers.Dense(original_dim, activation='sigmoid')(h_dec)
    vae = models.Model(x_in, x_dec)
    recon = losses.mse(x_in, x_dec) * original_dim
    kl    = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae.add_loss(tf.reduce_mean(recon + kl))
    vae.compile(optimizer='adam')
    return vae

original_dim, latent_dim, intermediate_dim = n_steps, 10, 64

# --- State & Reward ---
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
    if cursor < n_steps: return [0,0]
    cur = np.array([ts['value'].iloc[cursor-n_steps:cursor]])
    recon = vae_model.predict(cur)
    err   = np.mean((recon-cur)**2)
    pen   = dynamic_coef*err
    lbl   = ts['label'].iat[cursor]
    if lbl==0: return [TN_Value+pen, FP_Value+pen]
    if lbl==1: return [FN_Value+pen, TP_Value+pen]
    return [0,0]

def RNNBinaryRewardFucTest(ts, cursor, action=0):
    if cursor < n_steps: return [0,0]
    an = ts['anomaly'].iat[cursor]
    return [TN_Value, FP_Value] if an==0 else [FN_Value, TP_Value]

# --- Q-estimator ---
class Q_Estimator_Nonlinear:
    def __init__(self, learning_rate=0.01, scope="Q", summaries_dir=None):
        self.scope = scope
        with tf.compat.v1.variable_scope(scope):
            self.state  = tf.compat.v1.placeholder(
                tf.float32, [None, n_steps, n_input_dim], name="state")
            self.target = tf.compat.v1.placeholder(
                tf.float32, [None, len(action_space)], name="target")
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
        A = np.ones(nA, dtype='float32')*(eps/nA)
        q = estimator.predict([obs], sess=sess)[0]
        A[np.argmax(q)] += (1.0 - eps)
        return A
    return policy_fn

def update_dynamic_coef_proportional(cur, rew, target=100.0,
                                     alpha=0.01, min_coef=0.1, max_coef=10.0):
    nc = cur + alpha*(target - rew)
    return max(min(nc, max_coef), min_coef)

# --- Active learning & WarmUp ---
class active_learning:
    def __init__(self, env, N, estimator, already_selected):
        self.env = env; self.N = N; self.estimator = estimator; self.sel = already_selected
    def get_samples(self):
        dists = []
        for s in self.env.states_list:
            q = self.estimator.predict([s])[0]
            dists.append(abs(q[0]-q[1]))
        order = np.argsort(dists)
        return [i for i in order if i not in self.sel][:self.N]

class WarmUp:
    def warm_up_isolation_forest(self, outliers_fraction, X_train):
        from sklearn.ensemble import IsolationForest
        data = np.array(X_train)[:,-1].reshape(-1,1)
        clf  = IsolationForest(contamination=outliers_fraction)
        clf.fit(data)
        return clf

# --- Q-learning ---
def q_learning(env, sess, q_est, tgt_est, num_episodes, num_epoches,
               replay_memory_size=500000, replay_memory_init_size=50000,
               experiment_dir='./log/', update_target_every=10000,
               discount_factor=0.99, epsilon_start=1.0, epsilon_end=0.1,
               epsilon_decay_steps=500000, batch_size=256,
               num_LabelPropagation=20, num_active_learning=5,
               test=0, vae_model=None):

    from collections import namedtuple
    Transition = namedtuple("Transition", ["state","reward","next_state","done"])
    replay_memory = []
    # create checkpoints dir
    os.makedirs(os.path.join(experiment_dir,"checkpoints"), exist_ok=True)
    saver = tf.compat.v1.train.Saver()

    total_t  = 0
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    policy   = make_epsilon_greedy_policy(q_est, env.action_space_n, sess)

    # Warm-up isolation forest
    data_train = []
    for _ in range(env.datasetsize):
        env.reset()
        env.states_list = env.get_states_list()
        data_train.extend(env.states_list)
    iso = WarmUp().warm_up_isolation_forest(0.01, data_train)
    lp_model = LabelSpreading()

    # initial pseudo-labeling until memory filled
    for t in itertools.count():
        env.reset()
        env.states_list = env.get_states_list()
        data = np.array(env.states_list)[:,-1].reshape(-1,1)
        scores = iso.decision_function(data)
        preds  = [-s+0.5 for s in scores]
        warm   = np.argsort(preds)[:5].tolist() + np.argsort(preds)[-5:].tolist()
        labeled_index = [i-n_steps for i in range(n_steps,len(env.timeseries['label']))
                         if env.timeseries['label'].iat[i]!=-1]
        for idx in warm:
            if idx < len(env.states_list):
                env.timeseries_cursor = idx+n_steps
                action_probs = policy(env.states_list[idx], epsilons[min(total_t, epsilon_decay_steps-1)])
                act = np.random.choice(range(len(action_probs)), p=action_probs)
                env.timeseries['label'].iat[env.timeseries_cursor] = env.timeseries['anomaly'].iat[env.timeseries_cursor]
                replay_memory.append(Transition(env.states_list[idx], [0,0], env.timeseries_states, False))
        if len(replay_memory) >= replay_memory_init_size:
            break

    dynamic_coef=10.0
    episode_rewards=[]
    coef_history=[]

    for ep in range(num_episodes):
        env.rewardfnc = lambda ts,tc,a: RNNBinaryRewardFuc(ts,tc,a,vae_model,dynamic_coef)
        episode_reward=0.0
        state = env.reset()
        env.states_list = env.get_states_list()
        labeled_index = [i-n_steps for i in range(n_steps,len(env.timeseries['label']))
                         if env.timeseries['label'].iat[i]!=-1]

        # active learning
        al = active_learning(env, num_active_learning, q_est, labeled_index)
        al_samples = al.get_samples()
        for idx in al_samples:
            tpos = idx+n_steps
            env.timeseries['label'].iat[tpos] = env.timeseries['anomaly'].iat[tpos]
        labeled_index += al_samples

        # generate transitions
        for idx in labeled_index:
            env.timeseries_cursor = idx+n_steps
            probs = policy(env.states_list[idx], epsilons[min(total_t, epsilon_decay_steps-1)])
            act   = np.random.choice(range(len(probs)), p=probs)
            nxt, r, done, _ = env.step(act)
            episode_reward += r[act]
            replay_memory.append(Transition(env.states_list[idx], r, nxt, done))
            if len(replay_memory)>replay_memory_size:
                replay_memory.pop(0)

        # training
        for _ in range(num_epoches):
            batch = random.sample(replay_memory, min(batch_size,len(replay_memory)))
            states, rewards, next_states, dones = map(np.array, zip(*batch))
            # compute targets
            ns0, ns1 = np.split(next_states,2,axis=1)
            ns0 = ns0.reshape(-1,n_steps,n_input_dim)
            ns1 = ns1.reshape(-1,n_steps,n_input_dim)
            q0 = tgt_est.predict(ns0, sess); q1 = tgt_est.predict(ns1, sess)
            max0 = np.max(q0,axis=1); max1 = np.max(q1,axis=1)
            targets = rewards + discount_factor*np.stack((max0,max1),axis=-1)
            q_est.update(states, targets, sess)
            total_t += 1
            if total_t % update_target_every == 0:
                # copy weights
                vars_q   = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,"Q")
                vars_tgt = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,"target")
                for vq, vt in zip(vars_q, vars_tgt):
                    sess.run(vt.assign(vq))

        dynamic_coef = update_dynamic_coef_proportional(dynamic_coef, episode_reward,
                                                        target=0.0, alpha=0.001)
        episode_rewards.append(episode_reward)
        coef_history.append(dynamic_coef)
        print(f"Ep {ep}: rew={episode_reward:.2f}, coef={dynamic_coef:.2f}")

    return episode_rewards, coef_history

# --- Validation ---
def q_learning_validator(env, estimator, num_episodes, record_dir=None, plot=1):
    from sklearn.metrics import precision_recall_fscore_support
    precision_all, recall_all, f1_all = [], [], []
    for ep in range(num_episodes):
        state = env.reset()
        env.states_list = env.get_states_list()
        preds, truths, ts_vals = [], [], []
        policy = make_epsilon_greedy_policy(estimator, env.action_space_n,
                                            tf.compat.v1.get_default_session())
        while True:
            action = np.argmax(policy(state, 0))
            preds.append(action)
            truths.append(env.timeseries['anomaly'].iat[env.timeseries_cursor])
            ts_vals.append(state[-1][0])
            nxt, _, done, _ = env.step(action)
            if done: break
            state = nxt[action]
        p, r, f, _ = precision_recall_fscore_support(
            truths, preds, average='binary', zero_division=0)
        precision_all.append(p); recall_all.append(r); f1_all.append(f)
    return np.mean(f1_all)

# --- Plotting ---
def save_plots(experiment_dir, episode_rewards, coef_history):
    os.makedirs(os.path.join(experiment_dir,"plots"), exist_ok=True)
    plt.figure(); plt.plot(episode_rewards); plt.title("Rewards");
    plt.savefig(os.path.join(experiment_dir,"plots","rewards.png")); plt.close()
    plt.figure(); plt.plot(coef_history); plt.title("Coefs")
    plt.savefig(os.path.join(experiment_dir,"plots","coefs.png")); plt.close()

# --- Train wrapper ---
def train_wrapper(num_LP, num_AL, discount_factor):
    # Train VAE
    x_train = load_normal_data(os.path.join(current_dir,"normal-data"), n_steps)
    vae_model = build_vae(original_dim, latent_dim, intermediate_dim)
    vae_model.fit(x_train, epochs=2, batch_size=32)
    vae_model.save('vae_model.h5')

    # Build envs
    train_csv = os.path.join(current_dir,"KPI_data","train","phase2_train.csv")
    test_csv  = os.path.join(current_dir,"KPI_data","test","phase2_ground_truth.csv")
    env       = EnvKPI(train_csv, test_csv)
    env.statefnc  = RNNBinaryStateFuc
    env.rewardfnc = lambda ts,tc,a: RNNBinaryRewardFuc(ts,tc,a,vae_model, dynamic_coef=10.0)
    env_test = EnvKPI(train_csv, test_csv)
    env_test.statefnc  = RNNBinaryStateFuc
    env_test.rewardfnc = RNNBinaryRewardFucTest

    # Start TF
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sess)

    q_est   = Q_Estimator_Nonlinear(learning_rate=3e-4, scope="Q")
    tgt_est = Q_Estimator_Nonlinear(learning_rate=3e-4, scope="target")
    sess.run(tf.compat.v1.global_variables_initializer())

    # Train & validate
    ep_rewards, coef_hist = q_learning(
        env, sess, q_est, tgt_est,
        num_episodes=EPISODES, num_epoches=10,
        discount_factor=discount_factor,
        num_LabelPropagation=num_LP,
        num_active_learning=num_AL,
        test=0, vae_model=vae_model
    )

    avg_f1 = q_learning_validator(
        env_test, q_est,
        num_episodes=int(env.datasetsize*(1-validation_separate_ratio)),
        plot=1
    )

    save_plots("./exp_kpi", ep_rewards, coef_hist)
    print("Avg F1 on KPI test set:", avg_f1)
    return avg_f1

if __name__ == "__main__":
    train_wrapper(200, 1000, 0.96)
    train_wrapper(200, 5000, 0.96)
    train_wrapper(200,10000,0.96)
