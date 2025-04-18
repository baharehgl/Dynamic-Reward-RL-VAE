# train_wadi.py
import os
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import namedtuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from tensorflow.keras import layers, models, losses
from tensorflow.keras.models import load_model
from env_wadi import EnvTimeSeriesWaDi

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Hyperparameters
episodes = 100
n_steps = 25
discount = 0.5
TN, TP, FP, FN = 1, 10, -1, -10

action_space_n = 2

# ==== VAE Definition ====
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        eps = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


def build_vae(original_dim, latent_dim=10, intermediate_dim=64):
    inputs = layers.Input(shape=(original_dim,))
    h = layers.Dense(intermediate_dim, activation='relu')(inputs)
    h = layers.Dense(intermediate_dim, activation='relu')(h)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
    z = Sampling()([z_mean, z_log_var])
    decoder_h = layers.Dense(intermediate_dim, activation='relu')
    decoder_mean = layers.Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    outputs = decoder_mean(h_decoded)
    vae = models.Model(inputs, outputs)
    recon_loss = losses.mse(inputs, outputs) * original_dim
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae.add_loss(tf.reduce_mean(recon_loss + kl_loss))
    vae.compile(optimizer='adam')
    return vae

# ==== State & Reward Functions ====
def RNNBinaryStateFuc(ts, cursor, prev_state=None, action=None):
    if cursor == n_steps:
        state = [[ts['value'].iat[i], 0] for i in range(n_steps)]
        state.pop(0)
        state.append([ts['value'].iat[n_steps], 1])
        return np.array(state, dtype='float32')
    if cursor > n_steps:
        s0 = np.concatenate((prev_state[1:], [[ts['value'].iat[cursor], 0]]))
        s1 = np.concatenate((prev_state[1:], [[ts['value'].iat[cursor], 1]]))
        return np.array([s0, s1], dtype='float32')
    return None

def RNNBinaryRewardFuc(ts, cursor, action, vae_model=None, dynamic_coef=1.0, include_vae_penalty=True):
    if cursor < n_steps:
        return [0,0]
    penalty = 0.0
    if include_vae_penalty and vae_model is not None:
        window = ts['value'].values[cursor-n_steps:cursor].reshape(1,-1)
        recon = vae_model.predict(window)
        err = np.mean((recon - window)**2)
        penalty = dynamic_coef * err
    lbl = ts['label'].iat[cursor]
    return [TN+penalty, FP+penalty] if lbl==0 else [FN+penalty, TP+penalty]

def RNNBinaryRewardFucTest(ts, cursor, action):
    if cursor < n_steps:
        return [0,0]
    lbl = ts['anomaly'].iat[cursor]
    return [TN, FP] if lbl==0 else [FN, TP]

# ==== Q-Network Estimator ====
class Q_Estimator_Nonlinear:
    def __init__(self, learning_rate=0.0003, scope='qlearn'):
        self.scope = scope
        with tf.compat.v1.variable_scope(scope):
            self.state = tf.compat.v1.placeholder(tf.float32, [None, n_steps, 2], name='state')
            self.target = tf.compat.v1.placeholder(tf.float32, [None, action_space_n], name='target')
            unstack = tf.compat.v1.unstack(self.state, n_steps, axis=1)
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell(n_hidden_dim)
            outputs,_ = tf.compat.v1.nn.static_rnn(cell, unstack, dtype=tf.float32)
            self.logits = layers.Dense(action_space_n)(outputs[-1])
            self.loss = tf.reduce_mean(tf.square(self.logits - self.target))
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(self.loss)
    def predict(self, state, sess):
        return sess.run(self.logits, feed_dict={self.state: state})
    def update(self, state, target, sess):
        sess.run(self.optimizer, feed_dict={self.state: state, self.target: target})

def make_epsilon_greedy_policy(estimator, nA, sess):
    def policy_fn(obs, eps):
        A = np.ones(nA) * eps / nA
        q = estimator.predict([obs], sess)[0]
        best = np.argmax(q)
        A[best] += (1.0 - eps)
        return A
    return policy_fn

def copy_model_parameters(sess, src, dest):
    src_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=src.scope)
    dest_vars= tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=dest.scope)
    for s, d in zip(sorted(src_vars, key=lambda v: v.name), sorted(dest_vars, key=lambda v: v.name)):
        sess.run(d.assign(s))

# ==== Active Learning & WarmUp ====
class active_learning:
    def __init__(self, env, N, estimator, already_selected):
        self.env = env; self.N = N; self.estimator = estimator; self.already = already_selected
    def get_samples(self):
        distances=[]
        for s in self.env.states_list:
            q = self.estimator.predict([s], self.estimator.session)[0]
            distances.append(abs(q[0]-q[1]))
        idx = np.argsort(distances)
        return [i for i in idx if i not in self.already][:self.N]

class WarmUp:
    def warm_up_isolation_forest(self, frac, data):
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(contamination=frac)
        clf.fit(data)
        return clf

# ==== Q-Learning & Validation ====
def q_learning(env, sess, q_learn, q_target, num_episodes, num_epoches,
               update_target_every, epsilon_start, epsilon_end, epsilon_steps,
               batch_size, dynamic_coef):
    Transition = namedtuple('Transition',['state','reward','next','done'])
    memory=[]
    sess.run(tf.compat.v1.global_variables_initializer())
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_steps)
    policy = make_epsilon_greedy_policy(q_learn, action_space_n, sess)
    replay_init=5000
    # Warm-up sequence
    state = env.reset()
    states_list = env.states_list
    data_train = np.array([s[-1][0] for s in states_list[:replay_init]]).reshape(-1,1)
    if len(data_train)==0:
        raise ValueError('No warm-up data. Call env.reset().')
    warm_clf = WarmUp().warm_up_isolation_forest(0.01, data_train)
    # Main loop
    rewards=[]; coefs=[]; t=0
    for ep in range(num_episodes):
        state = env.reset(); ep_reward=0
        while True:
            action_probs = policy(state, epsilons[min(t,epsilon_steps-1)])
            action = np.random.choice(action_space_n, p=action_probs)
            next_state,reward,done,_ = env.step(action)
            ep_reward+=reward[action]
            memory.append(Transition(state,reward,next_state,done))
            if done: break
            state = next_state[action] if isinstance(next_state,np.ndarray) else next_state
            t+=1
            if t%update_target_every==0:
                copy_model_parameters(sess, q_learn, q_target)
        # training
        for _ in range(num_epoches):
            batch = random.sample(memory, batch_size)
            S,R,NS,D = map(np.array, zip(*batch))
            q0 = q_target.predict(NS[:,0],sess); q1 = q_target.predict(NS[:,1],sess)
            targets = R + discount * np.stack((q0.max(1),q1.max(1)),1)
            q_learn.update(S,targets,sess)
        rewards.append(ep_reward); coefs.append(dynamic_coef)
    return rewards, coefs

def q_learning_validator(env, sess, trained_estimator):
    # simple validation: one episode
    state = env.reset(); preds=[]; gts=[]
    policy = make_epsilon_greedy_policy(trained_estimator, action_space_n, sess)
    while True:
        action = np.argmax(policy(state,0))
        preds.append(action)
        gt = env.timeseries['anomaly'].iat[env.timeseries_curser]
        gts.append(gt)
        state,_,done,_ = env.step(action)
        if done: break
    precision,recall,f1,_ = precision_recall_fscore_support(gts,preds,average='binary')
    aupr = average_precision_score(gts,preds)
    return f1, aupr

def save_plots(exp_dir, rewards, coefs):
    os.makedirs(exp_dir, exist_ok=True)
    plt.figure(); plt.plot(rewards); plt.title('Rewards'); plt.savefig(os.path.join(exp_dir,'rewards.png')); plt.close()
    plt.figure(); plt.plot(coefs); plt.title('Coefs'); plt.savefig(os.path.join(exp_dir,'coefs.png')); plt.close()

# ==== Training Wrapper ====
def train_wrapper(num_LP, num_AL, discount_factor):
    # Train VAE
    tf.compat.v1.reset_default_graph()
    from tensorflow.compat.v1.keras import backend as K
    sess_vae = tf.compat.v1.Session(); K.set_session(sess_vae)
    df = pd.read_csv('WaDi/WADI_14days_new.csv')
    vals = df['TOTAL_CONS_REQUIRED_FLOW'].values
    X = np.array([vals[i:i+n_steps] for i in range(len(vals)-n_steps)])
    scaler = StandardScaler().fit(X); Xs=scaler.transform(X)
    vae = build_vae(n_steps)
    with sess_vae.as_default(): vae.fit(Xs,epochs=30,batch_size=32)
    vae.save('vae_wadi.h5'); sess_vae.close()

    # RL training
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session(); K.set_session(sess)
    vae_model = load_model('vae_wadi.h5', custom_objects={'Sampling': Sampling}, compile=False)
    env = EnvTimeSeriesWaDi('WaDi/WADI_14days_new.csv','WaDi/WADI_attackdataLABLE.csv',n_steps)
    q_learn = Q_Estimator_Nonlinear(scope='qlearn')
    q_target = Q_Estimator_Nonlinear(scope='qtarget')
    rewards, coefs = q_learning(env, sess, q_learn, q_target,
                                num_episodes=episodes,
                                num_epoches=10,
                                update_target_every=1000,
                                epsilon_start=1.0,
                                epsilon_end=0.1,
                                epsilon_steps=50000,
                                batch_size=256,
                                dynamic_coef=10.0)
    f1,aupr = q_learning_validator(env, sess, q_learn)
    exp_dir = f'exp_WaDi_LP{num_LP}_AL{num_AL}_d{discount_factor}'
    save_plots(exp_dir, rewards, coefs)
    print(f"LP={num_LP}, AL={num_AL}, d={discount_factor} → F1={f1:.4f}, AUPR={aupr:.4f}")

if __name__ == '__main__':
    train_wrapper(200, 1000, 0.96)
    train_wrapper(200, 5000, 0.96)
    train_wrapper(200,10000,0.96)