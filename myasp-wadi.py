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
from tensorflow.keras import layers, models, losses
from tensorflow.keras.models import load_model

from env_wadi import EnvTimeSeriesWaDi

# TF1.x compatibility
tf.compat.v1.disable_eager_execution()

# Hyperparameters
EPISODES       = 3
N_STEPS        = 25
N_INPUT_DIM    = 2
N_HIDDEN_DIM   = 128
DISCOUNT       = 0.5
TN, TP, FP, FN = 1, 10, -1, -10
ACTION_SPACE_N = 2

# Paths
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
WA_DI_DIR  = os.path.join(BASE_DIR, 'WaDi')
SENSOR_CSV = os.path.join(WA_DI_DIR, 'WADI_14days_new.csv')
LABEL_CSV  = os.path.join(WA_DI_DIR, 'WADI_attackdataLABLE.csv')

# ==== VAE Definition ====
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
    h         = layers.Dense(intermediate_dim, activation='relu')(h)
    z_mean    = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
    z         = Sampling()([z_mean, z_log_var])
    dh        = layers.Dense(intermediate_dim, activation='relu')
    dm        = layers.Dense(original_dim, activation='sigmoid')
    h_dec     = dh(z)
    out       = dm(h_dec)
    vae       = models.Model(inp, out)
    recon     = losses.mse(inp, out) * original_dim
    kl        = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae.add_loss(tf.reduce_mean(recon + kl))
    vae.compile(optimizer='adam')
    return vae

# ==== State & Reward ====
def RNNBinaryStateFuc(ts, c, prev=None, action=None):
    if c == N_STEPS:
        s = [[ts['value'].iat[i], 0] for i in range(N_STEPS)]
        s.pop(0); s.append([ts['value'].iat[N_STEPS], 1])
        return np.array(s, dtype='float32')
    if c > N_STEPS:
        s0 = np.concatenate((prev[1:], [[ts['value'].iat[c], 0]]))
        s1 = np.concatenate((prev[1:], [[ts['value'].iat[c], 1]]))
        return np.array([s0, s1], dtype='float32')
    return None

def RNNBinaryRewardFuc(ts, c, action, vae_model=None, coef=1.0, include_vae=True):
    if c < N_STEPS:
        return [0,0]
    penalty = 0.0
    if include_vae and vae_model is not None:
        win   = ts['value'].values[c-N_STEPS:c].reshape(1,-1)
        recon = vae_model.predict(win)
        err   = np.mean((recon - win)**2)
        penalty = coef * err
    lbl = ts['label'].iat[c]
    return ([TN+penalty, FP+penalty] if lbl==0 else [FN+penalty, TP+penalty])

def RNNBinaryRewardFucTest(ts, c, action):
    if c < N_STEPS:
        return [0,0]
    lbl = ts['anomaly'].iat[c]
    return ([TN, FP] if lbl==0 else [FN, TP])

# ==== Q‑Network ====
class Q_Estimator_Nonlinear:
    def __init__(self, lr=3e-4, scope='q'):
        self.scope = scope
        with tf.compat.v1.variable_scope(scope):
            self.state  = tf.compat.v1.placeholder(tf.float32,
                          [None, N_STEPS, N_INPUT_DIM], name='state')
            self.target = tf.compat.v1.placeholder(tf.float32,
                          [None, ACTION_SPACE_N],   name='target')
            seq      = tf.compat.v1.unstack(self.state, N_STEPS, axis=1)
            cell     = tf.compat.v1.nn.rnn_cell.LSTMCell(N_HIDDEN_DIM)
            outputs, _ = tf.compat.v1.nn.static_rnn(cell, seq, dtype=tf.float32)
            self.logits  = layers.Dense(ACTION_SPACE_N)(outputs[-1])
            self.loss    = tf.reduce_mean(tf.square(self.logits - self.target))
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)

    def predict(self, state, sess):
        return sess.run(self.logits, {self.state: state})

    def update(self, state, target, sess):
        sess.run(self.train_op, {self.state: state, self.target: target})

# ==== Helpers ====
def make_epsilon_greedy_policy(est, nA, sess):
    def policy_fn(obs, eps):
        A = np.ones(nA)*eps/nA
        q = est.predict([obs], sess)[0]
        best = np.argmax(q); A[best] += (1.0-eps)
        return A
    return policy_fn

def copy_model_parameters(sess, src, dest):
    src_vars  = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=src.scope)
    dest_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=dest.scope)
    for s, d in zip(sorted(src_vars, key=lambda v:v.name), sorted(dest_vars, key=lambda v:v.name)):
        sess.run(d.assign(s))

class WarmUp:
    def warm_up_isolation_forest(self, frac, data):
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(contamination=frac)
        clf.fit(data)
        return clf

class active_learning:
    def __init__(self, env, N, est, already):
        self.env = env; self.N = N
        self.est = est; self.already = already

    def get_samples(self):
        dists = []
        for s in self.env.states_list:
            q = self.est.predict([s], self.est.session)[0]
            dists.append(abs(q[0]-q[1]))
        idx = np.argsort(dists)
        return [i for i in idx if i not in self.already][:self.N]

# ==== Q‑Learning ====
def q_learning(env, sess, ql, qt,
               num_episodes, num_epoches,
               update_target_every,
               eps_start, eps_end, eps_steps,
               batch_size, coef, discount):
    T      = namedtuple('T',['s','r','ns','d'])
    memory = []
    sess.run(tf.compat.v1.global_variables_initializer())
    epsilons = np.linspace(eps_start, eps_end, eps_steps)
    policy   = make_epsilon_greedy_policy(ql, ACTION_SPACE_N, sess)

    # Warm‑up
    env.reset()
    dt = np.array([s[-1][0] for s in env.states_list[:batch_size]]).reshape(-1,1)
    if dt.size == 0:
        raise ValueError("No warm‑up data; call env.reset()")
    warm = WarmUp().warm_up_isolation_forest(0.01, dt)

    rewards, coefs = [], []
    t = 0
    for ep in range(num_episodes):
        state, ep_r = env.reset(), 0
        while True:
            probs = policy(state, epsilons[min(t, eps_steps-1)])
            a     = np.random.choice(ACTION_SPACE_N, p=probs)
            nxt, r, done, _ = env.step(a)
            ep_r += r[a]
            memory.append(T(state, r, nxt, done))
            if done: break
            state = nxt[a] if (isinstance(nxt,np.ndarray) and nxt.ndim>2) else nxt
            t += 1
            if t % update_target_every == 0:
                copy_model_parameters(sess, ql, qt)

        for _ in range(num_epoches):
            batch = random.sample(memory, batch_size)
            S,R,NS,D = map(np.array, zip(*batch))
            q0 = qt.predict(NS[:,0], sess)
            q1 = qt.predict(NS[:,1], sess)
            tgt= R + discount * np.stack((q0.max(1), q1.max(1)), axis=1)
            ql.update(S, tgt, sess)

        rewards.append(ep_r)
        coefs.append(coef)

    return rewards, coefs

def q_learning_validator(env, sess, est):
    state, preds, gts = env.reset(), [], []
    policy = make_epsilon_greedy_policy(est, ACTION_SPACE_N, sess)
    while True:
        a = np.argmax(policy(state, 0))
        preds.append(a)
        gts.append(env.timeseries['anomaly'].iat[env.timeseries_curser])
        nxt,_,done,_ = env.step(a)
        if done: break
        state = nxt[a] if (isinstance(nxt,np.ndarray) and nxt.ndim>2) else nxt

    p,r,f,_ = precision_recall_fscore_support(gts, preds, average='binary')
    aupr    = average_precision_score(gts, preds)
    return f, aupr

def save_plots(exp_dir, rewards, coefs):
    os.makedirs(exp_dir, exist_ok=True)
    plt.figure(); plt.plot(rewards); plt.title('Rewards'); plt.savefig(os.path.join(exp_dir,'rewards.png')); plt.close()
    plt.figure(); plt.plot(coefs);   plt.title('Coefs');   plt.savefig(os.path.join(exp_dir,'coefs.png'));   plt.close()

# ==== MAIN ====
def pretrain_vae():
    """Train VAE once, show progress, save model."""
    tf.compat.v1.reset_default_graph()
    from tensorflow.compat.v1.keras import backend as K
    sess_vae = tf.compat.v1.Session()
    K.set_session(sess_vae)

    df   = pd.read_csv(SENSOR_CSV)
    vals = df['TOTAL_CONS_REQUIRED_FLOW'].values
    X    = np.array([vals[i:i+N_STEPS] for i in range(len(vals)-N_STEPS)])
    scaler= StandardScaler().fit(X)
    Xs   = scaler.transform(X)

    vae = build_vae(N_STEPS)
    print("Training VAE for 50 epochs…")
    with sess_vae.as_default():
        vae.fit(Xs, epochs=3, batch_size=32, verbose=1)
    vae.save('vae_wadi.h5')
    sess_vae.close()
    print("VAE training complete, model saved to vae_wadi.h5\n")

def main():
    pretrain_vae()
    for LP, AL in [(200,1000),(200,5000),(200,10000)]:
        print(f"=== START RL: LP={LP}, AL={AL} ===")
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        from tensorflow.compat.v1.keras import backend as K2
        K2.set_session(sess)

        vae_model = load_model('vae_wadi.h5', custom_objects={'Sampling': Sampling}, compile=False)
        env       = EnvTimeSeriesWaDi(SENSOR_CSV, LABEL_CSV, N_STEPS)
        env.statefnc  = RNNBinaryStateFuc
        env.rewardfnc = lambda ts,tc,a: RNNBinaryRewardFuc(ts,tc,a,vae_model=vae_model,coef=20.0,include_vae=True)

        ql = Q_Estimator_Nonlinear(lr=3e-4, scope='qlearn')
        qt = Q_Estimator_Nonlinear(lr=3e-4, scope='qtarget')

        rewards, coefs = q_learning(
            env, sess, ql, qt,
            num_episodes        = EPISODES,
            num_epoches         = 10,
            update_target_every = 1000,
            eps_start           = 1.0,
            eps_end             = 0.1,
            eps_steps           = 50000,
            batch_size          = 256,
            coef                = 20.0,
            discount             = DISCOUNT
        )
        f1, aupr = q_learning_validator(env, sess, ql)
        exp_dir = f'exp_WaDi_LP{LP}_AL{AL}_d{DISCOUNT}'
        save_plots(exp_dir, rewards, coefs)
        print(f"→ LP={LP}, AL={AL}, d={DISCOUNT} → F1={f1:.4f}, AUPR={aupr:.4f}\n")

if __name__ == "__main__":
    main()
