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

###############################
# Hyperparameters & Paths
###############################
EPISODES                  = 100
N_STEPS                   = 25
N_INPUT_DIM               = 2
N_HIDDEN_DIM              = 128
DISCOUNT_FACTOR           = 0.5
TN, TP, FP, FN            = 1, 10, -1, -10
ACTION_SPACE_N            = 2
VALIDATION_SEPARATE_RATIO = 0.8
MAX_WARMUP_SAMPLES        = 10000  # cap for warm-up

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
WA_DI_DIR  = os.path.join(BASE_DIR, 'WaDi')
SENSOR_CSV = os.path.join(WA_DI_DIR, 'WADI_14days_new.csv')
LABEL_CSV  = os.path.join(WA_DI_DIR, 'WADI_attackdataLABLE.csv')

#######################
# VAE Definition
#######################
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
    kl        = -0.5 * tf.reduce_sum(1 + z_log_var
                                     - tf.square(z_mean)
                                     - tf.exp(z_log_var), axis=-1)
    vae.add_loss(tf.reduce_mean(recon + kl))
    vae.compile(optimizer='adam')
    return vae

#######################
# State & Reward
#######################
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
    if include_vae and vae_model:
        win   = ts['value'].values[c-N_STEPS:c].reshape(1,-1)
        recon = vae_model.predict(win)
        err   = np.mean((recon - win)**2)
        penalty = coef * err
    lbl = ts['label'].iat[c]
    return [TN+penalty, FP+penalty] if lbl==0 else [FN+penalty, TP+penalty]

def RNNBinaryRewardFucTest(ts, c, action):
    if c < N_STEPS:
        return [0,0]
    lbl = ts['anomaly'].iat[c]
    return [TN, FP] if lbl==0 else [FN, TP]

#######################
# Q‑Network
#######################
class Q_Estimator_Nonlinear:
    def __init__(self, lr=3e-4, scope='q'):
        self.scope = scope
        with tf.compat.v1.variable_scope(scope):
            self.state  = tf.compat.v1.placeholder(tf.float32,
                            [None, N_STEPS, N_INPUT_DIM], name='state')
            self.target = tf.compat.v1.placeholder(tf.float32,
                            [None, ACTION_SPACE_N],   name='target')
            seq       = tf.compat.v1.unstack(self.state, N_STEPS, axis=1)
            cell      = tf.compat.v1.nn.rnn_cell.LSTMCell(N_HIDDEN_DIM)
            outputs,_= tf.compat.v1.nn.static_rnn(cell, seq, dtype=tf.float32)
            self.logits = layers.Dense(ACTION_SPACE_N)(outputs[-1])
            self.loss   = tf.reduce_mean(tf.square(self.logits - self.target))
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr)\
                             .minimize(self.loss)

    def predict(self, state, sess):
        return sess.run(self.logits, {self.state: state})

    def update(self, state, target, sess):
        sess.run(self.train_op, {self.state: state, self.target: target})

#######################
# Helpers
#######################
def make_epsilon_greedy_policy(est, nA, sess):
    def policy_fn(obs, eps):
        A = np.ones(nA)*eps/nA
        q = est.predict([obs], sess)[0]
        b = np.argmax(q); A[b] += (1.0-eps)
        return A
    return policy_fn

def copy_model_parameters(sess, src, dest):
    s_vars  = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                           scope=src.scope)
    d_vars  = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                           scope=dest.scope)
    for s,d in zip(sorted(s_vars, key=lambda v:v.name),
                   sorted(d_vars, key=lambda v:v.name)):
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
        dists=[]
        for s in self.env.states_list:
            q = self.est.predict([s], self.est.session)[0]
            dists.append(abs(q[0]-q[1]))
        idx = np.argsort(dists)
        return [i for i in idx if i not in self.already][:self.N]

#######################
# Q‑Learning
#######################
def q_learning(env, sess, ql, qt,
               num_episodes, num_epoches,
               update_target_every,
               eps_start, eps_end, eps_steps,
               batch_size, coef, discount):
    T = namedtuple('T',['s','r','ns','d'])
    memory = []
    sess.run(tf.compat.v1.global_variables_initializer())
    epsilons = np.linspace(eps_start, eps_end, eps_steps)
    policy   = make_epsilon_greedy_policy(ql, ACTION_SPACE_N, sess)

    # Warm‑up: cap samples
    env.reset()
    all_states = [s for s in env.states_list if s is not None]
    samples    = all_states[:MAX_WARMUP_SAMPLES]
    data_train = np.array([s[-1][0] for s in samples]).reshape(-1,1)
    if data_train.size == 0:
        raise ValueError("No warm‑up data; call env.reset()")
    _ = WarmUp().warm_up_isolation_forest(0.01, data_train)

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

#######################
# Validation
#######################
def q_learning_validator(env, sess, trained, split_idx, record_dir, plot=True):
    os.makedirs(record_dir, exist_ok=True)
    f = open(os.path.join(record_dir, 'performance.txt'), 'w')
    precision_all, recall_all, f1_all, aupr_all = [], [], [], []

    for i in range(EPISODES):
        policy = make_epsilon_greedy_policy(trained, ACTION_SPACE_N, sess)
        st = env.reset()
        # skip training portion
        while env.timeseries_curser < split_idx:
            st = env.reset()
        preds, gts, ts_vals = [], [], []

        while True:
            a = np.argmax(policy(st, 0))
            preds.append(a)
            gts.append(env.timeseries['anomaly'].iat[env.timeseries_curser])
            ts_vals.append(st[-1][0] if isinstance(st, np.ndarray) else st)
            nxt, _, done, _ = env.step(a)
            if done: break
            st = nxt[a] if (isinstance(nxt,np.ndarray) and nxt.ndim>2) else nxt

        p, r, f1, _ = precision_recall_fscore_support(gts, preds, average='binary', zero_division=0)
        aupr = average_precision_score(gts, preds)
        precision_all.append(p); recall_all.append(r)
        f1_all.append(f1); aupr_all.append(aupr)
        f.write(f"Episode {i+1}: Prec={p:.4f}, Recall={r:.4f}, F1={f1:.4f}, AU-PR={aupr:.4f}\n")

        if plot:
            fig, ax = plt.subplots(4, 1, figsize=(6,8), sharex=True)
            ax[0].plot(ts_vals);          ax[0].set_title('Time Series')
            ax[1].plot(preds,  'g-');    ax[1].set_title('Predictions')
            ax[2].plot(gts,    'r-');    ax[2].set_title('Ground Truth')
            ax[3].plot([aupr]*len(ts_vals),'m-'); ax[3].set_title('AU-PR')
            fig.tight_layout()
            fig.savefig(os.path.join(record_dir, f'val_ep_{i+1}.png'))
            plt.close(fig)

    f.close()
    return np.mean(f1_all), np.mean(aupr_all)

#######################
# Plotting
#######################
def save_plots(dir, rewards, coefs):
    os.makedirs(dir, exist_ok=True)
    plt.figure(); plt.plot(rewards); plt.title('Rewards'); plt.savefig(os.path.join(dir,'rewards.png')); plt.close()
    plt.figure(); plt.plot(coefs);   plt.title('Coefs');   plt.savefig(os.path.join(dir,'coefs.png'));   plt.close()

#######################
# Main Training Wrapper
#######################
def pretrain_vae():
    tf.compat.v1.reset_default_graph()
    from tensorflow.compat.v1.keras import backend as K
    sess = tf.compat.v1.Session(); K.set_session(sess)

    # build only normal windows
    df     = pd.read_csv(SENSOR_CSV)
    raw    = pd.read_csv(LABEL_CSV, header=1, low_memory=False)["Attack LABLE (1:No Attack, -1:Attack)"].astype(int).values
    labels = np.where(raw==1, 0, 1)
    L      = min(len(df), len(labels))
    vals   = df['TOTAL_CONS_REQUIRED_FLOW'].values[:L]
    lab    = labels[:L]
    normal = vals[lab==0]
    W      = [normal[i:i+N_STEPS] for i in range(len(normal)-N_STEPS+1)]
    X      = np.array(W, dtype=np.float32)
    scaler = StandardScaler().fit(X)
    Xs     = scaler.transform(X)

    vae = build_vae(N_STEPS)
    print("Training VAE on normal windows…")
    with sess.as_default():
        vae.fit(Xs, epochs=50, batch_size=32, verbose=1)
    vae.save('vae_wadi.h5')
    sess.close()

def main():
    pretrain_vae()

    for LP, AL in [(200,1000),(200,5000),(200,10000)]:
        print(f"\n=== RL: LP={LP}, AL={AL} ===")
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        from tensorflow.compat.v1.keras import backend as K2
        K2.set_session(sess)

        vae_model = load_model('vae_wadi.h5', custom_objects={'Sampling': Sampling}, compile=False)
        env       = EnvTimeSeriesWaDi(SENSOR_CSV, LABEL_CSV, N_STEPS)
        env.statefnc  = RNNBinaryStateFuc
        env.rewardfnc = lambda ts,tc,a: RNNBinaryRewardFuc(ts,tc,a,
                                   vae_model=vae_model,
                                   coef=20.0,
                                   include_vae=True)

        ql = Q_Estimator_Nonlinear(lr=3e-4, scope='qlearn')
        qt = Q_Estimator_Nonlinear(lr=3e-4, scope='qtarget')

        # compute split index
        total_len = len(env.timeseries_repo[0])
        split_idx = int(total_len * VALIDATION_SEPARATE_RATIO)

        exp_dir = f'exp_WaDi_LP{LP}_AL{AL}'
        os.makedirs(exp_dir, exist_ok=True)

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
            discount             = DISCOUNT_FACTOR
        )
        save_plots(exp_dir, rewards, coefs)

        val_dir = os.path.join(exp_dir, 'validation')
        f1, aupr = q_learning_validator(env, sess, ql, split_idx, val_dir, plot=True)
        print(f"→ LP={LP}, AL={AL}: Validation F1={f1:.4f}, AUPR={aupr:.4f}")

if __name__ == "__main__":
    main()
