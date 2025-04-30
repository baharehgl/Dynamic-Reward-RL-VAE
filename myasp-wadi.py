import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

# TF1 compatibility
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import layers, models, losses
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from collections import namedtuple

from env_wadi import EnvTimeSeriesWaDi

# ─── HYPERPARAMETERS & PATHS ─────────────────────────────────────────────────────
EPISODES                  = 2
N_STEPS                   = 25
DISCOUNT_FACTOR           = 0.5
TN, TP, FP, FN            = 1, 10, -1, -10
ACTION_SPACE_N            = 2
VALIDATION_SPLIT          = 0.8
MAX_WARMUP_SAMPLES        = 10000
NUM_LP                    = 200  # Label-Propagation
BATCH_SIZE                = 128
VAE_EPOCHS                = 2
VAE_BATCH                 = 32

BASE      = os.path.dirname(os.path.abspath(__file__))
WA_DI_DIR = os.path.join(BASE, 'WaDi')
SENSOR_CSV= os.path.join(WA_DI_DIR,'WADI_14days_new.csv')
LABEL_CSV = os.path.join(WA_DI_DIR,'WADI_attackdataLABLE.csv')

# ─── FEATURES ────────────────────────────────────────────────────────────────────
_sensor_df   = pd.read_csv(SENSOR_CSV, nrows=1)
feature_cols = list(_sensor_df.columns)
n_features   = len(feature_cols)
N_INPUT_DIM  = n_features + 1  # +1 for action bit

# ─── VAE ─────────────────────────────────────────────────────────────────────────
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        eps = tf.random.normal(tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * eps

def build_vae(input_dim, latent_dim=10, hid=64):
    inp = layers.Input(shape=(input_dim,))
    h   = layers.Dense(hid, activation='relu')(inp)
    z_mean    = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    z_log_var = tf.clip_by_value(z_log_var, -10., 10.)
    z         = Sampling()([z_mean, z_log_var])
    h2        = layers.Dense(hid, activation='relu')(z)
    out       = layers.Dense(input_dim, activation='sigmoid')(h2)

    vae       = models.Model(inp, out)
    recon = losses.mse(inp, out) * input_dim
    kl    = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae.add_loss(tf.reduce_mean(recon + kl))
    vae.compile(optimizer='adam', loss=None)
    return vae

# ─── STATE & REWARD ──────────────────────────────────────────────────────────────
def RNNState(ts, c, prev=None, action=None):
    if c < N_STEPS: return None
    W = ts[feature_cols].iloc[c-N_STEPS+1:c+1].values.astype('float32')
    a0 = np.concatenate([W, np.zeros((N_STEPS,1),dtype='float32')], axis=1)
    a1 = np.concatenate([W, np.ones ((N_STEPS,1),dtype='float32')], axis=1)
    return np.stack([a0,a1], axis=0)

def RNNReward(ts, c, a, vae=None, coef=1.0):
    if c < N_STEPS: return [0,0]
    # compute VAE penalty
    win = ts[feature_cols].iloc[c-N_STEPS+1:c+1].values.flatten().reshape(1,-1).astype('float32')
    pen = coef * np.mean((vae.predict(win) - win)**2) if vae else 0.0
    lbl = ts['label'].iat[c]
    base = [TN,FP] if lbl==0 else [FN,TP]
    return [base[0]+pen, base[1]+pen]

def RNNRewardTest(ts, c, a):
    if c < N_STEPS: return [0,0]
    lbl = ts['anomaly'].iat[c]
    return [TN,FP] if lbl==0 else [FN,TP]

# ─── Q-NETWORK ─────────────────────────────────────────────────────────────────
class QNet:
    def __init__(self, scope='q', lr=3e-4):
        with tf.compat.v1.variable_scope(scope):
            self.S = tf.compat.v1.placeholder(tf.float32, [None, N_STEPS, N_INPUT_DIM], 'S')
            self.T = tf.compat.v1.placeholder(tf.float32, [None, ACTION_SPACE_N],    'T')
            seq = tf.compat.v1.unstack(self.S, N_STEPS, axis=1)
            cell= tf.compat.v1.nn.rnn_cell.LSTMCell(128)
            out,_ = tf.compat.v1.nn.static_rnn(cell, seq, dtype=tf.float32)
            self.logits   = layers.Dense(ACTION_SPACE_N)(out[-1])
            self.loss     = tf.reduce_mean(tf.square(self.logits - self.T))
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)

    def predict(self, s, sess):
        return sess.run(self.logits, {self.S: s})
    def update (self, s,t,sess):
        sess.run(self.train_op, {self.S: s, self.T: t})

def epsilon_policy(est,nA,sess):
    def f(obs,eps):
        A = np.ones(nA)*eps/nA
        q = est.predict([obs],sess)[0]
        b = np.argmax(q); A[b]+=1-eps
        return A
    return f

def copy_params(sess,src,dst):
    vs = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=src.scope)
    vt = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=dst.scope)
    for v_s,v_t in zip(sorted(vs,key=lambda v:v.name), sorted(vt,key=lambda v:v.name)):
        sess.run(v_t.assign(v_s))

# ─── DYNAMIC COEF UPDATE ────────────────────────────────────────────────────────
def update_coef(old, ep_reward, target=0.0, alpha=0.001, lo=0.1, hi=100.0):
    new = old + alpha*(ep_reward - target)
    return max(min(new,hi), lo)

# ─── Q-LEARNING WITH DYNAMIC REWARD ─────────────────────────────────────────────
def q_learning(env, sess, ql, qt,
               episodes, epochs,
               tgt_every,
               eps_start, eps_end, eps_steps,
               bsize, discount,
               nLP, nAL,
               vae_model,
               init_coef):

    Transition = namedtuple('T',['s','r','ns','d'])
    mem = []
    sess.run(tf.compat.v1.global_variables_initializer())
    epss = np.linspace(eps_start, eps_end, eps_steps)
    pol  = epsilon_policy(ql, ACTION_SPACE_N, sess)

    # warm-up via IsolationForest
    W=[]
    env.reset()
    for i in range(N_STEPS,len(env.timeseries)):
        if env.timeseries['label'].iat[i]==0:
            w = env.timeseries[feature_cols].iloc[i-N_STEPS+1:i+1].values.flatten()
            W.append(w)
    if W:
        IsolationForest(contamination=0.01).fit(np.array(W[:MAX_WARMUP_SAMPLES],dtype='float32'))

    dynamic_coef = init_coef
    coef_history = []

    for ep in range(episodes):
        # rebind reward with current coef
        env.rewardfnc = lambda ts,tc,a,coef=dynamic_coef: RNNReward(ts,tc,a,vae_model,coef)

        # Label-Propagation
        labs = np.array(env.timeseries['label'].iloc[N_STEPS:])
        if np.any(labs!=-1):
            arr = np.array([s for s in env.states_list if s is not None])
            flat= arr.reshape(arr.shape[0],-1)
            lp  = LabelSpreading(kernel='knn',n_neighbors=10).fit(flat,labs)
            uncert = np.argsort(-np.max(lp.label_distributions_,axis=1))[:nLP]
            for u in uncert:
                env.timeseries['label'].iat[u+N_STEPS] = lp.transduction_[u]
        # Active-Learning
        if 'uncert' in locals():
            for u in uncert[:nAL]:
                env.timeseries['label'].iat[u+N_STEPS] = env.timeseries['anomaly'].iat[u+N_STEPS]

        # one episode rollout
        state, done = env.reset(), False
        ep_reward = 0.0
        while not done:
            eps  = epss[min(ep,len(epss)-1)]
            probs= pol(state, eps)
            a    = np.random.choice(ACTION_SPACE_N, p=probs)
            raw, r, done, _ = env.step(a)
            next_state = raw[a] if raw.ndim>2 else raw
            mem.append(Transition(state, r, next_state, done))
            ep_reward += r[a]
            state = next_state

        # train on minibatches
        for _ in range(epochs):
            batch = random.sample(mem, bsize)
            S,R,NS,D = map(np.array,zip(*batch))
            q0 = qt.predict(NS[:,0],sess)
            q1 = qt.predict(NS[:,1],sess)
            tgt = R + discount * np.stack((q0.max(1),q1.max(1)),axis=1)
            ql.update(S, tgt, sess)

        # update target net
        if ep % tgt_every == 0:
            copy_params(sess, ql, qt)

        # adjust dynamic_coef
        dynamic_coef = update_coef(dynamic_coef, ep_reward)
        coef_history.append(dynamic_coef)
        print(f"[Ep {ep+1}] reward={ep_reward:.2f}, next_coef={dynamic_coef:.3f}")

    return coef_history

# ─── VALIDATION ────────────────────────────────────────────────────────────────
def validate(env,sess,trained,split,rec_dir):
    os.makedirs(rec_dir,exist_ok=True)
    f1s,auprs = [],[]
    with open(os.path.join(rec_dir,'perf.txt'),'w') as f:
        for i in range(EPISODES):
            pol = epsilon_policy(trained,ACTION_SPACE_N,sess)
            state, done = env.reset(), False
            # skip train part
            while env.timeseries_curser < split:
                raw,_,done,_ = env.step(0)
                state = raw[0] if raw.ndim>2 else raw
                if done: break

            preds, gts, vals = [],[],[]
            while not done:
                a = np.argmax(pol(state,0.0))
                preds.append(a)
                gts.append(env.timeseries['anomaly'].iat[env.timeseries_curser])
                vals.append(state[-1][0])
                raw,_,done,_ = env.step(a)
                state = raw[a] if raw.ndim>2 else raw

            p,r,f1,_ = precision_recall_fscore_support(gts,preds,average='binary',zero_division=0)
            aupr     = average_precision_score(gts,preds)
            f.write(f"E{i+1}:P={p:.3f},R={r:.3f},F1={f1:.3f},AUPR={aupr:.3f}\n")
            f1s.append(f1); auprs.append(aupr)

            # plot each
            fig,ax=plt.subplots(4,1,sharex=True)
            ax[0].plot(vals); ax[0].set_title('Value')
            ax[1].plot(preds,'g-'); ax[1].set_title('Pred')
            ax[2].plot(gts,'r-'); ax[2].set_title('True')
            ax[3].plot([aupr]*len(vals)); ax[3].set_title('AUPR')
            fig.savefig(os.path.join(rec_dir,f'v{i+1}.png')); plt.close(fig)

    return np.mean(f1s), np.mean(auprs)

# ─── TRAIN WRAPPER ──────────────────────────────────────────────────────────────
def train_wrapper(nLP, nAL, discount):
    # 1) Pretrain VAE
    df   = pd.read_csv(SENSOR_CSV)
    raw  = pd.read_csv(LABEL_CSV,header=1)["Attack LABLE (1:No Attack, -1:Attack)"].astype(int).values
    labels = np.where(raw==1,0,1)
    W=[]
    for i in range(N_STEPS,len(df)):
        if labels[i]==0:
            W.append(df[feature_cols].iloc[i-N_STEPS+1:i+1].values.flatten())
    X    = np.array(W,dtype='float32')
    Xs   = StandardScaler().fit_transform(X)
    vae  = build_vae(N_STEPS * n_features)
    print("Fitting VAE on", Xs.shape)
    vae.fit(Xs, epochs=VAE_EPOCHS, batch_size=VAE_BATCH, verbose=1)
    vae.save('vae_wadi.h5')

    # 2) Setup env & agents
    env = EnvTimeSeriesWaDi(SENSOR_CSV, LABEL_CSV, N_STEPS)
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    from tensorflow.compat.v1.keras import backend as K
    K.set_session(sess)

    ql = QNet(scope='qlearn'); qt = QNet(scope='qtarget')
    split = int(len(env.timeseries_repo[0]) * VALIDATION_SPLIT)

    # 3) Train with dynamic reward
    coefs = q_learning(env, sess, ql, qt,
                       EPISODES, 5, 1,
                       1.0, 0.1, 10000,
                       BATCH_SIZE, discount,
                       nLP, nAL,
                       vae, init_coef=20.0)

    # 4) Validate
    exp = f'exp_AL{nAL}'; val = os.path.join(exp,'val')
    f1, aupr = validate(env, sess, ql, split, val)
    print(f"AL={nAL}: Avg F1={f1:.3f}, Avg AUPR={aupr:.3f}")

    # 5) Plot coef history
    pdir = os.path.join(exp,'plots'); os.makedirs(pdir,exist_ok=True)
    plt.figure(); plt.plot(coefs); plt.title('dynamic_coef'); plt.savefig(os.path.join(pdir,'coef.png'))
    plt.close()

if __name__=="__main__":
    for AL in [1000,5000,10000]:
        train_wrapper(NUM_LP, AL, discount=0.96)
