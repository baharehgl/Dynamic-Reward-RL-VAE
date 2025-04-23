import os, random
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

tf.compat.v1.disable_eager_execution()

# ── Hyperparameters ────────────────────────────────────────────────────────────
EPISODES                  = 2
N_STEPS                   = 25
N_INPUT_DIM               = 2
N_HIDDEN_DIM              = 128
DISCOUNT_FACTOR           = 0.5
TN, TP, FP, FN            = 1, 10, -1, -10
ACTION_SPACE_N            = 2
VALIDATION_SEPARATE_RATIO = 0.8
MAX_WARMUP_SAMPLES        = 10000
NUM_LABELPROPAGATION      = 20
NUM_ACTIVE_LEARNING       = 10

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
WA_DI_DIR  = os.path.join(BASE_DIR,'WaDi')
SENSOR_CSV = os.path.join(WA_DI_DIR,'WADI_14days_new.csv')
LABEL_CSV  = os.path.join(WA_DI_DIR,'WADI_attackdataLABLE.csv')

# ── VAE Definition ─────────────────────────────────────────────────────────────
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]; dim = tf.shape(z_mean)[1]
        eps   = tf.keras.backend.random_normal(shape=(batch,dim))
        return z_mean + tf.exp(0.5*z_log_var)*eps

def build_vae(original_dim, latent_dim=10, intermediate_dim=64):
    inp       = layers.Input(shape=(original_dim,))
    h1        = layers.Dense(intermediate_dim, activation='relu')(inp)
    z_mean    = layers.Dense(latent_dim)(h1)
    z_log_var = layers.Dense(latent_dim)(h1)
    z_log_var = tf.clip_by_value(z_log_var,-10.0,10.0)
    z         = Sampling()([z_mean, z_log_var])
    out       = layers.Dense(original_dim, activation='sigmoid')(layers.Dense(intermediate_dim, activation='relu')(z))
    vae       = models.Model(inp, out)
    recon     = losses.mse(inp,out)*original_dim
    kl        = -0.5*tf.reduce_sum(1+z_log_var - tf.square(z_mean)-tf.exp(z_log_var),axis=-1)
    vae.add_loss(tf.reduce_mean(recon+kl))
    vae.compile(optimizer='adam')
    return vae

# ── State & Reward ────────────────────────────────────────────────────────────
def RNNBinaryStateFuc(ts,c,prev=None,action=None):
    if c==N_STEPS:
        s=[[ts['value'].iat[i],0] for i in range(N_STEPS)]
        s.pop(0); s.append([ts['value'].iat[N_STEPS],1])
        return np.array(s,dtype='float32')
    if c> N_STEPS:
        s0=np.concatenate((prev[1:],[[ts['value'].iat[c],0]]))
        s1=np.concatenate((prev[1:],[[ts['value'].iat[c],1]]))
        return np.array([s0,s1],dtype='float32')
    return None

def RNNBinaryRewardFuc(ts,c,action,vae_model=None,coef=1.0,include_vae=True):
    if c< N_STEPS: return [0,0]
    penalty=0.0
    if include_vae and vae_model:
        win   = ts['value'].values[c-N_STEPS:c].reshape(1,-1)
        recon = vae_model.predict(win)
        penalty = coef * np.mean((recon-win)**2)
    lbl = ts['label'].iat[c]
    return [TN+penalty,FP+penalty] if lbl==0 else [FN+penalty,TP+penalty]

# ── Q-Network ──────────────────────────────────────────────────────────────────
class Q_Estimator_Nonlinear:
    def __init__(self,lr=3e-4,scope='q'):
        self.scope=scope
        with tf.compat.v1.variable_scope(scope):
            self.state  = tf.compat.v1.placeholder(tf.float32,[None,N_STEPS,N_INPUT_DIM],name='state')
            self.target = tf.compat.v1.placeholder(tf.float32,[None,ACTION_SPACE_N],   name='target')
            seq = tf.compat.v1.unstack(self.state,N_STEPS,axis=1)
            cell= tf.compat.v1.nn.rnn_cell.LSTMCell(N_HIDDEN_DIM)
            out,_= tf.compat.v1.nn.static_rnn(cell,seq,dtype=tf.float32)
            self.logits   = layers.Dense(ACTION_SPACE_N)(out[-1])
            self.loss     = tf.reduce_mean(tf.square(self.logits-self.target))
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)

    def predict(self, state, sess):
        return sess.run(self.logits, {self.state: state})

    def update(self, state, target, sess):
        sess.run(self.train_op, {self.state: state, self.target: target})

# ── Helpers ────────────────────────────────────────────────────────────────────
def make_epsilon_greedy_policy(est,nA,sess):
    def policy_fn(obs,eps):
        A=np.ones(nA)*eps/nA
        q=est.predict([obs],sess)[0]
        b=np.argmax(q); A[b]+=(1.0-eps)
        return A
    return policy_fn

def copy_model_parameters(sess,src,dest):
    sv=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,scope=src.scope)
    dv=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,scope=dest.scope)
    for s,d in zip(sorted(sv,key=lambda v:v.name), sorted(dv,key=lambda v:v.name)):
        sess.run(d.assign(s))

# ── Q-Learning ────────────────────────────────────────────────────────────────
def q_learning(env,sess,ql,qt,
               num_episodes,num_epoches,
               update_target_every,
               eps_start,eps_end,eps_steps,
               batch_size,coef,discount,
               num_LP,num_AL):
    T=namedtuple('T',['s','r','ns','d'])
    mem=[]
    sess.run(tf.compat.v1.global_variables_initializer())
    epsilons=np.linspace(eps_start,eps_end,eps_steps)
    policy=make_epsilon_greedy_policy(ql,ACTION_SPACE_N,sess)

    # warm-up IsolationForest on first MAX_WARMUP_SAMPLES states
    env.reset()
    states = [s for s in env.states_list if s is not None][:MAX_WARMUP_SAMPLES]
    X      = np.array([s[-1][0] for s in states]).reshape(-1,1)
    IsolationForest(contamination=0.01).fit(X)

    for ep in range(num_episodes):
        # LabelSpreading on FLATTENED states_list
        arr = np.array(env.states_list)                    # (n,25,2)
        flat = arr.reshape(arr.shape[0], -1)               # (n,50)
        labels = np.array([env.timeseries['label'].iat[i] for i in range(N_STEPS, len(env.timeseries))])
        lp_model = LabelSpreading(kernel='knn',n_neighbors=10)
        lp_model.fit(flat, labels)
        uncert = np.argsort(-np.max(lp_model.label_distributions_,axis=1))[:num_LP]
        for u in uncert:
            env.timeseries['label'].iat[u+N_STEPS] = lp_model.transduction_[u]

        # Active learning
        labeled = {i for i,v in enumerate(env.timeseries['label']) if v!=-1}
        al = __import__('active_learning', fromlist=['active_learning']).active_learning if False else None
        # simplest: just re-use LP picks as “active” for now
        for u in uncert[:num_AL]:
            env.timeseries['label'].iat[u+N_STEPS] = env.timeseries['anomaly'].iat[u+N_STEPS]

        # rollout
        state = env.reset(); ep_r=0
        while True:
            a = np.random.choice(ACTION_SPACE_N,p=policy(state,epsilons[0]))
            nxt,r,done,_ = env.step(a)
            ep_r += r[a]
            mem.append(T(state,r,nxt,done))
            if done: break
            state = nxt[a] if (isinstance(nxt,np.ndarray) and nxt.ndim>2) else nxt

        # train
        for _ in range(num_epoches):
            batch = random.sample(mem,batch_size)
            S,R,NS,D = map(np.array,zip(*batch))
            q0=qt.predict(NS[:,0],sess); q1=qt.predict(NS[:,1],sess)
            tgt = R + discount * np.stack((q0.max(1),q1.max(1)),axis=1)
            ql.update(S,tgt,sess)
        copy_model_parameters(sess,ql,qt)

# ── Validation ─────────────────────────────────────────────────────────────────
def q_learning_validator(env,sess,trained,split_idx,record_dir,plot=True):
    os.makedirs(record_dir,exist_ok=True)
    f=open(os.path.join(record_dir,'perf.txt'),'w')
    all_f1,all_aupr=[],[]
    for i in range(EPISODES):
        policy=make_epsilon_greedy_policy(trained,ACTION_SPACE_N,sess)
        state=env.reset()
        while env.timeseries_curser<split_idx:
            state,_,done,_=env.step(0)
            if done: break
        preds,gts,vals=[],[],[]
        while True:
            a=np.argmax(policy(state,0))
            preds.append(a)
            gts.append(env.timeseries['anomaly'].iat[env.timeseries_curser])
            vals.append(state[-1][0])
            state,_,done,_=env.step(a)
            if done: break
            if isinstance(state,np.ndarray) and state.ndim>2: state=state[a]
        p,r,f1,_=precision_recall_fscore_support(gts,preds,average='binary',zero_division=0)
        aupr=average_precision_score(gts,preds)
        f.write(f"E{i+1}:P={p:.3f},R={r:.3f},F1={f1:.3f},AUPR={aupr:.3f}\n")
        all_f1.append(f1); all_aupr.append(aupr)
        if plot:
            fig,ax=plt.subplots(4,1,sharex=True)
            ax[0].plot(vals);ax[0].set_title('TS')
            ax[1].plot(preds,'g-');ax[1].set_title('Pred')
            ax[2].plot(gts,'r-');ax[2].set_title('GT')
            ax[3].plot([aupr]*len(vals),'m-');ax[3].set_title('AU')
            fig.savefig(os.path.join(record_dir,f'v{i+1}.png'));plt.close(fig)
    f.close()
    return np.mean(all_f1),np.mean(all_aupr)

# ── Main ───────────────────────────────────────────────────────────────────────
def pretrain_vae():
    tf.compat.v1.reset_default_graph()
    from tensorflow.compat.v1.keras import backend as K
    sess = tf.compat.v1.Session(); K.set_session(sess)

    df    = pd.read_csv(SENSOR_CSV)
    raw   = pd.read_csv(LABEL_CSV,header=1,low_memory=False)["Attack LABLE (1:No Attack, -1:Attack)"].astype(int).values
    labels= np.where(raw==1,0,1)
    L     = min(len(df),len(labels))
    vals  = df['TOTAL_CONS_REQUIRED_FLOW'].values[:L]
    normal= vals[labels[:L]==0]
    W     = [normal[i:i+N_STEPS] for i in range(len(normal)-N_STEPS+1)]
    X     = np.array(W,dtype=np.float32)
    Xs    = StandardScaler().fit_transform(X)

    vae = build_vae(N_STEPS)
    with sess.as_default():
        vae.fit(Xs,epochs=2,batch_size=32,verbose=1)
    vae.save('vae_wadi.h5'); sess.close()

def main():
    pretrain_vae()
    for LP,AL in [(200,1000),(200,5000)]:
        tf.compat.v1.reset_default_graph()
        sess = tf.compat.v1.Session()
        from tensorflow.compat.v1.keras import backend as K2; K2.set_session(sess)

        vae_model = load_model('vae_wadi.h5',custom_objects={'Sampling':Sampling},compile=False)
        env       = EnvTimeSeriesWaDi(SENSOR_CSV,LABEL_CSV,N_STEPS)
        env.statefnc  = RNNBinaryStateFuc
        env.rewardfnc = lambda ts,tc,a: RNNBinaryRewardFuc(ts,tc,a,vae_model,coef=20.0,include_vae=True)

        ql = Q_Estimator_Nonlinear(scope='qlearn')
        qt = Q_Estimator_Nonlinear(scope='qtarget')

        split_idx = int(len(env.timeseries_repo[0])*VALIDATION_SEPARATE_RATIO)
        q_learning(env,sess,ql,qt,EPISODES,5,500,1.0,0.1,10000,256,20.0,DISCOUNT_FACTOR,LP,AL)

        val_dir = f'exp_LP{LP}_AL{AL}/validation'
        f1,aupr = q_learning_validator(env,sess,ql,split_idx,val_dir,plot=True)
        print(f"Result LP={LP},AL={AL}: F1={f1:.3f},AUPR={aupr:.3f}")

if __name__=="__main__":
    main()
