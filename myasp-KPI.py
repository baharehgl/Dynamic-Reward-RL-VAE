# myasp_KPI.py

import os, sys
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
from sklearn.ensemble import IsolationForest

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import layers, models, losses

# ─── 1) CONFIG & VAE PRETRAIN ────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
NORMAL_DATA_DIR = os.path.join(current_dir, "normal-data")
n_steps, latent_dim, inter_dim = 25, 10, 64

def load_normal_data(path,n):
    w=[]
    for f in os.listdir(path):
        if not f.endswith('.csv'): continue
        v=pd.read_csv(os.path.join(path,f))['value'].values
        if len(v)<n: continue
        for i in range(len(v)-n+1): w.append(v[i:i+n])
    return StandardScaler().fit_transform(np.array(w))

class Sampling(layers.Layer):
    def call(self,inputs):
        m,lv=inputs
        eps=tf.keras.backend.random_normal(tf.shape(m))
        return m + tf.exp(0.5*lv)*eps

def build_and_train_vae():
    x_in=layers.Input((n_steps,))
    h=layers.Dense(inter_dim,activation='relu')(x_in)
    h=layers.Dense(inter_dim,activation='relu')(h)
    m=layers.Dense(latent_dim)(h)
    lv=layers.Dense(latent_dim)(h)
    lv=tf.clip_by_value(lv,-10,10)
    z=Sampling()([m,lv])
    d=layers.Dense(inter_dim,activation='relu')(z)
    x_dec=layers.Dense(n_steps,activation='sigmoid')(d)

    vae=models.Model(x_in,x_dec)
    recon=losses.mse(x_in,x_dec)*n_steps
    kl  = -0.5*tf.reduce_sum(1+lv-tf.square(m)-tf.exp(lv),axis=-1)
    vae.add_loss(tf.reduce_mean(recon+kl))
    vae.compile(optimizer='adam')

    X=load_normal_data(NORMAL_DATA_DIR,n_steps)
    vae.fit(X,epochs=10,batch_size=32,verbose=1)
    return vae

vae_model = build_and_train_vae()


# ─── 2) IMPORT KPI ENV & DEFINE STATE/REWARD ───────────────────────
sys.path.append(current_dir)
from env_KPI import EnvKPI

TN,TP,FP,FN = 1, 5, -1, -5

def state_fn(ts,cur,prev=None,act=None):
    if cur<n_steps: return None
    if cur==n_steps:
        s=[[ts['value'].iat[i],0] for i in range(n_steps)]
        s.pop(0); s.append([ts['value'].iat[cur],1])
        return np.array(s,'float32')
    s0=np.concatenate((prev[1:],[[ts['value'].iat[cur],0]]))
    s1=np.concatenate((prev[1:],[[ts['value'].iat[cur],1]]))
    return np.array([s0,s1],'float32')

def reward_fn(ts,cur,act,lam):
    cur_win=np.array([ts['value'].iloc[cur-n_steps:cur]])
    err=np.mean((vae_model.predict(cur_win)-cur_win)**2)
    pen=lam*err
    lbl=ts['label'].iat[cur]
    r1 = TP if (act==1 and lbl==1) else TN if (act==0 and lbl==0) else FP if act==1 else FN
    # return vector [r(normal),r(anomaly)] + penalty
    return [ (TN+pen) if lbl==0 else (FN+pen),
             (FP+pen) if lbl==0 else (TP+pen) ]

def reward_fn_test(ts,cur,act=0):
    an=ts['anomaly'].iat[cur]
    return [TN,FP] if an==0 else [FN,TP]


# ─── 3) Q-NET + POLICIES + ACTIVE LEARNING ──────────────────────────
nA=2; n_input=2; n_hidden=128

class QNet:
    def __init__(self,lr,scope,sess):
        self.sess=sess; self.scope=scope
        with tf.compat.v1.variable_scope(scope):
            self.st=tf.compat.v1.placeholder(tf.float32,[None,n_steps,n_input],"state")
            self.tg=tf.compat.v1.placeholder(tf.float32,[None,nA],"target")
            cell=tf.compat.v1.nn.rnn_cell.LSTMCell(n_hidden)
            outs,_=tf.compat.v1.nn.dynamic_rnn(cell,self.st,dtype=tf.float32)
            h=outs[:,-1,:]
            self.q=layers.Dense(nA)(h)
            self.loss=tf.reduce_mean(tf.square(self.q-self.tg))
            self.train=tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)

    def predict(self,s):
        return self.sess.run(self.q,{self.st:s})

    def update(self,s,t):
        self.sess.run(self.train,{self.st:s,self.tg:t})

def make_policy(est):
    def pol(obs,eps):
        A=np.ones(nA)*(eps/nA)
        q=est.predict([obs])[0]
        A[np.argmax(q)]+=1-eps
        return A
    return pol

class ActiveLearner:
    def __init__(self,env,N,est):
        self.env, self.N, self.est = env, N, est

    def select(self):
        D=[]
        for s in self.env.states_list:
            q=self.est.predict([s])[0]
            D.append(abs(q[0]-q[1]))
        idx=np.argsort(D)
        return idx[:self.N]


# ─── 4) TRAIN w/ VALIDATION & EARLY STOP ───────────────────────────
def train_with_validation(train_csv,test_csv,
                          episodes=100,
                          epoches=5,
                          val_every=10,
                          patience=2,
                          num_AL=10,
                          discount=0.96):

    env = EnvKPI(train_csv,test_csv)
    env.statefnc, env.rewardfnc = state_fn, None
    total=env.datasetsize
    cut=int(total*validation_separate_ratio)
    train_ids=list(range(cut))
    val_ids=list(range(cut,total))

    env_val = EnvKPI(train_csv,test_csv)
    env_val.statefnc, env_val.rewardfnc = state_fn, reward_fn_test

    tf.compat.v1.reset_default_graph()
    sess=tf.compat.v1.Session()
    tf.compat.v1.keras.backend.set_session(sess)

    qnet   = QNet(3e-4,"q",sess)
    tgt    = QNet(3e-4,"tgt",sess)
    sess.run(tf.compat.v1.global_variables_initializer())

    best_f1, no_imp, lam = 0.0, 0, 10.0
    R_hist, L_hist = [], []

    for ep in range(1,episodes+1):
        ki=random.choice(train_ids)
        state=env.reset(to_idx=ki)
        env.rewardfnc=lambda ts,c,a: reward_fn(ts,c,a,lam)
        env.states_list=env.get_states_list()

        # active learning
        al=ActiveLearner(env,num_AL,qnet)
        for s in al.select():
            p=s+n_steps
            env.timeseries['label'].iat[p]=env.timeseries['anomaly'].iat[p]

        # collect transitions via sliding-window states
        mem=[]
        policy=make_policy(qnet)
        for t,s in enumerate(env.states_list):
            eps=max(0.1,1-ep/episodes)
            a=np.random.choice(nA,p=policy(s,eps))
            r=env.rewardfnc(env.timeseries,t+n_steps,a)
            mem.append((s,r,s,False))

        # train
        for _ in range(epoches):
            batch=random.sample(mem,min(len(mem),64))
            S,R,NS,_=map(np.array,zip(*batch))
            qn=tgt.predict(NS)
            mx=np.max(qn,axis=1)
            tgtv=R + discount*mx[:,None]
            qnet.update(S,tgtv)

        # sync target
        if ep%5==0:
            vq=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,"q")
            vt=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,"tgt")
            for x,y in zip(vq,vt): sess.run(y.assign(x))

        # log & update lam
        total_r=sum(r[a] for (s,r,ns,d) in mem for a in [np.argmax(r)])
        R_hist.append(total_r)
        lam = max(0.1, min(10.0, lam + 0.001*(0.0-total_r)))
        L_hist.append(lam)

        # validation
        if ep%val_every==0:
            f1s=[]
            for vid in val_ids:
                st=env_val.reset(to_idx=vid)
                env_val.rewardfnc=reward_fn_test
                preds,truths=[],[]
                for t,s in enumerate(env_val.get_states_list()):
                    a=np.argmax(policy(s,0.0))
                    preds.append(a)
                    truths.append(env_val.timeseries['anomaly'].iat[t+n_steps])
                _,_,f,_=precision_recall_fscore_support(truths,preds,average='binary',zero_division=0)
                f1s.append(f)
            vf1=np.mean(f1s)
            print(f"[VAL] Ep {ep} → F1={vf1:.4f}")
            if vf1>best_f1: best_f1, no_imp = vf1, 0
            else:
                no_imp+=1
                if no_imp>=patience:
                    print(f"Early stopping at episode {ep}")
                    break

    # plot curves
    os.makedirs("exp",exist_ok=True)
    plt.figure(); plt.plot(R_hist); plt.title("Episode Reward"); plt.savefig("exp/rewards.png"); plt.close()
    plt.figure(); plt.plot(L_hist); plt.title("Lambda");        plt.savefig("exp/lambda.png");  plt.close()
    print("Best validation F1:", best_f1)

if __name__=="__main__":
    train_csv=os.path.join(current_dir,"KPI_data","train","phase2_train.csv")
    test_csv =os.path.join(current_dir,"KPI_data","test", "phase2_ground_truth.csv")
    train_with_validation(train_csv,test_csv,
                          episodes=100,
                          epoches=5,
                          val_every=10,
                          patience=2,
                          num_AL=10)
