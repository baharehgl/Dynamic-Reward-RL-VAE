#!/usr/bin/env python3
import os, random, itertools
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from sklearn.ensemble import IsolationForest

# ─────────────── 0) HYPERPARAMETERS ─────────────────────────
WA_DI_DIR    = "./WaDi"
SENSOR_CSV   = os.path.join(WA_DI_DIR, "WADI_14days_new.csv")
LABEL_CSV    = os.path.join(WA_DI_DIR, "WADI_attackdataLABLE.csv")
N_STEPS      = 25
LATENT_DIM   = 10
VAE_HIDDEN   = 64
VAE_SAMPLES  = 200
VAE_EPOCHS   = 2
VAE_BATCH    = 32
EPISODES     = 20
AL_BUDGETS   = [1000,5000,10000]
LP_BUDGET    = 200
DISCOUNT     = 0.5
BATCH_SIZE   = 128
LR           = 3e-4
COEF_ALPHA   = 0.001
COEF_MIN,COEF_MAX = 0.1,10.0
K_SLICES     = 5
TN,TP,FP,FN  = 1,10,-1,-10

# ─────────────── 1) LOAD & CLEAN ────────────────────────────
df = pd.read_csv(SENSOR_CSV, decimal='.') \
       .apply(pd.to_numeric, errors='coerce') \
       .dropna(axis=1, how='all') \
       .dropna(axis=0, how='all') \
       .reset_index(drop=True)
if 'Row' in df.columns: df.drop(columns='Row', inplace=True)
feature_cols = df.columns.tolist()
n_feat = len(feature_cols)

lbl = pd.read_csv(LABEL_CSV, header=1)["Attack LABLE (1:No Attack, -1:Attack)"].values
anomaly = np.where(lbl==-1,1,0)[:len(df)]
df['label']   = np.where(anomaly==1,1,0)
df['anomaly'] = anomaly

# ─────────────── 2) VAE PRETRAIN ────────────────────────────
normal_idx = np.where(df['label'].values[N_STEPS:]==0)[0] + N_STEPS
sample_idx = np.random.choice(normal_idx, min(VAE_SAMPLES,len(normal_idx)), replace=False)
X = np.stack([df.iloc[i-N_STEPS+1:i+1][feature_cols].values.flatten() for i in sample_idx])
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X).astype('float32')

from tensorflow.keras import layers, models, losses
inp = layers.Input((n_feat*N_STEPS,))
h1  = layers.Dense(VAE_HIDDEN, activation='relu')(inp)
z_mu = layers.Dense(LATENT_DIM)(h1)
z_lv = layers.Lambda(lambda x: tf.clip_by_value(x,-10.,10.))(layers.Dense(LATENT_DIM)(h1))
z    = layers.Lambda(lambda t: t[0]+tf.exp(0.5*t[1])*tf.random.normal(tf.shape(t[0])))([z_mu,z_lv])
h2 = layers.Dense(VAE_HIDDEN, activation='relu')(z)
out= layers.Dense(n_feat*N_STEPS, activation='sigmoid')(h2)
vae = models.Model(inp,out)
recon = losses.mse(inp,out)*(n_feat*N_STEPS)
kl    = -0.5*tf.reduce_sum(1+z_lv-tf.square(z_mu)-tf.exp(z_lv),axis=1)
vae.add_loss(tf.reduce_mean(recon+kl))
vae.compile(optimizer='adam')
vae.fit(X_scaled, epochs=VAE_EPOCHS, batch_size=VAE_BATCH, verbose=1)
vae.save('vae_wadi.h5')

all_wins = np.stack([df.iloc[i-N_STEPS+1:i+1][feature_cols].values.flatten()
                     for i in range(N_STEPS-1,len(df))])
all_scaled = scaler.transform(all_wins).astype('float32')
errs=[]
for i in range(0,len(all_scaled),VAE_BATCH):
    chunk=all_scaled[i:i+VAE_BATCH]
    pred=vae.predict(chunk,verbose=0)
    errs.append(((pred-chunk)**2).mean(axis=1))
penalty = np.concatenate(errs)
penalty = np.concatenate([np.zeros(N_STEPS-1), penalty])

# ─────────────── 3) ENV CLASS ───────────────────────────────
class EnvWaDi:
    def __init__(self, df, steps):
        self.repo = df
        self.steps= steps
    def reset(self):
        self.t = self.steps
        return self._make_state(self.t)
    def _make_state(self,c):
        W = self.repo.iloc[c-self.steps+1:c+1][feature_cols].values.astype('float32')
        a0 = np.concatenate([W, np.zeros((self.steps,1),dtype='float32')],1)
        a1 = np.concatenate([W, np.ones ((self.steps,1),dtype='float32')],1)
        return np.stack([a0,a1],0)
    def step(self,a,coef):
        pen = coef*penalty[self.t]
        base= [TN,FP] if self.repo['label'].iat[self.t]==0 else [FN,TP]
        r   = base[a]+pen
        self.t+=1
        done = self.t>=len(self.repo)
        s    = self._make_state(self.t) if not done else np.zeros_like(self._make_state(self.steps))
        return s, r, done

# ─────────────── 4) Q-NET (TF1) ─────────────────────────────
tf.compat.v1.disable_eager_execution()
class QNet:
    def __init__(self,scope):
        self.sc=scope
        with tf.compat.v1.variable_scope(scope):
            self.S = tf.compat.v1.placeholder(tf.float32,[None,N_STEPS,n_feat+1])
            self.T = tf.compat.v1.placeholder(tf.float32,[None,1])
            seq = tf.compat.v1.unstack(self.S,N_STEPS,1)
            cell= tf.compat.v1.nn.rnn_cell.LSTMCell(64)
            out,_= tf.compat.v1.nn.static_rnn(cell,seq,dtype=tf.float32)
            qv   = layers.Dense(1)(out[-1])
            self.loss= tf.reduce_mean(tf.square(qv-self.T))
            self.train= tf.compat.v1.train.AdamOptimizer(LR).minimize(self.loss)
            self.Q    = tf.identity(qv)
    def predict(self,s,sess): return sess.run(self.Q,{self.S:s})
    def update(self,s,t,sess): sess.run(self.train,{self.S:s,self.T:t})

def copy_params(sess,src,dst):
    vs= tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, src.sc)
    vt= tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, dst.sc)
    for a,b in zip(sorted(vs,key=lambda v:v.name), sorted(vt,key=lambda v:v.name)):
        sess.run(b.assign(a))

# ─────────────── 5) TRAIN & VALIDATE ────────────────────────
def train_and_validate(AL):
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    from tensorflow.compat.v1.keras import backend as K; K.set_session(sess)

    ql, qt = QNet('ql'), QNet('qt')
    sess.run(tf.compat.v1.global_variables_initializer())
    env = EnvWaDi(df,N_STEPS)
    mem, coef = [], 1.0

    # warm-up
    X0=[df.iloc[i-N_STEPS+1:i+1][feature_cols].values.flatten()
        for i in range(N_STEPS,len(df)) if df['label'].iat[i]==0]
    if X0: IsolationForest(contamination=0.01).fit(X0[:VAE_SAMPLES])

    eps_decay = np.linspace(1.0,0.1,EPISODES)
    for ep in range(EPISODES):
        # LP + AL
        states=[env._make_state(c).flatten() for c in range(N_STEPS,len(df))]
        labs = df['label'].values[N_STEPS:]
        if (labs!=-1).any():
            lp=LabelSpreading(kernel='knn',n_neighbors=10).fit(states,labs)
            uncert=1-lp.label_distributions_.max(axis=1)
            idx   = np.argsort(-uncert)
            for i in idx[:AL]: df.at[i+N_STEPS,'label']=df.at[i+N_STEPS,'anomaly']
            for i in idx[AL:AL+LP_BUDGET]: df.at[i+N_STEPS,'label']=lp.transduction_[i]

        s, done, ep_r = env.reset(), False, 0.0
        while not done:
            if random.random()<eps_decay[ep]:
                a = random.choice([0,1])
            else:
                q0 = ql.predict(s[0][None],sess)[0][0]
                q1 = ql.predict(s[1][None],sess)[0][0]
                a  = int(q1>q0)
            ns,r,done=env.step(a,coef)
            mem.append((s[a],r,ns[a],done))
            ep_r+=r; s=ns

        for _ in range(5):
            batch=random.sample(mem,min(BATCH_SIZE,len(mem)))
            S,R,NS,D=zip(*batch)
            S=np.stack(S)[:,None,:]; NS=np.stack(NS)[:,None,:]
            R=np.array(R)[:,None]
            qn=qt.predict(NS,sess)
            tgt=R+DISCOUNT*qn
            ql.update(S,tgt,sess)
        copy_params(sess,ql,qt)
        coef = max(min(coef+COEF_ALPHA*(ep_r),COEF_MAX),COEF_MIN)
        print(f"[train AL={AL}] ep{ep+1}/{EPISODES} coef={coef:.2f} reward={ep_r:.2f}")

    # equal-slice val
    seg=len(df)//K_SLICES
    f1s,auprs=[],[]
    outdir=f"exp_AL{AL}"; os.makedirs(outdir,exist_ok=True)
    for k in range(K_SLICES):
        sub=df.iloc[k*seg:(k+1)*seg].reset_index(drop=True)
        envv=EnvWaDi(sub,N_STEPS); s,done=envv.reset(),False
        P,G,V=[],[],[]
        while not done:
            q0=ql.predict(s[0][None],sess)[0][0]
            q1=ql.predict(s[1][None],sess)[0][0]
            a = int(q1>q0)
            P.append(a); G.append(envv.repo['anomaly'].iat[envv.t]); V.append(s[1][-1,0])
            s,_,done=envv.step(a,coef)
        p,r,f1,_=precision_recall_fscore_support(G,P,average='binary',zero_division=0)
        au=average_precision_score(G,P)
        f1s.append(f1); auprs.append(au)
        # plots
        fig,ax=plt.subplots(4,1,figsize=(6,8))
        ax[0].plot(V); ax[0].set_title("TS")
        ax[1].plot(P); ax[1].set_title("Pred")
        ax[2].plot(G); ax[2].set_title("True")
        ax[3].plot([au]*len(V)); ax[3].set_title(f"AUPR {au:.3f}")
        fig.savefig(os.path.join(outdir,f"slice{k}.png")); plt.close(fig)
        print(f"[val AL={AL}] slice{k+1}/{K_SLICES} P={p:.3f} R={r:.3f} F1={f1:.3f} AUPR={au:.3f}")

    print(f"[val AL={AL}] mean F1={np.mean(f1s):.3f} mean AUPR={np.mean(auprs):.3f}\n")

# ─────────────── 6) DRIVER ────────────────────────────────────
if __name__=="__main__":
    for AL in AL_BUDGETS:
        print(f"\n=== ACTIVE LEARNING BUDGET: {AL} ===")
        train_and_validate(AL)
