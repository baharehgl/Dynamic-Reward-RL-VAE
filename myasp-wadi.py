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
from tensorflow.keras import layers, models, losses
from tensorflow.keras.models import load_model

from env_wadi import EnvTimeSeriesWaDi

tf.compat.v1.disable_eager_execution()

#########################
# Hyperparameters & Paths
#########################
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

###################
# VAE Definition
###################
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]; dim = tf.shape(z_mean)[1]
        eps   = tf.keras.backend.random_normal(shape=(batch,dim))
        return z_mean + tf.exp(0.5*z_log_var)*eps

def build_vae(original_dim, latent_dim=10, intermediate_dim=64):
    inp       = layers.Input(shape=(original_dim,))
    h1        = layers.Dense(intermediate_dim,activation='relu')(inp)
    h2        = layers.Dense(intermediate_dim,activation='relu')(h1)
    z_mean    = layers.Dense(latent_dim)(h2)
    z_log_var = layers.Dense(latent_dim)(h2)
    z_log_var = tf.clip_by_value(z_log_var,-10.0,10.0)
    z         = Sampling()([z_mean,z_log_var])
    dh        = layers.Dense(intermediate_dim,activation='relu')
    dm        = layers.Dense(original_dim,activation='sigmoid')
    out       = dm(dh(z))

    vae       = models.Model(inp,out)
    recon     = losses.mse(inp,out)*original_dim
    kl        = -0.5*tf.reduce_sum(1+z_log_var - tf.square(z_mean)-tf.exp(z_log_var),axis=-1)
    vae.add_loss(tf.reduce_mean(recon+kl))
    vae.compile(optimizer='adam')
    return vae

####################
# State & Reward
####################
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
        err   = np.mean((recon-win)**2)
        penalty = coef*err
    lbl = ts['label'].iat[c]
    return [TN+penalty,FP+penalty] if lbl==0 else [FN+penalty,TP+penalty]

def RNNBinaryRewardFucTest(ts,c,action):
    if c< N_STEPS: return [0,0]
    lbl=ts['anomaly'].iat[c]
    return [TN,FP] if lbl==0 else [FN,TP]

####################
# Q-Network
####################
class Q_Estimator_Nonlinear:
    def __init__(self,lr=3e-4,scope='q'):
        self.scope=scope
        with tf.compat.v1.variable_scope(scope):
            self.state  = tf.compat.v1.placeholder(tf.float32,[None,N_STEPS,N_INPUT_DIM],name='state')
            self.target = tf.compat.v1.placeholder(tf.float32,[None,ACTION_SPACE_N],name='target')
            seq = tf.compat.v1.unstack(self.state,N_STEPS,axis=1)
            cell= tf.compat.v1.nn.rnn_cell.LSTMCell(N_HIDDEN_DIM)
            out,_= tf.compat.v1.nn.static_rnn(cell,seq,dtype=tf.float32)
            self.logits   = layers.Dense(ACTION_SPACE_N)(out[-1])
            self.loss     = tf.reduce_mean(tf.square(self.logits-self.target))
            self.train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(self.loss)

    def predict(self,state,sess):
        return sess.run(self.logits,{self.state:state})

    def update(self,state,target,sess):
        sess.run(self.train_op,{self.state:state,self.target:target})

###################
# Active Learning
###################
class active_learning:
    def __init__(self,env,N,estimator,already):
        self.env=env; self.N=N
        self.est=estimator; self.already=already
    def get_samples(self):
        distances=[]
        for s in self.env.states_list:
            q=self.est.predict([s],self.est.session)[0]
            distances.append(abs(q[0]-q[1]))
        idx=np.argsort(distances)
        return [i for i in idx if i not in self.already][:self.N]

###################
# Helpers
###################
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
    for s,d in zip(sorted(sv,key=lambda v:v.name),sorted(dv,key=lambda v:v.name)):
        sess.run(d.assign(s))

class WarmUp:
    def warm_up_isolation_forest(self,frac,data):
        from sklearn.ensemble import IsolationForest
        clf=IsolationForest(contamination=frac)
        clf.fit(data)
        return clf

####################
# Q-Learning
####################
def q_learning(env,sess,ql,qt,
               num_episodes,num_epoches,
               update_target_every,
               eps_start,eps_end,eps_steps,
               batch_size,coef,discount,
               num_LP, num_AL):
    Transition=namedtuple("T",["state","reward","next_state","done"])
    memory=[]
    sess.run(tf.compat.v1.global_variables_initializer())
    epsilons=np.linspace(eps_start,eps_end,eps_steps)
    policy=make_epsilon_greedy_policy(ql,ACTION_SPACE_N,sess)

    # warm-up
    env.reset()
    data=[s for s in env.states_list if s is not None][:MAX_WARMUP_SAMPLES]
    X=np.array([s[-1][0] for s in data]).reshape(-1,1)
    _=WarmUp().warm_up_isolation_forest(0.01,X)

    for ep in range(num_episodes):
        # --- LabelSpreading on a subsample to avoid OOM ---
        state_list = np.array(env.states_list)
        label_list = np.array([env.timeseries['label'].iat[i] for i in range(N_STEPS,len(env.timeseries))])
        M = min(len(state_list), 5000)
        idxs = random.sample(range(len(state_list)), M)
        sub_states = state_list[idxs]
        sub_labels = label_list[idxs]
        lp_model = LabelSpreading(kernel='knn', n_neighbors=10)
        lp_model.fit(sub_states, sub_labels)
        ent = np.apply_along_axis(lambda d: -np.sum(d*np.log(d+1e-12)), 1, lp_model.label_distributions_)
        uncert = np.argsort(ent)[:num_LP]
        for u in uncert:
            orig = idxs[u] + N_STEPS
            env.timeseries['label'].iat[orig] = lp_model.transduction_[u]

        # --- Active Learning ---
        labeled_idx=[i for i in range(len(env.timeseries['label'])) if env.timeseries['label'].iat[i]!=-1]
        al=active_learning(env,num_AL,ql,labeled_idx)
        for s in al.get_samples():
            env.timeseries['label'].iat[s+N_STEPS] = env.timeseries['anomaly'].iat[s+N_STEPS]

        # --- one episode rollout ---
        state, ep_r = env.reset(), 0
        while True:
            a=np.random.choice(ACTION_SPACE_N,p=policy(state,epsilons[0]))
            nxt,r,done,_=env.step(a)
            ep_r+=r[a]
            memory.append(Transition(state,r,nxt,done))
            if done: break
            state=nxt[a] if (isinstance(nxt,np.ndarray) and nxt.ndim>2) else nxt

        # --- training steps ---
        for _ in range(num_epoches):
            batch=random.sample(memory,batch_size)
            S,R,NS,D = map(np.array,zip(*batch))
            q0=qt.predict(NS[:,0],sess); q1=qt.predict(NS[:,1],sess)
            tgt=R+discount*np.stack((q0.max(1),q1.max(1)),axis=1)
            ql.update(S,tgt,sess)
        copy_model_parameters(sess,ql,qt)

    return

####################
# Validation
####################
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
        preds, gts, ts_vals = [],[],[]
        while True:
            a=np.argmax(policy(state,0))
            preds.append(a)
            gts.append(env.timeseries['anomaly'].iat[env.timeseries_curser])
            ts_vals.append(state[-1][0])
            state,_,done,_=env.step(a)
            if done: break
            if isinstance(state,np.ndarray) and state.ndim>2: state=state[a]
        p,r,f1,_=precision_recall_fscore_support(gts,preds,average='binary',zero_division=0)
        aupr=average_precision_score(gts,preds)
        f.write(f"E{i+1}:P={p:.3f},R={r:.3f},F1={f1:.3f},AUPR={aupr:.3f}\n")
        all_f1.append(f1); all_aupr.append(aupr)
        if plot:
            fig,ax=plt.subplots(4,1,sharex=True)
            ax[0].plot(ts_vals);ax[0].set_title('TS')
            ax[1].plot(preds,'g-');ax[1].set_title('Pred')
            ax[2].plot(gts,'r-');ax[2].set_title('GT')
            ax[3].plot([aupr]*len(ts_vals),'m-');ax[3].set_title('AUPR')
            fig.savefig(os.path.join(record_dir,f'v{i+1}.png'));plt.close(fig)
    f.close()
    return np.mean(all_f1), np.mean(all_aupr)

###################
# Plotting
###################
def save_plots(d,rews,coefs):
    os.makedirs(d,exist_ok=True)
    plt.figure();plt.plot(rews);plt.title('Rew');plt.savefig(os.path.join(d,'rews.png'));plt.close()
    plt.figure();plt.plot(coefs);plt.title('Coefs');plt.savefig(os.path.join(d,'coefs.png'));plt.close()

####################
# Main
####################
def pretrain_vae():
    tf.compat.v1.reset_default_graph()
    from tensorflow.compat.v1.keras import backend as K; sess= tf.compat.v1.Session(); K.set_session(sess)
    df=pd.read_csv(SENSOR_CSV)
    raw=pd.read_csv(LABEL_CSV,header=1,low_memory=False)["Attack LABLE (1:No Attack, -1:Attack)"].astype(int).values
    labels=np.where(raw==1,0,1);L=min(len(df),len(labels))
    vals=df['TOTAL_CONS_REQUIRED_FLOW'].values[:L];lab=labels[:L]
    normal=vals[lab==0]
    W=[normal[i:i+N_STEPS] for i in range(len(normal)-N_STEPS+1)]
    X=np.array(W,dtype=np.float32)
    scaler=StandardScaler().fit(X);Xs=scaler.transform(X)
    vae=build_vae(N_STEPS)
    print("VAE training…")
    with sess.as_default(): vae.fit(Xs,epochs=2,batch_size=32,verbose=1)
    vae.save('vae_wadi.h5'); sess.close()

def main():
    pretrain_vae()
    for LP,AL in [(200,1000),(200,5000),(200,10000)]:
        print(f"LP={LP},AL={AL}")
        tf.compat.v1.reset_default_graph()
        sess=tf.compat.v1.Session()
        from tensorflow.compat.v1.keras import backend as K2; K2.set_session(sess)
        vae_model=load_model('vae_wadi.h5',custom_objects={'Sampling':Sampling},compile=False)
        env=EnvTimeSeriesWaDi(SENSOR_CSV,LABEL_CSV,N_STEPS)
        env.statefnc=RNNBinaryStateFuc
        env.rewardfnc=lambda ts,tc,a:RNNBinaryRewardFuc(ts,tc,a,vae_model,coef=20.0,include_vae=True)
        ql=Q_Estimator_Nonlinear(scope='qlearn'); qt=Q_Estimator_Nonlinear(scope='qtarget')
        L=len(env.timeseries_repo[0]); split_idx=int(L*VALIDATION_SEPARATE_RATIO)
        exp=f'exp_LP{LP}_AL{AL}'; os.makedirs(exp,exist_ok=True)
        q_learning(env,sess,ql,qt,EPISODES,10,1000,1.0,0.1,50000,256,20.0,DISCOUNT_FACTOR,LP,AL)
        save_plots(exp,[],[])  # you can collect rewards/coefs if desired
        f1,aupr=q_learning_validator(env,sess,ql,split_idx,os.path.join(exp,'val'),plot=True)
        print(f"→ F1={f1:.3f},AUPR={aupr:.3f}")

if __name__=="__main__":
    main()
