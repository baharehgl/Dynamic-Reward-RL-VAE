import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import matplotlib.pyplot as plt

# ── Reproducibility ─────────────────────────────────────────────────────────────
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ── Paths ────────────────────────────────────────────────────────────────────────
data_dir       = "WaDi"
features_file  = os.path.join(data_dir, "WADI_14days_new.csv")
labels_file    = os.path.join(data_dir, "WADI_attackdataLABLE.csv")

# ── Load & Preprocess ───────────────────────────────────────────────────────────
df_feat = pd.read_csv(features_file)
df_lbl  = pd.read_csv(labels_file, header=1, low_memory=False)["Attack LABLE (1:No Attack, -1:Attack)"]
y_all   = np.where(df_lbl.values == 1, 0, 1)  # 0=normal, 1=anomaly

# keep only numeric, non-constant columns
df_feat = df_feat.select_dtypes(include=[np.number])
df_feat = df_feat.loc[:, df_feat.std() > 0.0]

# fill NaNs
df_feat.fillna(df_feat.mean(), inplace=True)
X_all = df_feat.values.astype(np.float32)

# ── 80/20 Split ─────────────────────────────────────────────────────────────────
n = len(X_all)
split_idx = int(n * 0.8)
X_val     = X_all[:split_idx]
y_val     = y_all[:split_idx]
X_hold    = X_all[split_idx:]
y_hold    = y_all[split_idx:]

# ── MinMax Scaling based on val set ──────────────────────────────────────────────
min_v = X_val.min(axis=0)
max_v = X_val.max(axis=0)
rng   = np.where(max_v - min_v == 0, 1.0, max_v - min_v)
X_val_s  = (X_val  - min_v) / rng
X_hold_s = (X_hold - min_v) / rng

# ── VAE Definition ──────────────────────────────────────────────────────────────
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim  = X_val_s.shape[1]
latent_dim = 5

class VAE(nn.Module):
    def __init__(self, D, L, H=64):
        super().__init__()
        self.enc1 = nn.Linear(D,H); self.enc2 = nn.Linear(H,H//2)
        self.mu   = nn.Linear(H//2, L); self.logv = nn.Linear(H//2, L)
        self.dec1 = nn.Linear(L, H//2); self.dec2 = nn.Linear(H//2,H)
        self.out  = nn.Linear(H, D); self.relu = nn.ReLU()
    def encode(self,x):
        h = self.relu(self.enc1(x)); h = self.relu(self.enc2(h))
        return self.mu(h), self.logv(h)
    def reparam(self,m,lv):
        std = torch.exp(0.5*lv); eps = torch.randn_like(std)
        return m + eps*std
    def decode(self,z):
        h = self.relu(self.dec1(z)); h = self.relu(self.dec2(h))
        return self.out(h)
    def forward(self,x):
        m,lv = self.encode(x); z=self.reparam(m,lv)
        return self.decode(z), m, lv

vae      = VAE(input_dim, latent_dim).to(device)
opt_vae  = optim.Adam(vae.parameters(), lr=1e-3)
epochs   = 10; bs = 256
vae.train()
for ep in range(epochs):
    perm, tot = np.random.permutation(len(X_val_s)), 0.0
    for i in range(0, len(X_val_s), bs):
        batch = torch.tensor(X_val_s[perm[i:i+bs]],device=device)
        opt_vae.zero_grad()
        recon, m, lv = vae(batch)
        mse = nn.functional.mse_loss(recon, batch, reduction="sum")
        kl  = -0.5*torch.sum(1+lv - m.pow(2) - lv.exp())
        loss = mse + kl; loss.backward(); opt_vae.step()
        tot += loss.item()
    print(f"VAE Ep{ep+1}/{epochs} avg loss {tot/len(X_val_s):.4f}")
vae.eval()

# ── Latent & Recon Errors ────────────────────────────────────────────────────────
with torch.no_grad():
    tv    = torch.tensor(X_val_s,device=device)
    rv, mv, lv_ = vae(tv)
    z_val    = mv.cpu().numpy()
    err_val  = np.mean((rv.cpu().numpy() - X_val_s)**2, axis=1)
    th99     = np.percentile(err_val,99)
    ths     = torch.tensor(X_hold_s,device=device)
    rh, mh, lh = vae(ths)
    z_hold   = mh.cpu().numpy()
    err_hold = np.mean((rh.cpu().numpy() - X_hold_s)**2, axis=1)

# ── DQN Agent ────────────────────────────────────────────────────────────────────
class DQN:
    def __init__(self,S,A=2,H=64,lr=1e-3,γ=0.99):
        self.net = nn.Sequential(nn.Linear(S,H),nn.ReLU(),
                                  nn.Linear(H,H),nn.ReLU(),
                                  nn.Linear(H,A)).to(device)
        self.tgt = nn.Sequential(*[l for l in self.net]).to(device)
        self.tgt.load_state_dict(self.net.state_dict())
        self.opt  = optim.Adam(self.net.parameters(), lr=lr)
        self.γ     = γ; self.eps=1.0; self.eps_end=0.1; self.eps_dec=1e-3
        self.mem   = []; self.mem_cap=50000; self.steps=0
    def act(self,s):
        if random.random()<self.eps:
            return random.randrange(2)
        with torch.no_grad():
            q = self.net(torch.tensor(s,device=device).float().unsqueeze(0))
            return int(q.argmax().cpu())
    def store(self,s,a,r,ns,d):
        if len(self.mem)<self.mem_cap: self.mem.append(None)
        self.mem[self.steps%self.mem_cap] = (s,a,r,ns,d)
        self.steps+=1
    def learn(self,bs=64):
        if len(self.mem)<bs: return
        batch = random.sample(self.mem,bs)
        s,a,r,ns,d = zip(*batch)
        s  = torch.tensor(s,device=device).float()
        ns = torch.tensor([[] if x is None else x for x in ns],device=device).float()
        a  = torch.tensor(a,device=device).long()
        r  = torch.tensor(r,device=device).float()
        d  = torch.tensor(d,device=device).float()
        qsa = self.net(s).gather(1,a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            qns,_  = self.tgt(ns).max(1)
            tgt     = r + self.γ*qns*(1-d)
        loss = nn.functional.mse_loss(qsa,tgt)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        if self.eps>self.eps_end: self.eps-=self.eps_dec
        if self.steps%1000==0:
            self.tgt.load_state_dict(self.net.state_dict())

# ── Active Learning + Training ─────────────────────────────────────────────────
budgets = [1000,5000,10000]; episodes = 10
for budget in budgets:
    os.makedirs(f"exp_AL{budget}",exist_ok=True)
    per_ep = budget // episodes
    known = np.full(len(z_val), -1, int)
    agent = DQN(latent_dim)
    metrics=[]
    for ep in range(1,episodes+1):
        unl = np.where(known==-1)[0]
        if len(unl)>0:
            margins = np.array([abs(agent.net(torch.tensor(z_val[i],device=device).float())
                                    .cpu().detach().numpy().ptp()) for i in unl])
            ask = unl[np.argsort(margins)[:per_ep]]
            known[ask] = y_val[ask]
        # propagate
        for i in np.where(known==1)[0]:
            j=i-1
            while j>=0 and y_val[j]==1: known[j]=1; j-=1
            j=i+1
            while j<len(y_val) and y_val[j]==1: known[j]=1; j+=1
        # train on val
        for i in range(len(z_val)):
            s = z_val[i]
            a = agent.act(s)
            ext = 1 if (known[i]==1 and a==1) else -1 if (known[i]==1 and a==0) else 0
            r   = ext + err_val[i]
            ns  = None if i==len(z_val)-1 else z_val[i+1]
            done= (i==len(z_val)-1)
            agent.store(s,a,r,ns,done)
            agent.learn()
        with torch.no_grad():
            qs = agent.net(torch.tensor(z_val,device=device).float()).cpu().numpy()
        pred = (qs[:,1]>qs[:,0]).astype(int)
        f1   = f1_score(y_val,pred)
        aupr = average_precision_score(y_val,qs[:,1]-qs[:,0])
        metrics.append((f1,aupr))
        print(f"AL{budget} Ep{ep} VAL f1 {f1:.3f} aupr {aupr:.3f}")
    pd.DataFrame(metrics,columns=["F1","AUPR"],index=range(1,episodes+1))\
      .to_csv(f"exp_AL{budget}/val_metrics.csv")
    # holdout eval
    with torch.no_grad():
        qs_o = agent.net(torch.tensor(z_hold,device=device).float()).cpu().numpy()
    pred_o = (qs_o[:,1]>qs_o[:,0]).astype(int)
    f1o = f1_score(y_hold,pred_o)
    aupo = average_precision_score(y_hold,qs_o[:,1]-qs_o[:,0])
    print(f"AL{budget} HOLD f1 {f1o:.3f} aupr {aupo:.3f}")
    pd.DataFrame([[f1o,aupo]],columns=["F1","AUPR"])\
      .to_csv(f"exp_AL{budget}/holdout_metrics.csv",index=False)
    plt.figure(figsize=(10,3))
    plt.plot(y_hold, label="GT")
    plt.plot(pred_o, label="Pred", alpha=0.7)
    plt.legend(); plt.title(f"AL{budget} Holdout")
    plt.savefig(f"exp_AL{budget}/holdout_pred.png"); plt.close()
