# Adapted anomaly detection code for WaDi dataset (originally for SMD)
# =========================================================

# 1. Data Loading and Preprocessing
# ---------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# File paths for WaDi dataset
train_path = 'WaDi/WADI_14days_new.csv'
test_path = 'WaDi/WADI_attackdataLABLE.csv'

# Load training data (unlabeled sensor data)
train_df = pd.read_csv(train_path)
# Remove any label column if present (not expected in training data)
if 'Attack LABLE' in train_df.columns:
    train_df.drop(columns=['Attack LABLE'], inplace=True)
# Remove non-sensor columns (e.g., timestamp)
for col in list(train_df.columns):
    if col.lower().startswith('time') or col.lower().startswith('date'):
        train_df.drop(columns=[col], inplace=True)

# Load test data (sensor data with attack labels)
test_df = pd.read_csv(test_path)
# Identify the label column (should contain 'label' in name)
label_col = None
for col in test_df.columns:
    if 'label' in col.lower():
        label_col = col
        break
if label_col is None:
    raise ValueError("Label column not found in test dataset.")
# Convert labels: 1 (no attack) -> 0 (normal), -1 (attack) -> 1 (anomaly)
test_df[label_col] = test_df[label_col].apply(lambda x: 0 if x == 1 else 1)
# Separate features and labels
y_test = test_df[label_col].values
X_test_df = test_df.drop(columns=[label_col])
# Remove non-sensor columns from test data (e.g., timestamp)
for col in list(X_test_df.columns):
    if col.lower().startswith('time') or col.lower().startswith('date'):
        X_test_df.drop(columns=[col], inplace=True)
# Align test features with train features (same columns and order)
X_test_df = X_test_df[train_df.columns]

# Convert features to numpy arrays
X_train = train_df.values.astype(np.float32)
X_test = X_test_df.values.astype(np.float32)

# Scale features (fit on train, apply to both train and test)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. VAE Model Definition and Training
# ---------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Define Variational Autoencoder (VAE) for multivariate data
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10, hidden_dim=64):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # sample z
    def decode(self, z):
        h = F.relu(self.fc2(z))
        return self.fc3(h)
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

# Initialize VAE and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae = VAE(input_dim=X_train.shape[1]).to(device)
optimizer_vae = optim.Adam(vae.parameters(), lr=1e-3)

# Train VAE on normal training data
num_epochs = 10
batch_size = 64
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
train_loader = torch.utils.data.DataLoader(X_train_tensor, batch_size=batch_size, shuffle=True)
vae.train()
for epoch in range(num_epochs):
    total_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer_vae.zero_grad()
        recon_batch, mu, logvar = vae(batch)
        # VAE loss = reconstruction loss + KL divergence
        recon_loss = F.mse_loss(recon_batch, batch, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + kld_loss
        loss.backward()
        optimizer_vae.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(X_train_scaled)
    print(f"Epoch {epoch+1}/{num_epochs}, VAE Loss: {avg_loss:.4f}")

# 3. Feature Extraction on Test Data (Reconstruction Errors)
# ---------------------------------------------------------
vae.eval()
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
recon_errors = []
with torch.no_grad():
    batch_size_test = 256
    for i in range(0, X_test_tensor.shape[0], batch_size_test):
        batch = X_test_tensor[i:i+batch_size_test]
        recon_batch, mu, logvar = vae(batch)
        # Compute MSE recon error for each sample
        errors = F.mse_loss(recon_batch, batch, reduction='none').mean(dim=1)
        recon_errors.append(errors.cpu().numpy())
recon_errors = np.concatenate(recon_errors)  # array of reconstruction errors per test sample

# 4. Reinforcement Learning Environment and Agent Setup
# ---------------------------------------------------------
# Define anomaly detection environment for RL
class AnomalyEnv:
    def __init__(self, errors, labels):
        self.errors = errors
        self.labels = labels.astype(int)
        self.n = len(errors)
        self.index = 0
    def reset(self):
        self.index = 0
        return None if self.n == 0 else float(self.errors[self.index])
    def step(self, action):
        # action: 0 = predict normal, 1 = predict anomaly
        actual = self.labels[self.index]
        # Dynamic reward: +1 for TP, 0 for TN, -1 for FP, -1 for FN
        if action == 1:  # flagged as anomaly
            reward = 1.0 if actual == 1 else -1.0
        else:  # predicted normal
            reward = -1.0 if actual == 1 else 0.0
        # move to next time step
        self.index += 1
        done = (self.index >= self.n)
        next_state = float(self.errors[self.index]) if not done else None
        return next_state, reward, done

# Define Q-network for RL agent (inputs state=error, outputs Q for two actions)
class QNetwork(nn.Module):
    def __init__(self, state_dim=1, action_dim=2, hidden_dim=16):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Initialize environment and Q-network
env = AnomalyEnv(recon_errors, y_test)
q_net = QNetwork().to(device)
optimizer_q = optim.Adam(q_net.parameters(), lr=1e-3)

# 5. Reinforcement Learning Training (Q-learning with experience replay)
# ---------------------------------------------------------
gamma = 0.99
num_episodes = 20
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = (epsilon_start - epsilon_end) / num_episodes
memory = deque(maxlen=10000)
batch_size_rl = 32

for episode in range(num_episodes):
    state_val = env.reset()
    total_reward = 0.0
    # Decay epsilon linearly over episodes
    epsilon = max(epsilon_end, epsilon_start - episode * epsilon_decay)
    done = False
    while not done and state_val is not None:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            state_tensor = torch.tensor([[state_val]], dtype=torch.float32).to(device)
            with torch.no_grad():
                q_values = q_net(state_tensor)
                action = int(torch.argmax(q_values).item())
        # Take action and observe reward
        next_state_val, reward, done = env.step(action)
        total_reward += reward
        # Store transition in replay memory
        memory.append((state_val, action, reward, next_state_val, done))
        # Update current state
        state_val = next_state_val
        # Train Q-network on a random batch from memory (experience replay)
        if len(memory) >= batch_size_rl:
            batch = random.sample(memory, batch_size_rl)
            # Prepare batch data
            state_batch = torch.tensor([[b[0]] for b in batch], dtype=torch.float32).to(device)
            action_batch = torch.tensor([b[1] for b in batch], dtype=torch.int64).to(device)
            reward_batch = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(device)
            next_state_batch = [b[3] for b in batch]
            done_batch = [b[4] for b in batch]
            # Compute current Q values for each state-action pair
            q_values = q_net(state_batch)
            curr_q = q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
            # Compute target Q values
            target_q = torch.zeros(batch_size_rl, dtype=torch.float32).to(device)
            for i in range(batch_size_rl):
                if done_batch[i] or next_state_batch[i] is None:
                    target_q[i] = reward_batch[i]
                else:
                    ns_tensor = torch.tensor([[next_state_batch[i]]], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        next_q_val = q_net(ns_tensor).max().item()
                    target_q[i] = reward_batch[i] + gamma * next_q_val
            # Update Q-network
            loss_q = F.mse_loss(curr_q, target_q)
            optimizer_q.zero_grad()
            loss_q.backward()
            optimizer_q.step()
    print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

# 6. Evaluation of the Trained Model
# ---------------------------------------------------------
env.reset()
predictions = []
for err in env.errors:
    state_tensor = torch.tensor([[err]], dtype=torch.float32).to(device)
    with torch.no_grad():
        q_values = q_net(state_tensor)
        action = int(torch.argmax(q_values).item())
    predictions.append(action)
predictions = np.array(predictions)
actual = env.labels

# Calculate metrics
TP = np.sum((predictions == 1) & (actual == 1))
FP = np.sum((predictions == 1) & (actual == 0))
FN = np.sum((predictions == 0) & (actual == 1))
TN = np.sum((predictions == 0) & (actual == 0))
precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)
print("Final Evaluation on Test Data:")
print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
