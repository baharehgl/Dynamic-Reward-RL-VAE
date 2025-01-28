import os

###############################################################################
# 1) Force GPU usage (if you have one). Must do this BEFORE any TF import.
###############################################################################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # or "0,1" if you have multiple GPUs

###############################################################################
# Now import TensorFlow in a TF1.x-compatible way
###############################################################################
import numpy as np
import pandas as pd
import random
import time
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# TensorFlow 1.x style:
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from collections import namedtuple
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.semi_supervised import LabelSpreading
from sklearn.svm import OneClassSVM

# If you have TF2.11+ and see "get_updates" error, use the legacy Keras optimizer:
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.legacy import Adam

###############################################################################
# Check GPU availability
###############################################################################
print("Is GPU available?:", tf.test.is_gpu_available(cuda_only=True))

###############################################################################
#                 Relative paths to your data directories
###############################################################################
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NORMAL_DATA_DIR = os.path.join(SCRIPT_DIR, "normal-data")
YAHOO_BASE_DIR = os.path.join(SCRIPT_DIR, "ydata-labeled-time-series-anomalies-v1_0")
DATASET_DIR = os.path.join(YAHOO_BASE_DIR, "A1Benchmark")

###############################################################################
#                   Global definitions for RL
###############################################################################
NOT_ANOMALY = 0
ANOMALY = 1
action_space = [NOT_ANOMALY, ANOMALY]
action_space_n = len(action_space)

TP_Value = 5
TN_Value = 1
FP_Value = -1
FN_Value = -5

validation_separate_ratio = 0.9
#n_steps = 25       # RNN sliding window
n_steps = 100
n_input_dim = 2    # input dimension per step
n_hidden_dim = 128 # hidden dimension for LSTM


###############################################################################
#                          Environment Definition
###############################################################################
REWARD_CORRECT = 1
REWARD_INCORRECT = -1

def defaultStateFuc(timeseries, timeseries_curser):
    return timeseries['value'][timeseries_curser]

def defaultRewardFuc(timeseries, timeseries_curser, action):
    if action == timeseries['anomaly'][timeseries_curser]:
        return REWARD_CORRECT
    else:
        return REWARD_INCORRECT


class EnvTimeSeriesfromRepo:
    """
    Environment that loads multiple time-series CSV files from a directory,
    each containing 'value' and 'anomaly'. We'll store each as a DataFrame:
        - value
        - anomaly
        - label = -1 initially
    """
    def __init__(self, repodir):
        self.repodir = repodir
        self.repodirext = []
        for subdir, dirs, files in os.walk(self.repodir):
            for file in files:
                if file.endswith('.csv'):
                    self.repodirext.append(os.path.join(subdir, file))

        self.action_space_n = len(action_space)

        self.timeseries = None
        self.timeseries_curser = -1
        self.timeseries_curser_init = 0
        self.timeseries_states = []

        self.statefnc = defaultStateFuc
        self.rewardfnc = defaultRewardFuc

        self.datasetsize = len(self.repodirext)
        self.datasetfix = 0
        self.datasetidx = random.randint(0, max(0, self.datasetsize - 1))
        self.datasetrng = self.datasetsize
        self.timeseries_repo = []
        self.states_list = []

        # Preload and scale all time-series
        for path in self.repodirext:
            ts = pd.read_csv(path, usecols=[1, 2], header=0, names=['value', 'anomaly'])
            # label col
            ts['label'] = -1
            ts = ts.astype(np.float32)

            # Scale the 'value' to [0,1] for a MSE+sigmoid-based VAE approach.
            scaler = MinMaxScaler(feature_range=(0,1))
            ts['value'] = scaler.fit_transform(ts[['value']])

            self.timeseries_repo.append(ts)

    def reset(self):
        if self.datasetfix == 0:
            self.datasetidx = (self.datasetidx + 1) % self.datasetrng
        print("Loading dataset:", self.repodirext[self.datasetidx])
        self.timeseries = self.timeseries_repo[self.datasetidx]
        self.timeseries_curser = self.timeseries_curser_init
        self.timeseries_states = self.statefnc(self.timeseries, self.timeseries_curser)
        self.states_list = self.get_states_list()
        return self.timeseries_states

    def reset_to(self, idx):
        self.datasetidx = idx
        self.timeseries = self.timeseries_repo[self.datasetidx]
        self.timeseries_curser = self.timeseries_curser_init
        self.timeseries_states = self.statefnc(self.timeseries, self.timeseries_curser)
        self.states_list = self.get_states_list()
        return self.timeseries_states

    def step(self, action):
        reward = self.rewardfnc(self.timeseries, self.timeseries_curser, action)
        self.timeseries_curser += 1
        if self.timeseries_curser >= len(self.timeseries['value']):
            done = 1
            # replicate last state
            state = np.array([self.timeseries_states, self.timeseries_states])
        else:
            done = 0
            state = self.statefnc(self.timeseries, self.timeseries_curser,
                                  self.timeseries_states, action)
        if isinstance(state, np.ndarray) and len(state.shape) > len(np.shape(self.timeseries_states)):
            # branching
            self.timeseries_states = state[action]
        else:
            self.timeseries_states = state
        return state, reward, done, {}

    def get_states_list(self):
        self.timeseries_curser = self.timeseries_curser_init
        out_list = []
        last_state = None
        for cursor in range(self.timeseries_curser_init, len(self.timeseries['value'])):
            if cursor == self.timeseries_curser_init:
                st = self.statefnc(self.timeseries, cursor)
            else:
                st = self.statefnc(self.timeseries, cursor, last_state, None)
                if isinstance(st, np.ndarray) and len(st.shape) == 3:
                    st = st[0]
            out_list.append(st)
            last_state = st
        return out_list


###############################################################################
#             RNN State function & VAE-based reward function
###############################################################################
def RNNBinaryStateFuc(timeseries, timeseries_curser, previous_state=[], action=None):
    """
    Returns an RNN-friendly state of shape (n_steps, n_input_dim).
    - Each row: [value_t, flag].
    - Flag is 0 or 1 depending on action in a branching scenario.
    """
    if timeseries_curser == n_steps:
        state = []
        for i in range(n_steps):
            state.append([timeseries['value'][i], 0])
        # pop the oldest row, add a row with flag=1
        state.pop(0)
        state.append([timeseries['value'][timeseries_curser], 1])
        return np.array(state, dtype='float32')

    if timeseries_curser > n_steps:
        state0 = np.concatenate((previous_state[1:], [[timeseries['value'][timeseries_curser], 0]]))
        state1 = np.concatenate((previous_state[1:], [[timeseries['value'][timeseries_curser], 1]]))
        return np.array([state0, state1], dtype='float32')

    return [timeseries['value'][timeseries_curser], 0]

tau_values = []

def kl_divergence(p, q):
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))

def adaptive_scaling_factor_dro(preference_strength, tau_min=0.1, tau_max=5.0, rho=0.5, max_iter=10):
    tau = 1.0
    for _ in range(max_iter):
        kl_term = kl_divergence([preference_strength, 1 - preference_strength], [0.5, 0.5])
        grad = -np.log(1 + np.exp(-preference_strength / tau)) + rho - kl_term
        hess = (preference_strength**2 * np.exp(-preference_strength / tau) /
                (tau**3 * (1 + np.exp(-preference_strength / tau))**2 + 1e-8))
        tau = tau - grad / (hess + 1e-8)
        tau = np.clip(tau, tau_min, tau_max)
    return tau

def RNNBinaryRewardFuc(timeseries, timeseries_curser, action=0, vae=None, scale_factor=10):
    """
    Returns [reward_if_action0, reward_if_action1] using a VAE-based reconstruction error + adaptive tau scaling.
    """
    if timeseries_curser >= n_steps:
        window_data = np.array([timeseries['value'][timeseries_curser - n_steps: timeseries_curser]])
        vae_recon = vae.predict(window_data)
        recon_error = np.mean((vae_recon - window_data)**2)

        vae_penalty = -scale_factor * recon_error
        preference_strength = np.clip(1 / (1 + np.exp(-recon_error)), 0.05, 0.95)
        tau = adaptive_scaling_factor_dro(preference_strength)
        tau_values.append(tau)

        lbl = timeseries['label'][timeseries_curser]
        if lbl == 0:
            return [
                tau * (TN_Value + vae_penalty),
                tau * (FP_Value + vae_penalty)
            ]
        elif lbl == 1:
            return [
                tau * (FN_Value + vae_penalty),
                tau * (TP_Value + vae_penalty)
            ]
    return [0, 0]

def RNNBinaryRewardFucTest(timeseries, timeseries_curser, action=0):
    """
    Reward for testing. Using 'anomaly' column as ground truth.
    """
    if timeseries_curser >= n_steps:
        if timeseries['anomaly'][timeseries_curser] == 0:
            return [TN_Value, FP_Value]
        else:
            return [FN_Value, TP_Value]
    return [0, 0]


###############################################################################
#                  VAE with MSE to Avoid NaNs
###############################################################################
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(original_dim=3, latent_dim=10, intermediate_dim=64):
    """
    MSE-based VAE to avoid NaN for generic data.
    """
    inputs = layers.Input(shape=(original_dim,))
    h = layers.Dense(intermediate_dim, activation='relu')(inputs)
    h = layers.Dense(intermediate_dim, activation='relu')(h)
    h = layers.Dense(intermediate_dim, activation='relu')(h)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    z = Sampling()([z_mean, z_log_var])

    # decoder
    decoder_h1 = layers.Dense(intermediate_dim, activation='relu')
    decoder_h2 = layers.Dense(intermediate_dim, activation='relu')
    decoder_h3 = layers.Dense(intermediate_dim, activation='relu')
    decoder_out = layers.Dense(original_dim, activation='sigmoid')  # or 'linear'

    h_decoded = decoder_h1(z)
    h_decoded = decoder_h2(h_decoded)
    h_decoded = decoder_h3(h_decoded)
    x_decoded_mean = decoder_out(h_decoded)

    encoder = models.Model(inputs, [z_mean, z_log_var, z])
    vae = models.Model(inputs, x_decoded_mean)

    # MSE reconstruction loss
    reconstruction_loss = tf.reduce_sum(tf.square(inputs - x_decoded_mean), axis=1)
    kl_loss = -0.5 * tf.reduce_sum(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
    )
    total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(total_loss)

    # Use the "legacy" Adam to avoid "get_updates" error in TF2.11+ with TF1.x
    opt = Adam(learning_rate=1e-4)
    vae.compile(optimizer=opt)
    return vae, encoder

def load_normal_data(data_path):
    all_files = [
        os.path.join(data_path, fname)
        for fname in os.listdir(data_path)
        if fname.endswith('.csv')
    ]
    if not all_files:
        raise FileNotFoundError(f"No CSV in {data_path}")

    data_list = [pd.read_csv(f) for f in all_files]
    data = pd.concat(data_list, axis=0, ignore_index=True).astype(np.float32)

    # Scale data into [0, 1] to match VAE's 'sigmoid' + MSE usage
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data.values)
    return scaled_data


###############################################################################
#          Q-network (RNN-based) with integer global_step
###############################################################################
class Q_Estimator_Nonlinear:
    """
    Approximates Q(s,a) for discrete actions via an LSTM (TF1.x).
    Input shape: (batch_size, n_steps, n_input_dim).
    """
    def __init__(self, learning_rate=3e-4, scope="Q_Estimator_Nonlinear", summaries_dir=None):
        self.scope = scope
        self.summary_writer = None

        with tf.variable_scope(scope):
            # placeholders
            self.state = tf.placeholder(
                shape=[None, n_steps, n_input_dim],
                dtype=tf.float32, name="state"
            )
            self.target = tf.placeholder(
                shape=[None, 2],
                dtype=tf.float32, name="target"
            )
            # LSTM
            lstm_cell = tf.nn.rnn_cell.LSTMCell(n_hidden_dim, forget_bias=1.0)
            state_unstack = tf.unstack(self.state, n_steps, axis=1)
            outputs, states_ = tf.nn.static_rnn(lstm_cell, state_unstack, dtype=tf.float32)

            # final linear
            W_out = tf.Variable(tf.random_normal([n_hidden_dim, 2]))
            b_out = tf.Variable(tf.random_normal([2]))
            self.action_values = tf.matmul(outputs[-1], W_out) + b_out

            # loss
            self.losses = tf.squared_difference(self.action_values, self.target)
            self.loss = tf.reduce_mean(self.losses)

            # Make sure global_step is int
            self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

            # Summaries
            self.summaries = tf.summary.merge([
                tf.summary.histogram("loss_hist", self.losses),
                tf.summary.scalar("loss", self.loss),
                tf.summary.histogram("q_values_hist", self.action_values),
                tf.summary.scalar("q_value_max", tf.reduce_max(self.action_values))
            ])
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, f"summaries_{scope}")
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_values, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        summaries, global_step_val, _ = sess.run(
            [self.summaries, self.global_step, self.train_op], feed_dict
        )
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step_val)


def copy_model_parameters(sess, estimator1, estimator2):
    e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        update_ops.append(e2_v.assign(e1_v))
    sess.run(update_ops)

def make_epsilon_greedy_policy(estimator, nA):
    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype=np.float32) * epsilon / nA
        q_values = estimator.predict([observation])
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


###############################################################################
#           Active Learning & Warm-Up
###############################################################################
class active_learning:
    def __init__(self, env, N, strategy, estimator, already_selected):
        self.env = env
        self.N = N
        self.strategy = strategy
        self.estimator = estimator
        self.already_selected = already_selected

    def get_samples(self):
        states_list = self.env.states_list
        distances = []
        for st in states_list:
            q = self.estimator.predict([st])[0]
            distances.append(abs(q[0] - q[1]))
        distances = np.array(distances)
        rank_ind = np.argsort(distances)  # margin
        rank_ind = [i for i in rank_ind if i not in self.already_selected]
        return rank_ind[:self.N]

    def label(self, active_samples):
        for smp in active_samples:
            print("Please label sample:", smp)
            label = input()
            self.env.timeseries.loc[smp + n_steps - 1, 'anomaly'] = label


class WarmUp:
    def warm_up_isolation_forest(self, outliers_fraction, X):
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(contamination=outliers_fraction, random_state=42)
        clf.fit(X)
        return clf


###############################################################################
#               Q-Learning Off-Policy with Replay
###############################################################################
def q_learning(env,
               sess,
               qlearn_estimator,
               target_estimator,
               num_episodes=300,
               num_epoches=10,
               replay_memory_size=500000,
               replay_memory_init_size=1500,
               experiment_dir='./exp/',
               update_target_estimator_every=10,
               discount_factor=0.99,
               epsilon_start=1.0,
               epsilon_end=0.1,
               epsilon_decay_steps=500000,
               batch_size=256,
               num_LabelPropagation=20,
               num_active_learning=5,
               test=0,
               vae_model=None):

    Transition = namedtuple("Transition", ["state", "reward", "next_state", "done"])
    replay_memory = []

    # Checkpoints
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver = tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading from checkpoint:", latest_checkpoint)
        saver.restore(sess, latest_checkpoint)
        if test:
            return

    total_t = 0

    # Epsilon schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    policy = make_epsilon_greedy_policy(qlearn_estimator, env.action_space_n)

    # Warm-up with e.g. IsolationForest
    print("Populating replay memory with warm-up...")
    t0 = time.time()
    data_train = []
    for ds_idx in range(env.datasetsize):
        env.reset_to(ds_idx)
        data_train.extend(env.states_list)

    numeric_data = []
    for st in data_train:
        # pick last row's value if st is (n_steps, 2)
        if isinstance(st, np.ndarray) and st.ndim==2 and st.shape[0]==n_steps:
            numeric_data.append(st[-1, 0])
        else:
            # fallback if st is just a single [val, 0]
            numeric_data.append(st[0] if isinstance(st,list) else st)

    numeric_data = np.array(numeric_data).reshape(-1,1)
    outliers_fraction = 0.01
    model = WarmUp().warm_up_isolation_forest(outliers_fraction, numeric_data)
    anomaly_score = model.decision_function(numeric_data)
    sorted_idx = np.argsort(anomaly_score)
    warm_samples = np.concatenate([sorted_idx[:5], sorted_idx[-5:]])
    labeled_count = 0

    current_ds = env.datasetidx
    for widx in warm_samples:
        env.reset_to(current_ds)
        if widx >= len(env.states_list):
            continue
        stt = env.states_list[widx]
        env.timeseries_curser = widx + n_steps
        if env.timeseries_curser < len(env.timeseries):
            env.timeseries['label'][env.timeseries_curser] = env.timeseries['anomaly'][env.timeseries_curser]
            labeled_count += 1
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
            action_probs = policy(stt, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)
            replay_memory.append(Transition(stt, reward, next_state, done))

    print(f"Warm-up done. Labeled {labeled_count} samples in {time.time()-t0:.2f}s.")
    print(f"Replay memory size: {len(replay_memory)}")

    lp_model = LabelSpreading()

    for i_episode in range(num_episodes):
        if i_episode % 50 == 49:
            print(f"Saving checkpoint at episode {i_episode+1}")
            saver.save(sess, checkpoint_path)

        t1 = time.time()
        state = env.reset()
        while env.datasetidx > env.datasetrng*validation_separate_ratio:
            state = env.reset()

        # Active learning
        ds_idx = env.datasetidx
        labeled_index = []
        for k in range(n_steps, len(env.timeseries)):
            if env.timeseries['label'][k] != -1:
                labeled_index.append(k - n_steps)

        # margin sampling
        al = active_learning(env, num_active_learning, 'margin_sampling',
                             qlearn_estimator, labeled_index)
        al_samples = al.get_samples()

        for smp in al_samples:
            env.timeseries_curser = smp + n_steps
            env.timeseries['label'][env.timeseries_curser] = env.timeseries['anomaly'][env.timeseries_curser]
            st_al = env.states_list[smp]
            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]
            action_probs = policy(st_al, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)
            replay_memory.append(Transition(st_al, reward, next_state, done))

        # Label propagation
        states_mat = np.array(env.states_list, dtype=object)
        compressed = []
        labels_lp = []
        for s_i, stt2 in enumerate(states_mat):
            if isinstance(stt2, np.ndarray) and stt2.ndim==2 and stt2.shape[0]==n_steps:
                compressed.append(stt2[-1])
            else:
                compressed.append(stt2 if isinstance(stt2,list) else [0,0])
            lbl = -1
            if (s_i+n_steps)<len(env.timeseries):
                lbl = env.timeseries['label'][s_i+n_steps]
            labels_lp.append(lbl)

        compressed = np.array(compressed, dtype=np.float32)
        labels_lp = np.array(labels_lp)
        unlabeled_indices = np.where(labels_lp==-1)[0]
        if len(unlabeled_indices)>0:
            lp_model.fit(compressed, labels_lp)
            pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)
            certainty_index = np.argsort(pred_entropies)
            certainty_index = [ci for ci in certainty_index if ci in unlabeled_indices]
            certainty_index = certainty_index[:num_LabelPropagation]
            for ci in certainty_index:
                pseudo_label = lp_model.transduction_[ci]
                true_idx = ci + n_steps
                if true_idx<len(env.timeseries):
                    env.timeseries['label'][true_idx] = pseudo_label

        t2 = time.time()

        # Train the Q network
        for _ in range(num_epoches):
            if len(replay_memory)<batch_size:
                break

            epsilon = epsilons[min(total_t, epsilon_decay_steps-1)]
            samples = random.sample(replay_memory, batch_size)
            states_batch, rewards_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
            # shape: next_states_batch -> (batch_size, 2, n_steps, n_input_dim)
            next_states_batch0 = next_states_batch[:,0]
            next_states_batch1 = next_states_batch[:,1]

            q_values_next0 = target_estimator.predict(next_states_batch0, sess=sess)
            q_values_next1 = target_estimator.predict(next_states_batch1, sess=sess)
            next_q0_max = np.amax(q_values_next0, axis=1)
            next_q1_max = np.amax(q_values_next1, axis=1)

            targets_batch = rewards_batch + discount_factor * np.stack((next_q0_max, next_q1_max), axis=-1)
            qlearn_estimator.update(states_batch, targets_batch.astype(np.float32), sess=sess)
            total_t += 1

            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, qlearn_estimator, target_estimator)
                print("Copied parameters to target network.")

        t3 = time.time()
        print(f"Episode {i_episode+1}/{num_episodes}, Global step={total_t}, Time: {(t2-t1):.2f}+{(t3-t2):.2f} s")


###############################################################################
#                Evaluation
###############################################################################
def evaluate_model(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    mcc = matthews_corrcoef(y_true, y_pred)
    if (tn+fp)>0:
        balanced_acc = (recall + (tn/(tn+fp)))/2
    else:
        balanced_acc = 0
    fpr = fp/(fp+tn) if (fp+tn)>0 else 0
    fnr = fn/(fn+tp) if (fn+tp)>0 else 0

    return {
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "MCC": mcc,
        "Balanced Accuracy": balanced_acc,
        "FPR": fpr,
        "FNR": fnr
    }

def q_learning_validator(env, estimator, num_episodes, record_dir=None, plot=1):
    y_true_all = []
    y_pred_all = []
    for i_episode in range(num_episodes):
        policy = make_epsilon_greedy_policy(estimator, env.action_space_n)
        st = env.reset()
        while env.datasetidx < env.datasetrng * validation_separate_ratio:
            st = env.reset()

        done = False
        while not done:
            action_probs = policy(st, 0)  # greedy
            action = np.argmax(action_probs)
            if env.timeseries_curser < len(env.timeseries):
                y_true_all.append(env.timeseries['anomaly'][env.timeseries_curser])
                y_pred_all.append(action)
            next_state, reward, done, _ = env.step(action)
            if not done:
                st = next_state[action]

    results = evaluate_model(y_true_all, y_pred_all)
    for k,v in results.items():
        print(k,":",v)
    return results


###############################################################################
#                     Main Training Function
###############################################################################
def train(num_LP=100, num_AL=30, discount_factor=0.92, learn_tau=True):
    # 1) Build VAE before launching session
    x_train = load_normal_data(NORMAL_DATA_DIR)
    vae, encoder = build_vae(original_dim=x_train.shape[1], latent_dim=10, intermediate_dim=64)

    # 2) Fit VAE with MSE-based reconstruction
    vae.fit(x_train, epochs=50, batch_size=32)  # If you see warnings about missing outputs, that's normal w/ add_loss

    # 3) Save & reload VAE (optional, but consistent with your earlier code)
    vae.save("vae_model.h5")
    from tensorflow.keras.models import load_model
    vae_loaded = load_model("vae_model.h5", custom_objects={"Sampling": Sampling}, compile=False)

    # 4) Prepare environment
    env = EnvTimeSeriesfromRepo(DATASET_DIR)
    env.statefnc = RNNBinaryStateFuc
    if learn_tau:
        env.rewardfnc = lambda ts,c,a: RNNBinaryRewardFuc(ts,c,a,vae=vae_loaded)
    else:
        env.rewardfnc = lambda ts,c,a: RNNBinaryRewardFuc(ts,c,a,vae=vae_loaded, scale_factor=10)
    env.timeseries_curser_init = n_steps
    env.datasetfix = 0
    env.datasetidx = 0

    # For testing
    env_test = env
    env_test.rewardfnc = RNNBinaryRewardFucTest

    exp_dir = os.path.abspath("./exp/learn_tau_experiment/")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    tf.reset_default_graph()

    # 5) Create Q-learn & target nets
    qlearn_estimator = Q_Estimator_Nonlinear(scope="qlearn", summaries_dir=exp_dir, learning_rate=3e-4)
    target_estimator = Q_Estimator_Nonlinear(scope="target")

    # 6) TF1.x Session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        q_learning(env,
                   sess,
                   qlearn_estimator,
                   target_estimator,
                   num_episodes=300,
                   num_epoches=10,
                   experiment_dir=exp_dir,
                   replay_memory_size=500000,
                   replay_memory_init_size=1500,
                   update_target_estimator_every=10,
                   epsilon_start=1,
                   epsilon_end=0.1,
                   epsilon_decay_steps=500000,
                   discount_factor=discount_factor,
                   batch_size=256,
                   num_LabelPropagation=num_LP,
                   num_active_learning=num_AL,
                   test=0,
                   vae_model=vae_loaded)

        # Evaluate on leftover "test"
        num_test_episodes = int(env.datasetsize*(1-validation_separate_ratio))
        stats = q_learning_validator(env_test, qlearn_estimator, num_test_episodes, exp_dir)
    return stats

def plot_tau_evolution():
    plt.figure(figsize=(8,5))
    plt.plot(tau_values, label="Tau over time", alpha=0.7)
    plt.xlabel("Training Steps")
    plt.ylabel("Tau")
    plt.title("Adaptive Tau Evolution")
    plt.legend()
    plt.show()

###############################################################################
#                              Main Entry
###############################################################################
if __name__ == "__main__":
    print("=== Starting Training with MSE-based VAE & GPU check ===")
    # Example usage
    #train(100, 30, discount_factor=0.92, learn_tau=True)
    #train(150, 50, discount_factor=0.94, learn_tau=True)
    train(200, 100, discount_factor=0.96, learn_tau=True)

    plot_tau_evolution()
    print("Done.")
