import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import random
import sys
import os
import time
from scipy import stats
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

# Enable GPUs "0" and "1" if available
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

tf.compat.v1.disable_eager_execution()

from mpl_toolkits.mplot3d import axes3d
from collections import deque, namedtuple
from tensorflow.keras import layers, models, losses
from tensorflow.keras.models import load_model
from sklearn.svm import OneClassSVM
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

# ------------------------------------------------------------------------------------
#                               Environment Code
# ------------------------------------------------------------------------------------
NOT_ANOMALY = 0
ANOMALY = 1

REWARD_CORRECT = 1
REWARD_INCORRECT = -1

action_space = [NOT_ANOMALY, ANOMALY]


def defaultStateFuc(timeseries, timeseries_curser):
    return timeseries['value'][timeseries_curser]

def defaultRewardFuc(timeseries, timeseries_curser, action):
    if action == timeseries['anomaly'][timeseries_curser]:
        return REWARD_CORRECT
    else:
        return REWARD_INCORRECT


class EnvTimeSeriesfromRepo():
    """
    Environment that loads multiple time-series CSV files from a directory,
    each containing 'value' and 'anomaly' columns.
    """

    def __init__(self, repodir='environment/time_series_repo/'):

        # gather all CSV file paths
        self.repodir = repodir
        self.repodirext = []
        for subdir, dirs, files in os.walk(self.repodir):
            for file in files:
                if file.endswith('.csv'):
                    self.repodirext.append(os.path.join(subdir, file))

        self.action_space_n = len(action_space)

        self.timeseries = []
        self.timeseries_curser = -1
        self.timeseries_curser_init = 0
        self.timeseries_states = []

        # by default, environment calls these. We'll override them externally:
        self.statefnc = defaultStateFuc
        self.rewardfnc = defaultRewardFuc

        self.datasetsize = len(self.repodirext)
        self.datasetfix = 0
        # pick random series initially
        self.datasetidx = random.randint(0, len(self.repodirext) - 1)
        self.datasetrng = self.datasetsize

        self.timeseries_repo = []
        self.states_list = []

        # Preload and scale all time-series
        for i in range(len(self.repodirext)):
            # For Yahoo data: columns [1,2] => 'value','anomaly'
            ts = pd.read_csv(self.repodirext[i], usecols=[1, 2], header=0, names=['value', 'anomaly'])

            # Insert an extra label column = -1 => unlabeled initially
            ts['label'] = -1
            ts = ts.astype(np.float32)

            # scale the 'value' column
            scaler = StandardScaler()
            ts['value'] = scaler.fit_transform(ts[['value']])

            self.timeseries_repo.append(ts)

    def reset(self):
        """
        Resets environment by picking the next time-series,
        sets self.timeseries_curser to timeseries_curser_init,
        and returns the first 'state'.
        """
        if self.datasetfix == 0:
            self.datasetidx = (self.datasetidx + 1) % self.datasetrng
        print("Loading dataset:", self.repodirext[self.datasetidx])
        self.timeseries = self.timeseries_repo[self.datasetidx]
        self.timeseries_curser = self.timeseries_curser_init

        self.timeseries_states = self.statefnc(self.timeseries, self.timeseries_curser)
        self.states_list = self.get_states_list()

        return self.timeseries_states

    def reset_to(self, idx):
        """
        Force the environment to reset to a specific dataset index 'idx'.
        """
        self.datasetidx = idx
        self.timeseries = self.timeseries_repo[self.datasetidx]
        self.timeseries_curser = self.timeseries_curser_init

        self.timeseries_states = self.statefnc(self.timeseries, self.timeseries_curser)
        self.states_list = self.get_states_list()
        return self.timeseries_states

    def reset_getall(self):
        """
        Loads a raw version of the next dataset (with minimal transformations).
        Primarily for analysis or debugging.
        """
        if self.datasetfix == 0:
            self.datasetidx = (self.datasetidx + 1) % self.datasetrng
        ts = pd.read_csv(self.repodirext[self.datasetidx],
                         usecols=[0, 1, 2],
                         header=0,
                         names=['timestamp', 'value', 'anomaly'])
        ts = ts.astype(np.float32)

        # scale the 'value'
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        ts['value'] = scaler.fit_transform(ts[['value']])
        self.timeseries = ts
        self.timeseries_curser = self.timeseries_curser_init
        return self.timeseries

    def step(self, action):
        """
        Applies the chosen action => returns (next_state, reward, done, info).
        """
        reward = self.rewardfnc(self.timeseries, self.timeseries_curser, action)

        self.timeseries_curser += 1
        if self.timeseries_curser >= len(self.timeseries['value']):
            # we've reached the end of the time-series
            done = 1
            # just replicate the last state
            state = np.array([self.timeseries_states, self.timeseries_states])
        else:
            done = 0
            state = self.statefnc(self.timeseries, self.timeseries_curser,
                                  self.timeseries_states, action)

        # If it's a branching state (shape?), pick the next sub-state
        if len(np.shape(state)) > len(np.shape(self.timeseries_states)):
            self.timeseries_states = state[action]
        else:
            self.timeseries_states = state

        return state, reward, done, {}

    def get_states_list(self):
        """
        For convenience, returns a list of states from 'timeseries_curser_init'
        all the way to the end.
        """
        self.timeseries_curser = self.timeseries_curser_init
        out_list = []
        # We'll keep track of the last state so that we can pass it to statefnc
        last_state = None
        for cursor in range(self.timeseries_curser_init, len(self.timeseries['value'])):
            if cursor == self.timeseries_curser_init:
                state = self.statefnc(self.timeseries, cursor)
            else:
                state = self.statefnc(self.timeseries, cursor, last_state, None)
                # some custom state functions might return a list of states
                # if it branches; pick index 0 for consistency
                if isinstance(state, np.ndarray) and len(state.shape) == 3:
                    state = state[0]
            out_list.append(state)
            last_state = state
        return out_list

# ------------------------------------------------------------------------------------
#                             RL + VAE Code
# ------------------------------------------------------------------------------------

NOT_ANOMALY = 0
ANOMALY = 1
action_space = [NOT_ANOMALY, ANOMALY]
action_space_n = len(action_space)

# You can adjust these reward values:
TP_Value = 5
TN_Value = 1
FP_Value = -1
FN_Value = -5

validation_separate_ratio = 0.9

# The sliding window size for RNN states:
n_steps = 25
# dimension of the input each step
n_input_dim = 2
# dimension of the hidden state in LSTM cell
n_hidden_dim = 128

# Path to (normal) data used for training the VAE
data_directory = r"C:\Users\Asus\Documents\PSU-Course\sbsplusplus-master\normal-data"


def load_normal_data(data_path):
    """Load and scale normal data from CSVs in 'data_path' directory."""
    all_files = [
        os.path.join(data_path, fname)
        for fname in os.listdir(data_path)
        if fname.endswith('.csv')
    ]
    data_list = [pd.read_csv(file) for file in all_files]
    data = pd.concat(data_list, axis=0, ignore_index=True)
    # Scale
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values)
    return scaled_data


class Sampling(layers.Layer):
    """Keras layer that samples (z_mean + exp(0.5*z_log_var)*epsilon)."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae(original_dim, latent_dim=2, intermediate_dim=64):
    """
    Build a simple VAE model:
    - Encoder: Dense -> Dense -> z_mean, z_log_var, sample z
    - Decoder: Reconstruct input dimension with a few dense layers
    """
    inputs = layers.Input(shape=(original_dim,))
    h = layers.Dense(intermediate_dim, activation='relu')(inputs)
    h = layers.Dense(intermediate_dim, activation='relu')(h)
    h = layers.Dense(intermediate_dim, activation='relu')(h)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    z = Sampling()([z_mean, z_log_var])

    decoder_h1 = layers.Dense(intermediate_dim, activation='relu')
    decoder_h2 = layers.Dense(intermediate_dim, activation='relu')
    decoder_h3 = layers.Dense(intermediate_dim, activation='relu')
    decoder_out = layers.Dense(original_dim, activation='sigmoid')

    h_decoded = decoder_h1(z)
    h_decoded = decoder_h2(h_decoded)
    h_decoded = decoder_h3(h_decoded)
    x_decoded_mean = decoder_out(h_decoded)

    # Models
    encoder = models.Model(inputs, [z_mean, z_log_var, z])
    vae = models.Model(inputs, x_decoded_mean)

    # VAE loss
    reconstruction_loss = losses.binary_crossentropy(inputs, x_decoded_mean)
    # multiply by dimension
    reconstruction_loss = reconstruction_loss * original_dim
    kl_loss = -0.5 * tf.reduce_sum(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
    )
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    return vae, encoder


# ------------------------------------------------------------------------------------
#           RNN State and Reward
# ------------------------------------------------------------------------------------
def RNNBinaryStateFuc(timeseries, timeseries_curser, previous_state=[], action=None):
    """
    Returns an RNN-friendly state of shape (n_steps, n_input_dim).
    Each row: [value_t, flag], where 'flag' can be 0 or 1 depending on action.
    """
    if timeseries_curser == n_steps:
        # first time we have a full window => set the last row's flag=1, others 0
        state = []
        for i in range(timeseries_curser):
            state.append([timeseries['value'][i], 0])
        # pop the oldest, push a 1
        state.pop(0)
        state.append([timeseries['value'][timeseries_curser], 1])
        return np.array(state, dtype='float32')

    if timeseries_curser > n_steps:
        # shift the window by 1
        state0 = np.concatenate((
            previous_state[1:n_steps],
            [[timeseries['value'][timeseries_curser], 0]]
        ))
        state1 = np.concatenate((
            previous_state[1:n_steps],
            [[timeseries['value'][timeseries_curser], 1]]
        ))
        return np.array([state0, state1], dtype='float32')


# Rewards used during training
TP_Value = 5
TN_Value = 1
FP_Value = -1
FN_Value = -5

def kl_divergence(p, q):
    """Compute the KL divergence KL(p || q)."""
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))

def adaptive_scaling_factor_dro(preference_strength, tau_min=0.1, tau_max=5.0, rho=0.5, max_iter=10):
    """
    Learn tau dynamically using a simple DRO (KL-constrained) style approach.
    This is a toy function for demonstration.
    """
    tau = 1.0
    for _ in range(max_iter):
        # compute a KL term with a 50-50 baseline
        kl_term = kl_divergence([preference_strength, 1 - preference_strength],[0.5, 0.5])
        # gradient
        grad = -np.log(1 + np.exp(-preference_strength / tau)) + rho - kl_term
        # approximate second derivative
        hess = preference_strength ** 2 * np.exp(-preference_strength / tau) / (
            tau**3 * (1 + np.exp(-preference_strength / tau))**2 + 1e-8
        )
        # update
        tau = tau - grad/(hess + 1e-8)
        tau = np.clip(tau, tau_min, tau_max)
    return tau

tau_values = []

def RNNBinaryRewardFuc(timeseries, timeseries_curser, action=0, vae=None, scale_factor=10):
    """
    Return [reward_if_action0, reward_if_action1].
    We incorporate the VAE reconstruction error and an adaptive scaling factor 'tau'.
    """
    if timeseries_curser >= n_steps:
        # build the window input to VAE
        # (this example just uses the last n_steps values from 'value')
        window_data = np.array([timeseries['value'][timeseries_curser - n_steps: timeseries_curser]])
        vae_recon = vae.predict(window_data)
        recon_error = np.mean(np.square(vae_recon - window_data))

        vae_penalty = -scale_factor * recon_error

        # approximate preference strength in [0,1] from recon error
        preference_strength = np.clip(1 / (1 + np.exp(-recon_error)), 0.05, 0.95)
        tau = adaptive_scaling_factor_dro(preference_strength)
        tau_values.append(tau)

        # check ground truth
        if timeseries['label'][timeseries_curser] == 0:
            # normal
            return [
                tau * (TN_Value + vae_penalty),  # if action=0
                tau * (FP_Value + vae_penalty)   # if action=1
            ]
        if timeseries['label'][timeseries_curser] == 1:
            # anomaly
            return [
                tau * (FN_Value + vae_penalty),  # if action=0
                tau * (TP_Value + vae_penalty)   # if action=1
            ]
    else:
        return [0, 0]

def RNNBinaryRewardFucTest(timeseries, timeseries_curser, action=0):
    """
    Rewards used only for testing. We assume ground truth is in 'anomaly' column.
    """
    if timeseries_curser >= n_steps:
        if timeseries['anomaly'][timeseries_curser] == 0:
            return [TN_Value, FP_Value]
        else:
            return [FN_Value, TP_Value]
    else:
        return [0, 0]


# ------------------------------------------------------------------------------------
#            Q_Estimator for RNN
# ------------------------------------------------------------------------------------
class Q_Estimator_Nonlinear():
    """
    Approximates Q(s,a) for discrete actions via an LSTM.
    Input shape: (batch_size, n_steps, n_input_dim).
    Output shape: Q(s, [0,1]) for each example in the batch.
    """

    def __init__(self, learning_rate=np.float32(0.01), scope="Q_Estimator_Nonlinear", summaries_dir=None):
        self.scope = scope
        self.summary_writer = None

        with tf.compat.v1.variable_scope(scope):
            self.state = tf.compat.v1.placeholder(shape=[None, n_steps, n_input_dim],
                                                  dtype=tf.float32,
                                                  name="state")
            self.target = tf.compat.v1.placeholder(shape=[None, action_space_n],
                                                   dtype=tf.float32,
                                                   name="target")

            # LSTM
            lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(n_hidden_dim, forget_bias=1.0)
            # unstack each time-step
            state_unstack = tf.unstack(self.state, n_steps, axis=1)
            outputs, states = tf.compat.v1.nn.static_rnn(lstm_cell, state_unstack, dtype=tf.float32)

            # final dense
            W_out = tf.Variable(tf.compat.v1.random_normal([n_hidden_dim, action_space_n]))
            b_out = tf.Variable(tf.compat.v1.random_normal([action_space_n]))

            # Q(s) = last output * W + b
            self.action_values = tf.matmul(outputs[-1], W_out) + b_out

            # MSE loss
            self.losses = tf.compat.v1.squared_difference(self.action_values, self.target)
            self.loss = tf.reduce_mean(self.losses)

            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)

            # Summaries
            self.summaries = tf.compat.v1.summary.merge([
                tf.compat.v1.summary.histogram("loss_hist", self.losses),
                tf.compat.v1.summary.scalar("loss", self.loss),
                tf.compat.v1.summary.histogram("q_values_hist", self.action_values),
                tf.compat.v1.summary.scalar("q_value_max", tf.reduce_max(self.action_values))
            ])
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.compat.v1.summary.FileWriter(summary_dir)

    def predict(self, state, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        return sess.run(self.action_values, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        summaries, _, _ = sess.run([self.summaries, self.train_op, self.loss], feed_dict)
        if self.summary_writer:
            global_step = tf.compat.v1.train.global_step(sess, tf.compat.v1.train.get_global_step())
            self.summary_writer.add_summary(summaries, global_step)


def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copy parameters from estimator1 to estimator2.
    """
    e1_params = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)
    sess.run(update_ops)

def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy for Q(s,a).
    """
    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype='float32') * epsilon / nA
        q_values = estimator.predict(state=[observation])
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

# ------------------------------------------------------------------------------------
#                    Active Learning / Warm-Up Helpers
# ------------------------------------------------------------------------------------
class active_learning(object):
    """
    Simple demonstration for 'margin sampling' style AL.
    It picks N states with the smallest margin between Q(s,a=0) and Q(s,a=1).
    """
    def __init__(self, env, N, strategy, estimator, already_selected):
        self.env = env
        self.N = N
        self.strategy = strategy
        self.estimator = estimator
        self.already_selected = already_selected

    def get_samples(self):
        states_list = self.env.states_list
        distances = []
        for state in states_list:
            q = self.estimator.predict(state=[state])[0]
            distances.append(abs(q[0] - q[1]))
        distances = np.array(distances)

        if len(distances.shape) < 2:
            min_margin = abs(distances)
        else:
            # If it's multi-dim, pick difference of largest two
            sort_distances = np.sort(distances, axis=1)[:, -2:]
            min_margin = sort_distances[:, 1] - sort_distances[:, 0]

        rank_ind = np.argsort(min_margin)
        # remove already selected
        rank_ind = [i for i in rank_ind if i not in self.already_selected]
        active_samples = rank_ind[0:self.N]
        return active_samples

    def get_samples_by_score(self, threshold):
        states_list = self.env.states_list
        distances = []
        for state in states_list:
            q = self.estimator.predict(state=[state])[0]
            distances.append(abs(q[0] - q[1]))
        distances = np.array(distances)
        if len(distances.shape) < 2:
            min_margin = abs(distances)
        else:
            sort_distances = np.sort(distances, axis=1)[:, -2:]
            min_margin = sort_distances[:, 1] - sort_distances[:, 0]

        rank_ind = np.argsort(min_margin)
        rank_ind = [i for i in rank_ind if i not in self.already_selected]
        active_samples = [t for t in rank_ind if distances[t] < threshold]
        return active_samples

    def label(self, active_samples):
        """
        Dummy function: asks for user label.
        In an automated setting, you could simply set timeseries['label'][sample+n_steps]
        from timeseries['anomaly'][sample+n_steps].
        """
        for sample in active_samples:
            print('Please label sample with index =', sample)
            print('0 => normal, 1 => anomaly:')
            label = input()
            self.env.timeseries.loc[sample + n_steps - 1, 'anomaly'] = label
        return


class WarmUp(object):
    """
    For demonstration, a simple warm-up with e.g. OneClassSVM or IsolationForest.
    """
    def warm_up_isolation_forest(self, outliers_fraction, X_train):
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(contamination=outliers_fraction, random_state=42)
        clf.fit(X_train)
        return clf


# ------------------------------------------------------------------------------------
#                           Main Q-Learning Loop
# ------------------------------------------------------------------------------------
def q_learning(env,
               sess,
               qlearn_estimator,
               target_estimator,
               num_episodes,
               num_epoches,
               replay_memory_size=500000,
               replay_memory_init_size=50000,
               experiment_dir='./log/',
               update_target_estimator_every=10000,
               discount_factor=0.99,
               epsilon_start=1.0,
               epsilon_end=0.1,
               epsilon_decay_steps=500000,
               batch_size=512,
               num_LabelPropagation=20,
               num_active_learning=5,
               test=0,
               vae_model=None):
    """
    Off-policy Q-Learning with function approximation.
    """
    Transition = namedtuple("Transition", ["state", "reward", "next_state", "done"])
    replay_memory = []

    # checkpoint setup
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver = tf.compat.v1.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
        if test:
            return

    total_t = 0  # total steps

    # Epsilon schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    policy = make_epsilon_greedy_policy(qlearn_estimator, env.action_space_n)

    # 1) Warm-up / populate memory with a baseline approach (e.g., IsolationForest)
    # so we have some labeled data
    print("Populating replay memory with warm-up approach...")
    warmup_time = time.time()

    # We'll gather data from *all* timeseries up front
    data_train = []
    for ds_idx in range(env.datasetsize):
        env.reset_to(ds_idx)
        data_train.extend(env.states_list)

    # Convert to e.g. shape (N, n_steps*n_input_dim) if needed
    # For demonstration, let's just keep last dimension to do outlier detection
    # We only keep "value" from the last step:
    # BUT your code or approach may differ. Adjust as needed.
    numeric_data = []
    for st in data_train:
        # st is shape (n_steps, n_input_dim)
        # We'll flatten, or just keep st[-1, 0]
        numeric_data.append(st[-1, 0])
    numeric_data = np.array(numeric_data).reshape(-1, 1)

    # Fit IsolationForest
    outliers_fraction = 0.01
    model = WarmUp().warm_up_isolation_forest(outliers_fraction, numeric_data)

    # We pick some "most uncertain" or "most anomalous" points to label
    anomaly_score = model.decision_function(numeric_data)
    # Lower = more likely outlier => we pick extremes
    sorted_idx = np.argsort(anomaly_score)
    # pick e.g. 10 from low + 10 from high
    warm_samples = np.concatenate([sorted_idx[:5], sorted_idx[-5:]])

    # We'll label them (set env.timeseries['label']) and push to replay
    # Because we want the transitions in replay. We'll do this for each DS if we like,
    # but here's a simpler approach for demonstration.
    # We'll store them in memory with the environment set to the correct dataset.
    # This is just a skeleton approach. Feel free to adapt to your labeling logic.

    # We'll also demonstrate label propagation
    lp_model = LabelSpreading()

    # Actually warm up
    already_labeled_count = 0
    current_ds = env.datasetidx
    for idx in warm_samples:
        # figure out which dataset that index belongs to
        # if data_train was built by simply appending states from each dataset in a row,
        # you'd need to track the boundaries. For brevity, let's just assume everything
        # belongs to the current DS (or re-run for each DS).
        # This logic is heavily dependent on how you appended your data.
        # We'll do a single DS approach for demonstration.
        env.reset_to(current_ds)

        # idx points to state idx in env.states_list
        if idx >= len(env.states_list):
            continue
        state = env.states_list[idx]
        # set the env's cursor
        env.timeseries_curser = idx + n_steps

        # we "label" => timeseries['label'] = ground truth from 'anomaly'
        if env.timeseries_curser < len(env.timeseries):
            env.timeseries['label'][env.timeseries_curser] = env.timeseries['anomaly'][env.timeseries_curser]
            already_labeled_count += 1

            # choose action with the current policy
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
            action_probs = policy(state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)
            replay_memory.append(Transition(state, reward, next_state, done))

    warmup_time = time.time() - warmup_time
    print(f"Warm-up done, labeled ~{already_labeled_count} samples in {warmup_time:.2f} seconds.")
    print(f"Replay memory size after warm-up: {len(replay_memory)}")

    # 2) Main training episodes
    # We'll do AL each new dataset and store transitions.
    dict_labeled = {}  # track labeled indices per dataset

    for i_episode in range(num_episodes):
        if i_episode % 50 == 49:
            print("Saving checkpoint in episode {}/{}".format(i_episode + 1, num_episodes))
            saver.save(sess, checkpoint_path)

        per_loop_time1 = time.time()

        # Reset the environment
        state = env.reset()
        while env.datasetidx > env.datasetrng * validation_separate_ratio:
            # skip test portion if it tries
            state = env.reset()

        # Perform Active Learning
        ds_idx = env.datasetidx
        labeled_index = [k - n_steps for k in range(n_steps, len(env.timeseries))
                         if env.timeseries['label'][k] != -1]
        # Prepare AL:
        al = active_learning(env, num_active_learning, 'margin_sampling',
                             estimator=qlearn_estimator,
                             already_selected=labeled_index)
        al_samples = al.get_samples()
        # label them
        for smp in al_samples:
            env.timeseries_curser = smp + n_steps
            env.timeseries['label'][env.timeseries_curser] = env.timeseries['anomaly'][env.timeseries_curser]

            # store transitions
            state_al = env.states_list[smp]
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
            action_probs = policy(state_al, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)
            replay_memory.append(Transition(state_al, reward, next_state, done))

        # Label Propagation on unlabeled
        # Build input for label propagation
        unlabeled_indices = []
        states_mat = np.array(env.states_list)  # shape (#samples, n_steps, n_input_dim)
        # We'll compress to (N, 2) or (N, n_steps*2)? For example:
        compressed = []
        labels_lp = []
        for s_i, st in enumerate(states_mat):
            # compress to last step's values or flatten
            compressed.append(st[-1])
            cur_label = env.timeseries['label'][s_i + n_steps] if (s_i + n_steps < len(env.timeseries)) else -1
            labels_lp.append(cur_label)
        compressed = np.array(compressed)
        labels_lp = np.array(labels_lp)

        unlabeled_indices = np.where(labels_lp == -1)[0]
        if len(unlabeled_indices) > 0:
            # fit LabelSpreading
            lp_model.fit(compressed, labels_lp)
            pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)
            # pick the top certain
            certainty_index = np.argsort(pred_entropies)
            # only keep unlabeled
            certainty_index = [ci for ci in certainty_index if ci in unlabeled_indices]
            # up to num_LabelPropagation
            certainty_index = certainty_index[:num_LabelPropagation]
            # give them pseudo labels
            for ci in certainty_index:
                pseudo_label = lp_model.transduction_[ci]
                true_idx = ci + n_steps
                if true_idx < len(env.timeseries):
                    env.timeseries['label'][true_idx] = pseudo_label

        per_loop_time2 = time.time()

        # 3) Update model
        # We'll do 'num_epoches' gradient updates each episode
        for _ in range(num_epoches):
            if len(replay_memory) < batch_size:
                break

            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
            samples = random.sample(replay_memory, batch_size)
            states_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # next_states_batch shape: (batch_size, 2, n_steps, n_input_dim)
            # Because we store 2 possible next states from RNNBinaryStateFuc
            # We want Q-values from target
            next_states_batch0 = next_states_batch[:, 0]
            next_states_batch1 = next_states_batch[:, 1]

            q_values_next0 = target_estimator.predict(next_states_batch0, sess=sess)
            q_values_next1 = target_estimator.predict(next_states_batch1, sess=sess)
            # best next actions
            next_q0_max = np.amax(q_values_next0, axis=1)
            next_q1_max = np.amax(q_values_next1, axis=1)
            # form the target
            # reward_batch is shape (batch_size,). We want shape (batch_size, 2)
            # reward has the same 2-d structure as the RNNBinaryRewardFuc
            targets_batch = reward_batch + discount_factor * np.stack((next_q0_max, next_q1_max), axis=-1)

            qlearn_estimator.update(states_batch, targets_batch.astype(np.float32), sess=sess)
            total_t += 1

            # Periodically update target
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, qlearn_estimator, target_estimator)
                print("Copied model parameters to target network.")

        per_loop_time_popu = per_loop_time2 - per_loop_time1
        per_loop_time_updt = time.time() - per_loop_time2
        print("Global step {} @ Episode {}/{}, time: {} + {}"
              .format(total_t, i_episode + 1, num_episodes,
                      per_loop_time_popu, per_loop_time_updt))


def evaluate_model(y_true, y_pred):
    """
    Compute classification metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_acc = (recall + (tn / (tn + fp))) / 2 if (tn+fp)>0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

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
    """
    Validate using the Q-model in a purely greedy manner (epsilon=0).
    Returns averaged metrics across episodes.
    """
    y_true_all = []
    y_pred_all = []

    for i_episode in range(num_episodes):
        policy = make_epsilon_greedy_policy(estimator, env.action_space_n)
        state = env.reset()
        while env.datasetidx < env.datasetrng * validation_separate_ratio:
            # skip training portion; or if you want to specifically pick the test portion
            state = env.reset()

        done = False
        while not done:
            action_probs = policy(state, 0)  # greedy
            action = np.argmax(action_probs)
            # record
            if env.timeseries_curser < len(env.timeseries):
                y_true_all.append(env.timeseries['anomaly'][env.timeseries_curser])
                y_pred_all.append(action)

            next_state, reward, done, _ = env.step(action)
            if not done:
                state = next_state[action]

    results = evaluate_model(y_true_all, y_pred_all)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    return results


# ------------------------------------------------------------------------------------
#                              Training Function
# ------------------------------------------------------------------------------------
def train(num_LP, num_AL, discount_factor, learn_tau=True):
    """
    Train the RL agent using the environment, hooking in a VAE,
    with or without learned tau.
    """
    original_dim = 3
    latent_dim = 10
    intermediate_dim = 64

    x_train = load_normal_data(data_directory)
    vae, _ = build_vae(original_dim, latent_dim, intermediate_dim)
    vae.fit(x_train, epochs=50, batch_size=32)
    vae.save('vae_model.h5')

    # load the VAE (GPU-compatible if your TF is a GPU build)
    vae_loaded = load_model('vae_model.h5', custom_objects={'Sampling': Sampling}, compile=False)

    # Prepare environment
    # example path for the A1Benchmark folder:
    dataset_dir = r"C:\Users\Asus\Documents\PSU-Course\ydata-labeled-time-series-anomalies-v1_0\A1Benchmark"
    env = EnvTimeSeriesfromRepo(dataset_dir)
    env.statefnc = RNNBinaryStateFuc
    if learn_tau:
        env.rewardfnc = lambda ts, c, a: RNNBinaryRewardFuc(ts, c, a, vae=vae_loaded)
    else:
        # can provide a simpler function if not learning tau
        env.rewardfnc = lambda ts, c, a: RNNBinaryRewardFuc(ts, c, a, vae=vae_loaded, scale_factor=10)

    env.timeseries_curser_init = n_steps
    env.datasetfix = 0
    env.datasetidx = 0

    # for testing
    env_test = env
    env_test.rewardfnc = RNNBinaryRewardFucTest

    exp_dir = os.path.abspath("./exp/learn_tau_experiment/")
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    tf.compat.v1.reset_default_graph()
    global_step = tf.Variable(0, name="global_step", trainable=False)

    qlearn_estimator = Q_Estimator_Nonlinear(scope="qlearn", summaries_dir=exp_dir, learning_rate=0.0003)
    target_estimator = Q_Estimator_Nonlinear(scope="target")

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        q_learning(env,
                   sess=sess,
                   qlearn_estimator=qlearn_estimator,
                   target_estimator=target_estimator,
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

        # Evaluate
        dataset_for_testing = int(env.datasetsize * (1 - validation_separate_ratio))
        stats = q_learning_validator(env_test, qlearn_estimator, dataset_for_testing, exp_dir)
    return stats


def plot_tau_evolution():
    plt.figure(figsize=(10, 5))
    plt.plot(tau_values, label="Tau over time", alpha=0.8)
    plt.xlabel("Training Steps")
    plt.ylabel("Tau Value")
    plt.title("Evolution of Tau during Training")
    plt.legend()
    plt.show()


# ------------------------------------------------------------------------------------
#                                  Main Example
# ------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Example usage:
    # run a few trainings with different discount_factor or AL/LP combos
    # and possibly learning tau or not
    # Then plot tau if we used it
    print("Starting training with learned tau...")

    train(100, 30, 0.92, learn_tau=True)
    train(150, 50, 0.94, learn_tau=True)
    train(200, 100, 0.96, learn_tau=True)

    # If you want to see how tau changes
    plot_tau_evolution()
    print("Done.")
