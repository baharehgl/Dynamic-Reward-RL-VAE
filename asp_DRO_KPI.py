import os
import sys
import time
import zipfile
import itertools
import random
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             confusion_matrix, matthews_corrcoef)
from sklearn.ensemble import IsolationForest
from tensorflow.keras import layers, models, losses
from tensorflow.keras.models import load_model
from collections import deque, namedtuple
import glob

# --------------------------- Custom Environment Import ---------------------------
# Adjust path or comment this out if your environment is in a different place
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
try:
    from environment.time_series_repo_ext import EnvTimeSeriesfromRepo
except ImportError as e:
    print("Error importing EnvTimeSeriesfromRepo:", e)
    sys.exit(1)

# --------------------------- CUDA Configuration ---------------------------
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

# --------------------------- Hyperparameters ---------------------------
DATAFIXED = 0  # 0 means iterate among multiple time series if environment supports it
EPISODES = 500  # Number of episodes for training (not always used directly)
DISCOUNT_FACTOR = 0.5  # Reward discount factor
EPSILON = 0.5  # Epsilon for epsilon-greedy
EPSILON_DECAY = 1.00

NOT_ANOMALY = 0
ANOMALY = 1
action_space = [NOT_ANOMALY, ANOMALY]
action_space_n = len(action_space)

n_steps = 25  # Sliding window length
n_input_dim = 1  # LSTM input dimension per time step (assuming 1 feature: 'value')
n_hidden_dim = 128  # Hidden dimension of LSTM

# Reward numeric values
TP_Value = 5
TN_Value = 1
FP_Value = -1
FN_Value = -5

validation_separate_ratio = 0.9

# --------------------------- For test usage or logging ---------------------------
tau_values = []  # Keep track of how tau evolves


# --------------------------- Data Extraction ---------------------------
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")


# Paths to the zip files
kpi_train_zip = os.path.join(current_dir, "KPI_train.csv.zip")
kpi_test_zip = os.path.join(current_dir, "KPI_ground_truth.hdf.zip")

train_extract_dir = os.path.join(current_dir, 'KPI_data', 'train')
test_extract_dir = os.path.join(current_dir, 'KPI_data', 'test')
os.makedirs(train_extract_dir, exist_ok=True)
os.makedirs(test_extract_dir, exist_ok=True)

# Unzip
if os.path.exists(kpi_train_zip):
    unzip_file(kpi_train_zip, train_extract_dir)
if os.path.exists(kpi_test_zip):
    unzip_file(kpi_test_zip, test_extract_dir)


def find_file(directory, pattern):
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)
    if not files:
        raise FileNotFoundError(f"No file matching pattern '{pattern}' found in directory '{directory}'")
    return files[0]


def list_hdf_keys(hdf_path):
    with pd.HDFStore(hdf_path, 'r') as store:
        print("\nAvailable keys in the HDF5 file:")
        print(store.keys())


def load_normal_data_kpi(data_path, exclude_columns=None):
    """
    Load numeric columns from the CSV, optionally excluding columns.
    Scale them with StandardScaler. Return numpy array.
    """
    data = pd.read_csv(data_path)
    print("\nColumns in the CSV and their data types:")
    print(data.dtypes)

    if exclude_columns:
        metric_columns = [col for col in data.columns if col not in exclude_columns]
    else:
        metric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    print(f"\nSelected metric columns for scaling: {metric_columns}")

    # Convert to numeric, forward/back fill if needed
    data = data[metric_columns].apply(pd.to_numeric, errors='coerce')
    data = data.fillna(method='ffill').fillna(method='bfill')

    if data.isnull().values.any():
        missing_cols = data.columns[data.isnull().any()].tolist()
        raise ValueError(f"Data contains non-numeric values in columns: {missing_cols}.")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values)
    return scaled_data


def load_test_data_kpi_pandas(data_path, key='data'):
    try:
        df = pd.read_hdf(data_path, key=key)
        print(f"\nSuccessfully loaded data from key '{key}'.")
    except Exception as e:
        print(f"\nError reading HDF5 file with Pandas: {e}")
        sys.exit(1)

    print("\nLoaded DataFrame head:")
    print(df.head())

    if 'anomaly' not in df.columns:
        print("Error: 'anomaly' column not found in the test data.")
        sys.exit(1)

    # Create a 'value' column as the mean of all numeric metrics except these columns:
    metric_columns = [col for col in df.columns if col not in ['timestamp', 'label', 'KPI ID', 'anomaly']]
    if not metric_columns:
        print("Error: No metric columns found for creating the 'value' column.")
        sys.exit(1)

    df['value'] = df[metric_columns].mean(axis=1)
    print("Created 'value' column as the mean of metric columns.")

    return df


def load_test_data_kpi(data_path):
    return load_test_data_kpi_pandas(data_path, key='data')


# Dynamically locate files after extraction
try:
    kpi_train_csv = find_file(train_extract_dir, '*.csv')
    kpi_test_hdf = find_file(test_extract_dir, '*.hdf')
except FileNotFoundError as e:
    print(e)
    sys.exit(1)

# Optional: see the available HDF keys
list_hdf_keys(kpi_test_hdf)


# --------------------------- VAE Components ---------------------------
class Sampling(layers.Layer):
    """Reparameterization trick layer."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae(original_dim, latent_dim=2, intermediate_dim=64):
    """
    Build a simple fully-connected VAE for 1D data.
    If you have time-series windows of length n_steps,
    set original_dim = n_steps.
    """
    inputs = layers.Input(shape=(original_dim,))
    h = layers.Dense(intermediate_dim, activation='relu')(inputs)
    h = layers.Dense(intermediate_dim, activation='relu')(h)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    z = Sampling()([z_mean, z_log_var])

    # Decoder
    decoder_h = layers.Dense(intermediate_dim, activation='relu')(z)
    decoder_h = layers.Dense(intermediate_dim, activation='relu')(decoder_h)
    decoder_mean = layers.Dense(original_dim, activation='sigmoid')(decoder_h)

    vae = models.Model(inputs, decoder_mean, name="vae")
    encoder = models.Model(inputs, [z_mean, z_log_var, z], name="encoder")

    reconstruction_loss = losses.binary_crossentropy(inputs, decoder_mean)
    reconstruction_loss = tf.reduce_sum(reconstruction_loss, axis=1)  # sum over features
    kl_loss = -0.5 * tf.reduce_sum(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
        axis=1
    )
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return vae, encoder


# --------------------------- DRO and Tau Scaling ---------------------------
def kl_divergence(p, q):
    """Compute KL divergence KL(p || q). p and q are distributions that sum to 1."""
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))


def adaptive_scaling_factor_dro(preference_strength, tau_min=0.1, tau_max=5.0, rho=0.5, max_iter=10):
    """
    Simple iterative approach to adapt tau.
    `preference_strength` = sigmoid(reconstruction_error).
    This is purely illustrative; you may adapt the logic.
    """
    tau = 1.0
    for _ in range(max_iter):
        # We treat [p, 1-p] as distribution, compare to [0.5, 0.5]
        p = np.array([preference_strength, 1 - preference_strength])
        kl_term = kl_divergence(p, np.array([0.5, 0.5]))

        # Pseudo gradient
        grad = -np.log(1 + np.exp(-preference_strength / tau)) + rho - kl_term
        # Approx. second derivative
        # Because we do not do real gradient of the KL wrt tau, this is simplistic
        hess = 0.1  # a small constant to avoid 0

        tau = tau - grad / (hess + 1e-8)
        tau = np.clip(tau, tau_min, tau_max)

    return tau


# --------------------------- Reward Functions ---------------------------
def RNNBinaryRewardFuc(timeseries, timeseries_curser, action=0, vae=None, scale_factor=10):
    """
    If timeseries_curser >= n_steps, we take the last n_steps points from 'value',
    pass to VAE as shape (1, n_steps) if our VAE is built that way.

    For simplicity, assume VAE has original_dim = n_steps.
    If your VAE has original_dim=1, adapt accordingly (e.g., single point).
    """
    if timeseries_curser >= n_steps:
        # shape = (1, n_steps)
        window = timeseries['value'][timeseries_curser - n_steps: timeseries_curser].values
        window = np.reshape(window, (1, n_steps))  # feed shape (batch, n_steps) to VAE

        vae_reconstruction = vae.predict(window)
        reconstruction_error = np.mean(np.square(vae_reconstruction - window))

        vae_penalty = -scale_factor * reconstruction_error

        # Sigmoid of reconstruction_error
        preference_strength = 1 / (1 + np.exp(-reconstruction_error))
        preference_strength = np.clip(preference_strength, 0.05, 0.95)

        tau = adaptive_scaling_factor_dro(preference_strength)
        tau_values.append(tau)

        true_label = timeseries['anomaly'][timeseries_curser]
        if true_label == 0:
            # (r_not_anom, r_anom)
            return [tau * (TN_Value + vae_penalty), tau * (FP_Value + vae_penalty)]
        else:
            return [tau * (FN_Value + vae_penalty), tau * (TP_Value + vae_penalty)]
    else:
        return [0, 0]


def RNNBinaryRewardFucTest(timeseries, timeseries_curser, action=0):
    """
    Simpler reward for testing (no VAE overhead).
    """
    if timeseries_curser >= n_steps:
        true_label = timeseries['anomaly'][timeseries_curser]
        if true_label == 0:
            return [TN_Value, FP_Value]
        else:
            return [FN_Value, TP_Value]
    else:
        return [0, 0]


# --------------------------- Q-Learning Network ---------------------------
class Q_Estimator_Nonlinear():
    """
    Q(s,a) approximator with an LSTM on top of n_steps of 1D input
    (i.e., each state is shape (n_steps, n_input_dim)).
    """

    def __init__(self,
                 learning_rate=np.float32(0.001),
                 scope="Q_Estimator_Nonlinear",
                 summaries_dir=None):
        self.scope = scope
        self.summary_writer = None

        with tf.compat.v1.variable_scope(scope):
            # Placeholders
            self.state = tf.compat.v1.placeholder(shape=[None, n_steps, n_input_dim],
                                                  dtype=tf.float32, name="state")
            self.target = tf.compat.v1.placeholder(shape=[None, action_space_n],
                                                   dtype=tf.float32, name="target")

            # LSTM definition
            lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(n_hidden_dim, forget_bias=1.0)
            # Unstack the input to a list of [batch_size, n_input_dim] of length n_steps
            # shape self.state: (batch, n_steps, n_input_dim)
            state_unstack = tf.unstack(self.state, n_steps, axis=1)
            outputs, _ = tf.compat.v1.nn.static_rnn(lstm_cell, state_unstack, dtype=tf.float32)
            # outputs[-1] has shape (batch_size, n_hidden_dim)

            # Final dense layer to produce action values
            self.W_out = tf.compat.v1.get_variable("W_out", shape=[n_hidden_dim, action_space_n],
                                                   initializer=tf.compat.v1.random_normal_initializer())
            self.b_out = tf.compat.v1.get_variable("b_out", shape=[action_space_n],
                                                   initializer=tf.compat.v1.constant_initializer(0.0))
            self.action_values = tf.matmul(outputs[-1], self.W_out) + self.b_out

            # Loss and training
            self.losses = tf.compat.v1.squared_difference(self.action_values, self.target)
            self.loss = tf.reduce_mean(self.losses)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

            # Global step
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

            # Summaries
            self.summaries = tf.compat.v1.summary.merge([
                tf.compat.v1.summary.histogram("loss_hist", self.losses),
                tf.compat.v1.summary.scalar("loss", self.loss),
                tf.compat.v1.summary.histogram("q_values_hist", self.action_values),
                tf.compat.v1.summary.scalar("q_value_max", tf.reduce_max(self.action_values))
            ])

            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                os.makedirs(summary_dir, exist_ok=True)
                self.summary_writer = tf.compat.v1.summary.FileWriter(summary_dir)

    def predict(self, state, sess=None):
        """
        state shape: (batch_size, n_steps, n_input_dim)
        returns shape: (batch_size, action_space_n)
        """
        sess = sess or tf.compat.v1.get_default_session()
        return sess.run(self.action_values, {self.state: state})

    def update(self, state, target, sess=None):
        """
        state shape: (batch_size, n_steps, n_input_dim)
        target shape: (batch_size, action_space_n)
        """
        sess = sess or tf.compat.v1.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        summaries, global_step, _, loss_val = sess.run(
            [self.summaries, self.global_step, self.train_op, self.loss],
            feed_dict
        )
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss_val


def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.
    """
    e1_params = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith(estimator1.scope)]
    e1_params = sorted(e1_params, key=lambda v: v.name)
    e2_params = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith(estimator2.scope)]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        update_ops.append(e2_v.assign(e1_v))
    sess.run(update_ops)


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy for the given Q approximator.
    """

    def policy_fn(observation, epsilon, sess=None):
        # observation shape = (n_steps, n_input_dim), we pass as (1, n_steps, n_input_dim)
        q_values = estimator.predict(np.expand_dims(observation, axis=0), sess=sess)[0]
        A = np.ones(nA, dtype='float32') * epsilon / nA
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


# --------------------------- Replay Memory Transition ---------------------------
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


# --------------------------- Active Learning & Warm-Up ---------------------------
class active_learning(object):
    """
    Example of an active-learning sampler.
    We define a 'margin_sampling' strategy on the Q-values themselves.
    """

    def __init__(self, env, N, strategy, estimator, already_selected):
        self.env = env
        self.N = N
        self.strategy = strategy
        self.estimator = estimator
        self.already_selected = set(already_selected)

    def get_samples(self, sess=None):
        states_list = self.env.states_list  # list of shape (n_samples, n_steps, n_input_dim)
        distances = []
        for state in states_list:
            # state shape is (n_steps,) or (n_steps,1) depending on your environment
            # ensure shape is (n_steps, n_input_dim)
            s_reshaped = np.reshape(state, (n_steps, n_input_dim))
            q_vals = self.estimator.predict(np.expand_dims(s_reshaped, axis=0), sess=sess)[0]
            # margin = difference between top 2 Q-values
            sorted_q = np.sort(q_vals)
            margin = sorted_q[-1] - sorted_q[-2]
            distances.append(abs(margin))

        distances = np.array(distances)
        # We pick smallest margins => highest confusion
        rank_ind = np.argsort(distances)  # ascending
        # Exclude already selected
        rank_ind = [i for i in rank_ind if i not in self.already_selected]
        active_samples = rank_ind[:self.N]
        return active_samples


class WarmUp(object):
    """
    Simple warm-up heuristics to label potential anomalies using unsupervised methods
    (One-Class SVM, IsolationForest, etc.).
    """

    def warm_up_SVM(self, outliers_fraction, X_train):
        model = OneClassSVM(gamma='auto', nu=0.95 * outliers_fraction)
        model.fit(X_train)
        return model

    def warm_up_isolation_forest(self, outliers_fraction, X_train):
        clf = IsolationForest(contamination=outliers_fraction, random_state=42)
        clf.fit(X_train)
        return clf


# --------------------------- State Function ---------------------------
def RNNBinaryStateFuc(timeseries, timeseries_curser):
    """
    Return the last n_steps of 'value' as a 1D or 2D array for the agent.
    """
    if timeseries_curser < n_steps:
        # If there's not enough history, pad or return zeros
        pad_length = n_steps - timeseries_curser
        front_pad = np.zeros(pad_length)
        val_part = timeseries['value'][:timeseries_curser].values
        state_1d = np.concatenate([front_pad, val_part], axis=0)
    else:
        state_1d = timeseries['value'][timeseries_curser - n_steps: timeseries_curser].values

    # Convert to (n_steps, n_input_dim)
    return np.reshape(state_1d, (n_steps, n_input_dim))


# --------------------------- Q-Learning Main ---------------------------
def q_learning(env,
               sess,
               q_estimator,
               target_estimator,
               num_episodes,
               num_epoches,
               replay_memory_size=50000,
               replay_memory_init_size=1000,
               experiment_dir='./log/',
               update_target_estimator_every=1000,
               discount_factor=0.99,
               epsilon_start=1.0,
               epsilon_end=0.1,
               epsilon_decay_steps=5000,
               batch_size=32,
               num_LabelPropagation=20,
               num_active_learning=5,
               test=False):
    """
    Main training loop for Q-learning with experience replay, label propagation,
    and active learning. This is an illustrative skeleton; adapt as needed.
    """
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    os.makedirs(checkpoint_dir, exist_ok=True)
    saver = tf.compat.v1.train.Saver()

    # Possibly load checkpoint
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print(f"\nLoading model checkpoint {latest_checkpoint}...")
        saver.restore(sess, latest_checkpoint)
        if test:
            return

    # Epsilon schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    replay_memory = deque(maxlen=replay_memory_size)

    # Policy
    policy = make_epsilon_greedy_policy(q_estimator, env.action_space_n)
    total_t = sess.run(q_estimator.global_step)

    # Warm-up stage using isolation forest or SVM
    print("Populating replay memory with a warm-up approach...\n")
    outliers_fraction = 0.01
    # Example: gather some states from environment
    data_train = []
    # Here, for each dataset in environment, or for multiple resets:
    for num in range(env.datasetsize):
        env.reset()
        data_train.extend(env.states_list)  # env.states_list are states already built
    # data_train is list of states (each shape (n_steps, n_input_dim)?).
    # For simple isolation forest, flatten them to 1D:
    data_train_flat = [np.ravel(s) for s in data_train]
    data_train_flat = np.array(data_train_flat)

    # Fit an unsupervised model
    warm_model = WarmUp().warm_up_isolation_forest(outliers_fraction, data_train_flat)
    # Use it to populate replay memory with (state, action, reward, next_state, done)
    # We'll do a random pass over states, guess anomaly if model says so
    # Then environment will give a reward.
    # This is purely to fill replay memory with some transitions.

    # Make sure timeseries have 'label' column or at least 'anomaly' to compute reward
    steps_populated = 0
    while steps_populated < replay_memory_init_size:
        env.reset()
        for i in range(len(env.states_list)):
            if env.timeseries_curser < n_steps:
                # Not enough data to step
                continue

            state = env.states_list[i]  # shape (n_steps, n_input_dim)
            # Flatten for isolation_forest
            state_flat = np.ravel(state).reshape(1, -1)
            anomaly_score = warm_model.decision_function(state_flat)[0]  # smaller => more anomalous
            # pick action = 1 if anomaly_score < threshold, else 0
            # or do random, etc.
            action = int(anomaly_score < 0.0)
            next_state, reward, done, _ = env.step(action)

            # Store in replay
            transition = Transition(state=state,
                                    action=action,
                                    reward=reward[action],  # reward is a list [reward_if_action=0, reward_if_action=1]
                                    next_state=next_state[action],
                                    done=done)
            replay_memory.append(transition)
            steps_populated += 1
            if done or steps_populated >= replay_memory_init_size:
                break

    print(f"Replay memory populated with {len(replay_memory)} transitions.\n")

    # ---------------------------------
    # Start the actual training episodes
    for i_episode in range(num_episodes):
        # Save checkpoint periodically
        if i_episode % 50 == 0:
            saver.save(sess, checkpoint_path)
            print(f"Checkpoint saved at episode {i_episode}.")

        # We do a fresh reset
        state = env.reset()
        done = False

        # Active Learning (label some unlabeled samples)
        labeled_index = np.where(env.timeseries['label'] != -1)[0]
        # or handle it differently.
        al = active_learning(env, N=num_active_learning, strategy='margin_sampling',
                             estimator=q_estimator, already_selected=labeled_index)

        al_samples = al.get_samples(sess=sess)
        print(f"Active learning picked samples: {al_samples}")
        # Suppose we assign ground-truth to them
        for idx in al_samples:
            # The real label = env.timeseries.at[idx, 'anomaly']
            env.timeseries.at[idx, 'label'] = env.timeseries.at[idx, 'anomaly']

        # Label Propagation
        lp_model = LabelSpreading()
        # We can do: X = flatten states, y = known labels or -1
        X = []
        y = []
        for i, st in enumerate(env.states_list):
            X.append(np.ravel(st))
            if env.timeseries.at[i, 'label'] != -1:
                y.append(env.timeseries.at[i, 'label'])
            else:
                y.append(-1)
        X = np.array(X)
        y = np.array(y, dtype=int)
        lp_model.fit(X, y)

        # Now pseudo-label uncertain ones
        pred_entropies = stats.entropy(lp_model.label_distributions_.T)
        unlabeled_indices = np.where(y == -1)[0]
        # pick up to num_LabelPropagation most certain
        certainty_index = np.argsort(pred_entropies)[:num_LabelPropagation]
        for cid in certainty_index:
            if cid in unlabeled_indices:
                pseudo_label = lp_model.transduction_[cid]
                env.timeseries.at[cid, 'label'] = pseudo_label

        # Next, we do a "mini-epoch" of environment steps
        for epoch_step in range(num_epoches):
            # Possibly step through the entire time series, or random sample
            idx_list = list(range(len(env.states_list)))
            random.shuffle(idx_list)
            for i_idx in idx_list:
                if i_idx < n_steps:
                    continue
                env.timeseries_curser = i_idx
                # Epsilon
                epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
                st = env.states_list[i_idx]  # shape (n_steps, n_input_dim)
                act_probs = policy(st, epsilon, sess=sess)
                action = np.random.choice(np.arange(action_space_n), p=act_probs)

                next_state, reward, done, _ = env.step(action)
                # Store in replay
                trans = Transition(st, action, reward[action], next_state[action], done)
                replay_memory.append(trans)

                # Update from replay
                if len(replay_memory) < batch_size:
                    continue
                samples = random.sample(replay_memory, batch_size)
                states_batch = []
                target_batch = []

                next_states_np = []
                for t_ in samples:
                    states_batch.append(t_.state)
                    next_states_np.append(t_.next_state)

                # shape => (batch_size, n_steps, n_input_dim)
                states_batch = np.array(states_batch)
                next_states_batch = np.array(next_states_np)
                # Predict next Q using target
                q_next = target_estimator.predict(next_states_batch, sess=sess)
                # Build target
                # We need the current Q for each transition to get target
                q_current = q_estimator.predict(states_batch, sess=sess)
                for i_b, t_ in enumerate(samples):
                    # Q-learning target:
                    if t_.done:
                        td_target = t_.reward
                    else:
                        td_target = t_.reward + discount_factor * np.max(q_next[i_b])
                    # copy from q_current
                    updated = np.array(q_current[i_b])
                    updated[t_.action] = td_target
                    target_batch.append(updated)

                target_batch = np.array(target_batch)
                # update
                loss_val = q_estimator.update(states_batch, target_batch, sess=sess)

                total_t += 1

                # periodically update target net
                if total_t % update_target_estimator_every == 0:
                    copy_model_parameters(sess, q_estimator, target_estimator)
                    print("Copied model parameters to target network.")

        print(f"Episode {i_episode} ended. Global step = {total_t}")

    # Finally, save at the end
    saver.save(sess, checkpoint_path)
    print("Final model saved.")


# --------------------------- Evaluation ---------------------------
def evaluate_model(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    mcc = matthews_corrcoef(y_true, y_pred)
    balanced_acc = (recall + (tn / (tn + fp))) / 2 if (tn + fp) > 0 else recall
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


def q_learning_validator(env, estimator, num_episodes=1, record_dir=None, plot=True):
    """
    Validate the model by running a (greedy) policy on test portion of the environment.
    """
    y_true_all = []
    y_pred_all = []

    policy = make_epsilon_greedy_policy(estimator, env.action_space_n)

    # For each of num_episodes, we go into test portion
    for i_episode in range(num_episodes):
        print(f"\nValidation Episode {i_episode + 1}/{num_episodes}")
        env.reset()

        # skip until test range in environment
        while env.datasetidx < env.datasetrng * validation_separate_ratio:
            env.reset()

        done = False
        # step through entire time series
        for t in itertools.count():
            st = env.statefnc(env.timeseries, env.timeseries_curser)
            action_probs = policy(st, 0.0)  # greedy
            action = np.argmax(action_probs)

            # record
            y_true_all.append(env.timeseries.at[env.timeseries_curser, 'anomaly'])
            y_pred_all.append(action)

            next_state, reward, done, _ = env.step(action)
            if done:
                break

    results = evaluate_model(y_true_all, y_pred_all)
    print("\nValidation Metrics:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    if record_dir:
        with open(os.path.join(record_dir, "validation_metrics.txt"), 'w') as f:
            for k, v in results.items():
                f.write(f"{k}: {v:.4f}\n")

    if plot:
        # Plot tau evolution
        plt.figure(figsize=(8, 4))
        plt.plot(tau_values, label="Tau")
        plt.xlabel("Steps (approx)")
        plt.ylabel("Tau")
        plt.title("Adaptive Tau Evolution")
        plt.legend()
        if record_dir:
            plt.savefig(os.path.join(record_dir, "tau_evolution.png"))
        else:
            plt.savefig("tau_evolution.png")
        plt.close()

    return results


# --------------------------- Training Wrapper ---------------------------
def train(num_LP=20, num_AL=5, discount_factor=0.92, learn_tau=True):
    """
    Full pipeline:
      1) Load & scale train data
      2) Train or load VAE
      3) Create environment
      4) Q-learning with label propagation & active learning
      5) Validate
    """
    # 1) Load data
    x_train = load_normal_data_kpi(kpi_train_csv, exclude_columns=['timestamp', 'label', 'KPI ID'])
    df_test = load_test_data_kpi(kpi_test_hdf)

    # 2) Build and train a VAE
    #   If you want the VAE to handle windows of length n_steps, set original_dim=n_steps,
    #   but then you have to also reshape x_train accordingly (sliding windows).
    #   Here we assume original_dim=1 (each time point is a single value).
    original_dim = 1
    latent_dim = 2
    intermediate_dim = 16

    vae, encoder = build_vae(original_dim, latent_dim, intermediate_dim)
    print("\nTraining VAE on single points (original_dim=1)...")
    vae.fit(x_train, x_train, epochs=10, batch_size=32, validation_split=0.1)

    vae_save_path = os.path.join(current_dir, 'vae_model_kpi.h5')
    vae.save(vae_save_path)
    print(f"VAE saved to {vae_save_path}")

    # 3) Create environment
    env = EnvTimeSeriesfromRepo()
    env.set_train_data(x_train)  # Implementation depends on your environment
    env.set_test_data(df_test)

    # Use the dro-based reward if learn_tau = True, otherwise some simpler reward
    if learn_tau:
        env.rewardfnc = lambda ts, cur, act: RNNBinaryRewardFuc(ts, cur, act, vae=vae)
    else:
        # Example no VAE, or simpler approach
        env.rewardfnc = RNNBinaryRewardFucTest

    # The environment must rely on RNNBinaryStateFuc for generating states
    env.statefnc = RNNBinaryStateFuc
    env.timeseries_curser_init = n_steps
    env.datasetfix = DATAFIXED
    env.datasetidx = 0

    # Test environment
    env_test = EnvTimeSeriesfromRepo()
    env_test.set_train_data(x_train)
    env_test.set_test_data(df_test)
    env_test.rewardfnc = RNNBinaryRewardFucTest
    env_test.statefnc = RNNBinaryStateFuc

    # 4) Setup Q Estimators
    exp_dir = os.path.abspath("./exp/AdaptiveTauRL/")
    os.makedirs(exp_dir, exist_ok=True)
    tf.compat.v1.reset_default_graph()
    q_estimator = Q_Estimator_Nonlinear(scope="qlearn", summaries_dir=exp_dir, learning_rate=3e-4)
    target_estimator = Q_Estimator_Nonlinear(scope="target", learning_rate=3e-4)

    sess = tf.compat.v1.Session()
    with sess.as_default():
        sess.run(tf.compat.v1.global_variables_initializer())

        # Train
        q_learning(env=env,
                   sess=sess,
                   q_estimator=q_estimator,
                   target_estimator=target_estimator,
                   num_episodes=20,  # Adjust as needed
                   num_epoches=5,  # inner epoch steps each episode
                   replay_memory_size=5000,
                   replay_memory_init_size=500,
                   experiment_dir=exp_dir,
                   update_target_estimator_every=200,
                   discount_factor=discount_factor,
                   epsilon_start=1.0,
                   epsilon_end=0.1,
                   epsilon_decay_steps=1000,
                   batch_size=32,
                   num_LabelPropagation=num_LP,
                   num_active_learning=num_AL,
                   test=False)

        # 5) Validate
        metrics = q_learning_validator(env_test, q_estimator,
                                       num_episodes=1,
                                       record_dir=exp_dir,
                                       plot=True)
        print("Validation metrics:", metrics)
        return metrics


# --------------------------- Main Execution ---------------------------
if __name__ == "__main__":
    try:
        print("\n--- Starting Training Run ---")
        results = train(num_LP=100, num_AL=30, discount_factor=0.92, learn_tau=True)
        print("Final Results:", results)
    except Exception as e:
        print("Error during training:", e)
