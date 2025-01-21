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

# ------------------------------------------------------------------------
# Custom environment import
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
try:
    from environment.time_series_repo_ext import EnvTimeSeriesfromRepo
except ImportError as e:
    print("Error importing EnvTimeSeriesfromRepo:", e)
    sys.exit(1)

# ------------------------------------------------------------------------
# GPU Configuration (optional)
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

# ------------------------------------------------------------------------
# Global Hyperparams
DATAFIXED = 0         # 0 means iterate among multiple time series if environment supports it
EPISODES = 500        # Not always used directly; can override in train calls
DISCOUNT_FACTOR = 0.5
EPSILON = 0.5
EPSILON_DECAY = 1.00

NOT_ANOMALY = 0
ANOMALY = 1
action_space = [NOT_ANOMALY, ANOMALY]
action_space_n = len(action_space)

n_steps = 25        # sliding window length
n_input_dim = 1     # dimension of each time step if 'value' is single-col
n_hidden_dim = 128  # LSTM hidden dimension

# Reward numeric values
TP_Value = 5
TN_Value = 1
FP_Value = -1
FN_Value = -5

validation_separate_ratio = 0.9

# We'll keep track of tau evolution in a global variable for plotting
tau_values = []

# ------------------------------------------------------------------------
# Data Extraction
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

kpi_train_zip = os.path.join(current_dir, "KPI_train.csv.zip")
kpi_test_zip  = os.path.join(current_dir, "KPI_ground_truth.hdf.zip")

train_extract_dir = os.path.join(current_dir, 'KPI_data', 'train')
test_extract_dir  = os.path.join(current_dir, 'KPI_data', 'test')
os.makedirs(train_extract_dir, exist_ok=True)
os.makedirs(test_extract_dir, exist_ok=True)

# Unzip if present
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
    Load numeric columns from CSV, optionally excluding certain columns.
    Scale with StandardScaler. Return a numpy array.
    """
    data = pd.read_csv(data_path)
    print("\nColumns in the CSV and their data types:")
    print(data.dtypes)

    if exclude_columns:
        metric_columns = [col for col in data.columns if col not in exclude_columns]
    else:
        metric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    print(f"\nSelected metric columns for scaling: {metric_columns}")

    data = data[metric_columns].apply(pd.to_numeric, errors='coerce')
    data = data.fillna(method='ffill').fillna(method='bfill')
    if data.isnull().values.any():
        missing_cols = data.columns[data.isnull().any()].tolist()
        raise ValueError(f"Data contains non-numeric values in columns: {missing_cols}.")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values)
    return scaled_data

def load_test_data_kpi_pandas(data_path, key='data'):
    """
    Load the HDF data. Rename 'label' -> 'anomaly' if necessary.
    Then create a 'value' column as the mean of all numeric metrics
    except 'timestamp','anomaly','KPI ID'.
    """
    try:
        df = pd.read_hdf(data_path, key=key)
        print(f"\nSuccessfully loaded data from key '{key}'.")
    except Exception as e:
        print(f"\nError reading HDF5 file with Pandas: {e}")
        sys.exit(1)

    print("\nLoaded DataFrame head:")
    print(df.head())

    # If there's no 'anomaly' column but there is a 'label' column, rename it
    if 'anomaly' not in df.columns and 'label' in df.columns:
        df.rename(columns={'label': 'anomaly'}, inplace=True)

    if 'anomaly' not in df.columns:
        print("Error: 'anomaly' column not found in the test data.")
        sys.exit(1)

    # Create 'value' column as mean of all numeric columns except these
    exclude_cols = ['timestamp','anomaly','KPI ID']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    metric_columns = [c for c in numeric_cols if c not in exclude_cols]

    if not metric_columns:
        print("Error: No metric columns found for creating the 'value' column.")
        sys.exit(1)

    df['value'] = df[metric_columns].mean(axis=1)
    print("Created 'value' column as the mean of metric columns.")

    return df

def load_test_data_kpi(data_path):
    return load_test_data_kpi_pandas(data_path, key='data')

# Find the extracted CSV/HDF
try:
    kpi_train_csv = find_file(train_extract_dir, '*.csv')
    kpi_test_hdf  = find_file(test_extract_dir,  '*.hdf')
except FileNotFoundError as e:
    print(e)
    sys.exit(1)

list_hdf_keys(kpi_test_hdf)

# ------------------------------------------------------------------------
# VAE Components
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
    Build a VAE that handles single-dimensional inputs (original_dim=1)
    without raising the 'Invalid reduction dimension' error.
    """
    from tensorflow.keras import layers, models, losses
    import tensorflow as tf

    # Inputs
    inputs = layers.Input(shape=(original_dim,))
    x = layers.Dense(intermediate_dim, activation='relu')(inputs)
    x = layers.Dense(intermediate_dim, activation='relu')(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)

    # Reparameterization trick
    class Sampling(layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = Sampling()([z_mean, z_log_var])

    # Decoder
    decoder_h = layers.Dense(intermediate_dim, activation='relu')(z)
    decoder_h = layers.Dense(intermediate_dim, activation='relu')(decoder_h)
    decoder_mean = layers.Dense(original_dim, activation='sigmoid')(decoder_h)

    # Full VAE model & separate encoder
    vae = models.Model(inputs, decoder_mean, name="vae")
    encoder = models.Model(inputs, [z_mean, z_log_var, z], name="encoder")

    # Instead of calling losses.binary_crossentropy(...) directly,
    # we define a BinaryCrossentropy with `reduction='none'`,
    # which yields a per-sample loss of shape (batch,).
    bce = losses.BinaryCrossentropy(from_logits=False, reduction=losses.Reduction.NONE)

    # reconstruction_loss => shape (batch,)
    reconstruction_loss = bce(inputs, decoder_mean)

    # KL loss => shape (batch,)
    kl_loss = -0.5 * tf.reduce_sum(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
        axis=1
    )

    # total loss => shape (batch,); we then do mean across the batch
    total_loss = reconstruction_loss + kl_loss
    vae_loss = tf.reduce_mean(total_loss)

    # Add to model and compile
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    return vae, encoder


# ------------------------------------------------------------------------
# DRO and Tau Scaling
def kl_divergence(p, q):
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))

def adaptive_scaling_factor_dro(preference_strength, tau_min=0.1, tau_max=5.0, rho=0.5, max_iter=10):
    """
    Simple iterative approach to adapt tau.
    """
    tau = 1.0
    for _ in range(max_iter):
        # We treat [p, 1-p] as distribution, compare to [0.5,0.5]
        p = np.array([preference_strength, 1 - preference_strength])
        kl_term = kl_divergence(p, np.array([0.5, 0.5]))

        # Pseudo gradient (illustrative)
        grad = -np.log(1 + np.exp(-preference_strength / tau)) + rho - kl_term
        hess = 0.1  # a small constant
        tau = tau - grad / (hess + 1e-8)
        tau = np.clip(tau, tau_min, tau_max)
    return tau

# ------------------------------------------------------------------------
# Reward Functions
def RNNBinaryRewardFuc(timeseries, timeseries_curser, action=0, vae=None, scale_factor=10):
    """
    Reward with VAE reconstruction error -> penalty,
    plus adaptive tau scaling by DRO.
    """
    if timeseries_curser >= n_steps:
        # Gather the last n_steps from 'value'
        window = timeseries['value'][timeseries_curser - n_steps : timeseries_curser].values
        window = np.reshape(window, (1, n_steps))

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
            # normal
            return [tau * (TN_Value + vae_penalty), tau * (FP_Value + vae_penalty)]
        else:
            # anomaly
            return [tau * (FN_Value + vae_penalty), tau * (TP_Value + vae_penalty)]
    else:
        return [0, 0]

def RNNBinaryRewardFucTest(timeseries, timeseries_curser, action=0):
    """
    For testing, just standard confusion-based reward without the VAE overhead.
    """
    if timeseries_curser >= n_steps:
        true_label = timeseries['anomaly'][timeseries_curser]
        if true_label == 0:
            return [TN_Value, FP_Value]
        else:
            return [FN_Value, TP_Value]
    else:
        return [0, 0]

# ------------------------------------------------------------------------
# Q-Learning Network
class Q_Estimator_Nonlinear():
    """
    Q(s,a) approximator with an LSTM on top of n_steps of 1D input.
    """
    def __init__(self,
                 learning_rate=np.float32(0.001),
                 scope="Q_Estimator_Nonlinear",
                 summaries_dir=None):
        self.scope = scope
        self.summary_writer = None

        with tf.compat.v1.variable_scope(scope):
            self.state = tf.compat.v1.placeholder(shape=[None, n_steps, n_input_dim],
                                                 dtype=tf.float32, name="state")
            self.target = tf.compat.v1.placeholder(shape=[None, action_space_n],
                                                  dtype=tf.float32, name="target")

            lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(n_hidden_dim, forget_bias=1.0)
            state_unstack = tf.unstack(self.state, n_steps, axis=1)
            outputs, _ = tf.compat.v1.nn.static_rnn(lstm_cell, state_unstack, dtype=tf.float32)

            # Final dense layer
            self.W_out = tf.compat.v1.get_variable("W_out", shape=[n_hidden_dim, action_space_n],
                                                   initializer=tf.compat.v1.random_normal_initializer())
            self.b_out = tf.compat.v1.get_variable("b_out", shape=[action_space_n],
                                                   initializer=tf.compat.v1.constant_initializer(0.0))
            self.action_values = tf.matmul(outputs[-1], self.W_out) + self.b_out

            # Loss + optimizer
            self.losses = tf.compat.v1.squared_difference(self.action_values, self.target)
            self.loss = tf.reduce_mean(self.losses)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

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
        returns (batch_size, action_space_n)
        """
        sess = sess or tf.compat.v1.get_default_session()
        return sess.run(self.action_values, {self.state: state})

    def update(self, state, target, sess=None):
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
    Creates an epsilon-greedy policy for Q(s,a).
    """
    def policy_fn(observation, epsilon, sess=None):
        q_values = estimator.predict(np.expand_dims(observation, axis=0), sess=sess)[0]
        A = np.ones(nA, dtype='float32') * epsilon / nA
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

# ------------------------------------------------------------------------
# Replay Memory Transition
Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

# ------------------------------------------------------------------------
# Active Learning & Warm-Up
class active_learning(object):
    def __init__(self, env, N, strategy, estimator, already_selected):
        self.env = env
        self.N = N
        self.strategy = strategy
        self.estimator = estimator
        self.already_selected = set(already_selected)

    def get_samples(self, sess=None):
        states_list = self.env.states_list
        distances = []
        for state in states_list:
            st_reshaped = np.reshape(state, (n_steps, n_input_dim))
            q_vals = self.estimator.predict(np.expand_dims(st_reshaped, axis=0), sess=sess)[0]
            # margin = difference between top 2 Q-values
            sorted_q = np.sort(q_vals)
            margin = sorted_q[-1] - sorted_q[-2]
            distances.append(abs(margin))

        distances = np.array(distances)
        rank_ind = np.argsort(distances)  # ascending => smallest margin first
        rank_ind = [i for i in rank_ind if i not in self.already_selected]
        active_samples = rank_ind[:self.N]
        return active_samples

class WarmUp(object):
    def warm_up_SVM(self, outliers_fraction, X_train):
        model = OneClassSVM(gamma='auto', nu=0.95 * outliers_fraction)
        model.fit(X_train)
        return model

    def warm_up_isolation_forest(self, outliers_fraction, X_train):
        clf = IsolationForest(contamination=outliers_fraction, random_state=42)
        clf.fit(X_train)
        return clf

# ------------------------------------------------------------------------
# State Function
def RNNBinaryStateFuc(timeseries, timeseries_curser):
    """
    Return last n_steps of 'value' as shape (n_steps, n_input_dim).
    If there's not enough history, we zero-pad at the front.
    """
    if timeseries_curser < n_steps:
        pad_length = n_steps - timeseries_curser
        front_pad = np.zeros(pad_length)
        val_part = timeseries['value'][:timeseries_curser].values
        state_1d = np.concatenate([front_pad, val_part], axis=0)
    else:
        state_1d = timeseries['value'][timeseries_curser - n_steps : timeseries_curser].values

    return np.reshape(state_1d, (n_steps, n_input_dim))

# ------------------------------------------------------------------------
# Main Q-Learning
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
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    os.makedirs(checkpoint_dir, exist_ok=True)
    saver = tf.compat.v1.train.Saver()

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print(f"\nLoading model checkpoint {latest_checkpoint}...")
        saver.restore(sess, latest_checkpoint)
        if test:
            return

    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    replay_memory = deque(maxlen=replay_memory_size)
    policy = make_epsilon_greedy_policy(q_estimator, env.action_space_n)
    total_t = sess.run(q_estimator.global_step)

    # Warm-up using isolation forest
    print("Populating replay memory with a warm-up approach...\n")
    outliers_fraction = 0.01
    data_train = []
    for num in range(env.datasetsize):
        env.reset()
        data_train.extend(env.states_list)
    data_train_flat = [np.ravel(s) for s in data_train]
    data_train_flat = np.array(data_train_flat)

    warm_model = WarmUp().warm_up_isolation_forest(outliers_fraction, data_train_flat)
    steps_populated = 0
    while steps_populated < replay_memory_init_size:
        env.reset()
        for i in range(len(env.states_list)):
            if env.timeseries_curser < n_steps:
                continue
            state = env.states_list[i]
            state_flat = np.ravel(state).reshape(1, -1)
            anomaly_score = warm_model.decision_function(state_flat)[0]
            action = int(anomaly_score < 0.0)
            next_state, reward, done, _ = env.step(action)

            transition = Transition(
                state=state,
                action=action,
                reward=reward[action],
                next_state=next_state[action],
                done=done
            )
            replay_memory.append(transition)
            steps_populated += 1
            if done or steps_populated >= replay_memory_init_size:
                break

    print(f"Replay memory populated with {len(replay_memory)} transitions.\n")

    for i_episode in range(num_episodes):
        if i_episode % 50 == 0:
            saver.save(sess, checkpoint_path)
            print(f"Checkpoint saved at episode {i_episode}.")

        state = env.reset()
        done = False

        # Active Learning
        labeled_index = np.where(env.timeseries['label'] != -1)[0]
        al = active_learning(env, N=num_active_learning, strategy='margin_sampling',
                             estimator=q_estimator, already_selected=labeled_index)

        al_samples = al.get_samples(sess=sess)
        print(f"Active learning picked samples: {al_samples}")
        for idx in al_samples:
            env.timeseries.at[idx, 'label'] = env.timeseries.at[idx, 'anomaly']

        # Label Propagation
        lp_model = LabelSpreading()
        X = []
        y = []
        for i, st in enumerate(env.states_list):
            X.append(np.ravel(st))
            y_val = env.timeseries.at[i, 'label']
            y.append(y_val if y_val != -1 else -1)
        X = np.array(X)
        y = np.array(y, dtype=int)
        lp_model.fit(X, y)

        pred_entropies = stats.entropy(lp_model.label_distributions_.T)
        unlabeled_indices = np.where(y == -1)[0]
        certainty_index = np.argsort(pred_entropies)[:num_LabelPropagation]
        for cid in certainty_index:
            if cid in unlabeled_indices:
                pseudo_label = lp_model.transduction_[cid]
                env.timeseries.at[cid, 'label'] = pseudo_label

        # mini-epoch
        for epoch_step in range(num_epoches):
            idx_list = list(range(len(env.states_list)))
            random.shuffle(idx_list)
            for i_idx in idx_list:
                if i_idx < n_steps:
                    continue
                env.timeseries_curser = i_idx
                epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
                st = env.states_list[i_idx]
                act_probs = policy(st, epsilon, sess=sess)
                action = np.random.choice(np.arange(action_space_n), p=act_probs)

                next_state, reward, done, _ = env.step(action)
                replay_memory.append(Transition(
                    state=st,
                    action=action,
                    reward=reward[action],
                    next_state=next_state[action],
                    done=done
                ))

                if len(replay_memory) < batch_size:
                    continue
                samples = random.sample(replay_memory, batch_size)
                states_batch = []
                next_states_batch = []
                target_batch = []

                for t_ in samples:
                    states_batch.append(t_.state)
                    next_states_batch.append(t_.next_state)

                states_batch = np.array(states_batch)
                next_states_batch = np.array(next_states_batch)
                q_next = target_estimator.predict(next_states_batch, sess=sess)
                q_current = q_estimator.predict(states_batch, sess=sess)

                for i_b, t_ in enumerate(samples):
                    if t_.done:
                        td_target = t_.reward
                    else:
                        td_target = t_.reward + discount_factor * np.max(q_next[i_b])
                    updated = np.array(q_current[i_b])
                    updated[t_.action] = td_target
                    target_batch.append(updated)

                target_batch = np.array(target_batch)
                loss_val = q_estimator.update(states_batch, target_batch, sess=sess)
                total_t += 1

                if total_t % update_target_estimator_every == 0:
                    copy_model_parameters(sess, q_estimator, target_estimator)
                    print("Copied model parameters to target network.")

        print(f"Episode {i_episode} ended. Global step = {total_t}")

    saver.save(sess, checkpoint_path)
    print("Final model saved.")

# ------------------------------------------------------------------------
# Evaluation
def evaluate_model(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall    = recall_score(y_true, y_pred, zero_division=1)
    f1        = f1_score(y_true, y_pred, zero_division=1)
    mcc       = matthews_corrcoef(y_true, y_pred)
    balanced_acc = (recall + (tn / (tn + fp))) / 2 if (tn+fp)>0 else recall
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
    Validate the model by running greedy policy on test portion of environment.
    """
    y_true_all = []
    y_pred_all = []

    policy = make_epsilon_greedy_policy(estimator, env.action_space_n)

    for i_episode in range(num_episodes):
        print(f"\nValidation Episode {i_episode+1}/{num_episodes}")
        env.reset()

        # skip until test portion
        while env.datasetidx < env.datasetrng * validation_separate_ratio:
            env.reset()

        done = False
        for t in itertools.count():
            st = env.statefnc(env.timeseries, env.timeseries_curser)
            action_probs = policy(st, 0.0)  # greedy
            action = np.argmax(action_probs)

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
        plt.figure(figsize=(8,4))
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

# ------------------------------------------------------------------------
# Training Wrapper
def train(num_LP=20, num_AL=5, discount_factor=0.92, learn_tau=True):
    """
    Full pipeline:
      1) Load & scale train data
      2) Train or load VAE
      3) Create environment
      4) Q-learning
      5) Validation
    """
    # 1) Load data
    x_train = load_normal_data_kpi(kpi_train_csv, exclude_columns=['timestamp','label','KPI ID'])
    df_test = load_test_data_kpi(kpi_test_hdf)

    # 2) Build and train VAE
    # If you want the VAE to handle single points, set original_dim=1
    # If you want sliding windows in the VAE, you'd need original_dim=n_steps and
    # also reshape x_train into windows. We'll keep it simple as single points.
    original_dim = 1
    latent_dim   = 2
    intermediate_dim = 16

    vae, encoder = build_vae(original_dim, latent_dim, intermediate_dim)
    print("\nTraining VAE on single points (original_dim=1)...")
    vae.fit(x_train, x_train, epochs=10, batch_size=32, validation_split=0.1)

    vae_save_path = os.path.join(current_dir, 'vae_model_kpi.h5')
    vae.save(vae_save_path)
    print(f"VAE saved to {vae_save_path}")

    # 3) Create environment
    env = EnvTimeSeriesfromRepo()
    env.set_train_data(x_train)
    env.set_test_data(df_test)

    if learn_tau:
        env.rewardfnc = lambda ts, cur, act: RNNBinaryRewardFuc(ts, cur, act, vae=vae)
    else:
        env.rewardfnc = RNNBinaryRewardFucTest

    env.statefnc = RNNBinaryStateFuc
    env.timeseries_curser_init = n_steps
    env.datasetfix = DATAFIXED
    env.datasetidx = 0

    # test environment
    env_test = EnvTimeSeriesfromRepo()
    env_test.set_train_data(x_train)
    env_test.set_test_data(df_test)
    env_test.rewardfnc = RNNBinaryRewardFucTest
    env_test.statefnc  = RNNBinaryStateFuc

    # 4) Setup Q Estimators + train
    exp_dir = os.path.abspath("./exp/AdaptiveTauRL/")
    os.makedirs(exp_dir, exist_ok=True)
    tf.compat.v1.reset_default_graph()

    q_estimator = Q_Estimator_Nonlinear(scope="qlearn", summaries_dir=exp_dir, learning_rate=3e-4)
    target_estimator = Q_Estimator_Nonlinear(scope="target", learning_rate=3e-4)

    sess = tf.compat.v1.Session()
    with sess.as_default():
        sess.run(tf.compat.v1.global_variables_initializer())

        q_learning(env=env,
                   sess=sess,
                   q_estimator=q_estimator,
                   target_estimator=target_estimator,
                   num_episodes=20,
                   num_epoches=5,
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

# ------------------------------------------------------------------------
# Main
if __name__ == "__main__":
    try:
        print("\n--- Starting Training Run ---")
        results = train(num_LP=100, num_AL=30, discount_factor=0.92, learn_tau=True)
        print("Final Results:", results)
    except Exception as e:
        print("Error during training:", e)

    # --------------------------------------------------------------------
    # Multiple Training Runs Example (uncomment if you want more):
    """
    if __name__ == "__main__":
        try:
            print("\n--- Training Run 1: (LP=100, AL=30, discount=0.92) ---")
            metric1 = train(num_LP=100, num_AL=30, discount_factor=0.92, learn_tau=True)
            print("Run 1 Metrics:", metric1)

            print("\n--- Training Run 2: (LP=150, AL=50, discount=0.94) ---")
            metric2 = train(num_LP=150, num_AL=50, discount_factor=0.94, learn_tau=True)
            print("Run 2 Metrics:", metric2)

            print("\n--- Training Run 3: (LP=200, AL=100, discount=0.96) ---")
            metric3 = train(num_LP=200, num_AL=100, discount_factor=0.96, learn_tau=True)
            print("Run 3 Metrics:", metric3)

        except Exception as e:
            print("Error in multiple runs:", e)
    """
