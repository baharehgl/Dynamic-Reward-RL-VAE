import matplotlib

# Use 'Agg' backend for environments without display (e.g., servers)
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
from mpl_toolkits.mplot3d import axes3d
from collections import deque, namedtuple
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

# Disable eager execution for TensorFlow 1.x compatibility
tf.compat.v1.disable_eager_execution()

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")
# Get the parent directory of the current directory
parent_dir = os.path.dirname(current_dir)
# Append the parent directory to the sys.path list for module imports
sys.path.append(parent_dir)

# Import the custom environment
from environment.time_series_repo_ext import EnvTimeSeriesfromRepo

# Specify CUDA devices for GPU acceleration
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

# ========================== Hyperparameters ==========================
# Training Parameters
DATAFIXED = 0  # Whether target is at a single time series dataset
EPISODES = 500  # Number of episodes for training
DISCOUNT_FACTOR = 0.5  # Reward discount factor [0,1]
EPSILON = 0.5  # Epsilon-greedy method parameter for action selection
EPSILON_DECAY = 1.00  # Epsilon-greedy method decay parameter

# Action Definitions
NOT_ANOMALY = 0
ANOMALY = 1
action_space = [NOT_ANOMALY, ANOMALY]
action_space_n = len(action_space)

# Environment and Model Parameters
n_steps = 50  # Size of the sliding window for SLIDE_WINDOW state and reward functions
n_input_dim = 2  # Dimension of the input for an LSTM cell
n_hidden_dim = 128  # Dimension of the hidden state in LSTM cell

# Reward Values
TP_Value = 5
TN_Value = 1
FP_Value = -1
FN_Value = -5

validation_separate_ratio = 0.9  # Ratio to separate validation data


# ========================== VAE Setup ================================
def load_normal_data(data_path, n_steps=50):
    """
    Load and concatenate all CSV files from the specified directory into overlapping sequences.

    Args:
        data_path (str): Path to the directory containing CSV files.
        n_steps (int): Number of steps in each sequence.

    Returns:
        np.ndarray: Scaled sequences with shape (samples, n_steps, features).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory {data_path} does not exist.")

    all_files = [os.path.join(data_path, fname) for fname in os.listdir(data_path) if fname.endswith('.csv')]
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in directory {data_path}.")

    print(f"Found {len(all_files)} CSV files in {data_path}.")

    data_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            if df.shape[0] < n_steps:
                print(f"Skipping {file} as it has less than {n_steps} rows.")
                continue
            data_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not data_list:
        raise ValueError(f"No valid data found in {data_path}. Ensure CSV files have at least {n_steps} rows.")

    data = pd.concat(data_list, axis=0, ignore_index=True)
    print(f"Total data shape after concatenation: {data.shape}")

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values)

    # Create overlapping sequences
    sequences = []
    for i in range(len(scaled_data) - n_steps + 1):
        seq = scaled_data[i:i + n_steps]
        sequences.append(seq)

    sequences = np.array(sequences)
    print(f"Generated {sequences.shape[0]} sequences of shape {sequences.shape[1:]} from data.")
    return sequences


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae(original_dim, latent_dim=2, intermediate_dim=64):
    """
    Build a Variational Autoencoder (VAE) model.

    Args:
        original_dim (int): Dimension of the input data.
        latent_dim (int): Dimension of the latent space.
        intermediate_dim (int): Dimension of the intermediate layers.

    Returns:
        tuple: VAE model and encoder model.
    """
    inputs = layers.Input(shape=(original_dim,))
    h = layers.Dense(intermediate_dim, activation='relu')(inputs)
    h = layers.Dense(intermediate_dim, activation='relu')(h)
    h = layers.Dense(intermediate_dim, activation='relu')(h)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    z = Sampling()([z_mean, z_log_var])

    # Decoder
    decoder_h = layers.Dense(intermediate_dim, activation='relu')(z)
    decoder_h = layers.Dense(intermediate_dim, activation='relu')(decoder_h)
    decoder_h = layers.Dense(intermediate_dim, activation='relu')(decoder_h)
    decoder_mean = layers.Dense(original_dim, activation='sigmoid')(decoder_h)

    # Models
    encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    vae = models.Model(inputs, decoder_mean, name='vae')

    # Losses
    reconstruction_loss = losses.binary_crossentropy(inputs, decoder_mean) * original_dim
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    print("VAE model built and compiled.")
    return vae, encoder


# Constants for VAE
original_dim = 3  # Depends on the dimension of your input data
latent_dim = 10
intermediate_dim = 64


# ========================== Q-Learning Components =========================
class Q_Estimator_Nonlinear():
    """
    Action-Value Function Approximator Q(s,a) with TensorFlow RNN.
    Uses TensorFlow's native LSTMCell and dynamic_rnn for better compatibility.
    """

    def __init__(self, learning_rate=np.float32(0.01), scope="Q_Estimator_Nonlinear", summaries_dir=None):
        self.scope = scope
        self.summary_writer = None

        with tf.compat.v1.variable_scope(scope):
            # TensorFlow Graph inputs
            self.state = tf.compat.v1.placeholder(shape=[None, n_steps, n_input_dim],
                                                  dtype=tf.float32, name="state")
            self.target = tf.compat.v1.placeholder(shape=[None, action_space_n],
                                                   dtype=tf.float32, name="target")

            # Define weights and biases for output layer
            self.weights = {
                'out': tf.Variable(tf.compat.v1.random_normal([n_hidden_dim, action_space_n]))
            }
            self.biases = {
                'out': tf.Variable(tf.compat.v1.random_normal([action_space_n]))
            }

            # Define an LSTM cell using TensorFlow's native API
            lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(n_hidden_dim, forget_bias=1.0)

            # Use dynamic_rnn for processing the input sequences
            outputs, states = tf.compat.v1.nn.dynamic_rnn(lstm_cell, self.state,
                                                          dtype=tf.float32)

            # Extract the last output for each sequence
            last_output = outputs[:, -1, :]  # Shape: [batch_size, n_hidden_dim]

            # Compute action values
            self.action_values = tf.matmul(last_output, self.weights['out']) + self.biases['out']

            # Define the loss and optimizer
            self.losses = tf.compat.v1.squared_difference(self.action_values, self.target)
            self.loss = tf.reduce_mean(self.losses)

            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss,
                                                    global_step=tf.compat.v1.train.get_or_create_global_step())

            # Summaries for TensorBoard
            self.summaries = tf.compat.v1.summary.merge([
                tf.compat.v1.summary.histogram("loss_hist", self.losses),
                tf.compat.v1.summary.scalar("loss", self.loss),
                tf.compat.v1.summary.histogram("q_values_hist", self.action_values),
                tf.compat.v1.summary.scalar("q_value", tf.reduce_max(self.action_values))
            ])

            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                    print(f"Created summary directory at {summary_dir}")
                self.summary_writer = tf.compat.v1.summary.FileWriter(summary_dir)
                print(f"Summary writer initialized at {summary_dir}")

        print(f"Q_Estimator_Nonlinear '{scope}' initialized.")

    def predict(self, state, sess=None):
        """Predict Q-values for given states."""
        sess = sess or tf.compat.v1.get_default_session()
        return sess.run(self.action_values, {self.state: state})

    def update(self, state, target, sess=None):
        """Update the Q-network based on states and target Q-values."""
        sess = sess or tf.compat.v1.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        summaries, global_step, _ = sess.run([self.summaries,
                                              tf.compat.v1.train.get_or_create_global_step(),
                                              self.train_op], feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return


def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.

    Args:
      sess: TensorFlow session instance
      estimator1: Estimator to copy the parameters from
      estimator2: Estimator to copy the parameters to
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
    print(f"Copied parameters from '{estimator1.scope}' to '{estimator2.scope}'.")


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns Q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation and epsilon as arguments and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """

    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype='float32') * epsilon / nA
        q_values = estimator.predict(state=[observation])
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


# ========================== Reward Functions ==========================
def kl_divergence(p, q):
    """Compute the KL divergence KL(p || q)."""
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))


def adaptive_scaling_factor_dro(preference_strength, tau_min=0.1, tau_max=5.0, rho=0.5, max_iter=10):
    """
    Learn tau dynamically using DRO (KL-constrained optimization).

    Args:
        preference_strength (float): Probability of preference (sigmoid of reconstruction error).
        tau_min (float): Minimum bound for tau.
        tau_max (float): Maximum bound for tau.
        rho (float): Regularization term for DRO.
        max_iter (int): Maximum iterations for optimizing tau.

    Returns:
        float: Optimized tau.
    """
    tau = 1.0  # Initialize tau

    for _ in range(max_iter):
        # Compute KL divergence constraint
        kl_term = kl_divergence([preference_strength, 1 - preference_strength], [0.5, 0.5])

        # Gradient of the DRO loss w.r.t. tau
        grad = -np.log(1 + np.exp(-preference_strength / tau)) + rho - kl_term

        # Hessian (second derivative)
        hess = (preference_strength ** 2) * np.exp(-preference_strength / tau) / (
                tau ** 3 * (1 + np.exp(-preference_strength / tau)) ** 2
        )

        # Newton's update step for tau
        tau = tau - grad / (hess + 1e-8)

        # Project tau to be within valid bounds
        tau = np.clip(tau, tau_min, tau_max)

    return tau


# List to store tau values for visualization
tau_values = []


def RNNBinaryRewardFuc(timeseries, timeseries_curser, action=0, vae=None, scale_factor=10):
    """
    Compute the reward based on the current action and state, incorporating APS.

    Args:
        timeseries (DataFrame): The time series data.
        timeseries_curser (int): Current position in the time series.
        action (int): Action taken (0 or 1).
        vae (Model): Trained VAE model.
        scale_factor (float): Scaling factor for the VAE penalty.

    Returns:
        list: Reward values for actions [NOT_ANOMALY, ANOMALY].
    """
    if timeseries_curser >= n_steps:
        # Extract the current time step for VAE reconstruction
        current_time_step = timeseries['value'][timeseries_curser]
        current_state = np.array([current_time_step])  # Shape: (1, 3)

        # Predict reconstruction
        vae_reconstruction = vae.predict(current_state)
        reconstruction_error = np.mean(np.square(vae_reconstruction - current_state))

        # Intrinsic reward component (VAE penalty)
        vae_penalty = -scale_factor * reconstruction_error

        # Compute preference strength using sigmoid activation
        preference_strength = np.clip(1 / (1 + np.exp(-reconstruction_error)), 0.05, 0.95)

        # Compute adaptive scaling factor tau using DRO
        tau = adaptive_scaling_factor_dro(preference_strength)
        tau_values.append(tau)  # Store tau for visualization

        # Retrieve the true label
        true_label = timeseries['label'][timeseries_curser]

        # Define rewards based on the true label and action
        if true_label == NOT_ANOMALY:
            # If true label is NOT_ANOMALY
            reward_not_anomaly = TN_Value + tau * vae_penalty
            reward_anomaly = FP_Value + tau * vae_penalty
            return [reward_not_anomaly, reward_anomaly]
        elif true_label == ANOMALY:
            # If true label is ANOMALY
            reward_not_anomaly = FN_Value + tau * vae_penalty
            reward_anomaly = TP_Value + tau * vae_penalty
            return [reward_not_anomaly, reward_anomaly]
    else:
        return [0, 0]  # No reward for initial steps


def RNNBinaryRewardFucTest(timeseries, timeseries_curser, action=0):
    """
    Reward function for testing without APS.

    Args:
        timeseries (DataFrame): The time series data.
        timeseries_curser (int): Current position in the time series.
        action (int): Action taken (0 or 1).

    Returns:
        list: Reward values for actions [NOT_ANOMALY, ANOMALY].
    """
    if timeseries_curser >= n_steps:
        if timeseries['anomaly'][timeseries_curser] == NOT_ANOMALY:
            return [TN_Value, FP_Value]
        elif timeseries['anomaly'][timeseries_curser] == ANOMALY:
            return [FN_Value, TP_Value]
    else:
        return [0, 0]


# ========================== State Function ================================
def RNNBinaryStateFuc(timeseries, timeseries_curser, previous_state=[], action=None):
    """
    Generate the current state for the RL agent based on the time series.

    Args:
        timeseries (DataFrame): The time series data.
        timeseries_curser (int): Current position in the time series.
        previous_state (list): Previous state (if any).
        action (int): Previous action taken (if any).

    Returns:
        np.ndarray: Current state representation.
    """
    if timeseries_curser == n_steps:
        # Initial state
        state = []
        for i in range(timeseries_curser):
            state.append([timeseries['value'][i], 0])  # Action flag set to 0
        state.pop(0)  # Remove the first element to maintain window size
        state.append([timeseries['value'][timeseries_curser], 1])  # Latest data point with action flag
        return np.array(state, dtype='float32')

    if timeseries_curser > n_steps:
        # Two possible next states based on the current action
        state0 = np.concatenate((previous_state[1:n_steps],
                                 [[timeseries['value'][timeseries_curser], 0]]))
        state1 = np.concatenate((previous_state[1:n_steps],
                                 [[timeseries['value'][timeseries_curser], 1]]))
        return np.array([state0, state1], dtype='float32')


# ========================== Warm-Up Strategies ============================
class WarmUp(object):
    """
    Warm-up strategies for initializing replay memory.
    """

    def warm_up_isolation_forest(self, outliers_fraction, X_train):
        """
        Initialize and train Isolation Forest for anomaly detection.

        Args:
            outliers_fraction (float): Proportion of outliers in the data.
            X_train (np.ndarray): Training data.

        Returns:
            IsolationForest: Trained Isolation Forest model.
        """
        from sklearn.ensemble import IsolationForest
        # X_train is assumed to be (samples, n_steps, features)
        # Extract the last time step's features
        if X_train.ndim != 3:
            raise ValueError(f"Expected X_train to have 3 dimensions, got {X_train.ndim}")

        data = X_train[:, -1, :]  # Shape: (samples, features)
        print(f"Data shape for IsolationForest: {data.shape}")  # Debugging
        clf = IsolationForest(contamination=outliers_fraction, random_state=42)
        clf.fit(data)
        print("Isolation Forest trained.")
        return clf


# ========================== Q-Learning Algorithm ==========================
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
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.

    Args:
        env: The environment.
        sess: TensorFlow session.
        qlearn_estimator: Q-learning estimator.
        target_estimator: Target estimator.
        num_episodes: Number of episodes to run for.
        num_epoches: Number of epochs per episode.
        replay_memory_size: Maximum size of replay memory.
        replay_memory_init_size: Initial size of replay memory.
        experiment_dir: Directory for experiment logs.
        update_target_estimator_every: Steps between target updates.
        discount_factor: Discount factor for RL.
        epsilon_start: Starting value of epsilon.
        epsilon_end: Ending value of epsilon.
        epsilon_decay_steps: Steps over which epsilon decays.
        batch_size: Batch size for training.
        num_LabelPropagation: Number of samples for label propagation.
        num_active_learning: Number of active learning samples.
        test: Flag for testing mode.
        vae_model: Pretrained VAE model.

    Returns:
        None
    """
    # Define a named tuple for storing experiences
    Transition = namedtuple("Transition", ["state", "reward", "next_state", "done"])

    # The replay memory
    replay_memory = deque(maxlen=replay_memory_size)

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        print(f"Created checkpoint directory at {checkpoint_dir}")
    else:
        print(f"Checkpoint directory exists at {checkpoint_dir}")

    # Initialize saver for checkpoints
    saver = tf.compat.v1.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
        if test:
            print("Test flag is set. Exiting after loading checkpoint.")
            return
    else:
        print("No checkpoint found. Starting fresh training.")

    # Get the current global step
    global_step = tf.compat.v1.train.get_or_create_global_step()
    sess.run(global_step.assign(0))  # Initialize global_step to 0
    print("Global step initialized to 0.")

    # Epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # The policy we're following
    policy = make_epsilon_greedy_policy(
        qlearn_estimator,
        env.action_space_n
    )

    num_label = 0  # Counter for labeled samples
    total_t = 0  # Total training steps

    # 2. Populate the replay memory with initial experience by Isolation Forest
    popu_time = time.time()

    # Warm up with active learning
    print('Warm up starting...')

    outliers_fraction = 0.01
    data_train = []
    for num in range(env.datasetsize):
        env.reset()
        print(f"After reset {num + 1}/{env.datasetsize}, states_list length: {len(env.states_list)}")
        if not env.states_list:
            print(f"Warning: states_list is empty after reset {num + 1}")
        else:
            data_train.extend(env.states_list)

    data_train = np.array(data_train)
    print(f"data_train shape after extending: {data_train.shape}")

    if data_train.size == 0:
        raise ValueError(
            "No data collected in replay memory. Ensure that 'env.reset()' populates 'env.states_list' correctly.")

    # Reshape data_train if necessary
    if data_train.ndim == 2:
        data_train = data_train.reshape(data_train.shape[0], 1, data_train.shape[1])
        print(f"data_train reshaped to: {data_train.shape}")
    elif data_train.ndim == 3:
        print(f"data_train already has correct shape: {data_train.shape}")
    else:
        raise ValueError(f"Unexpected data_train shape: {data_train.shape}")

    # Initialize Isolation Forest model
    model = WarmUp().warm_up_isolation_forest(outliers_fraction, data_train)
    print("Isolation Forest model trained.")

    # Initialize Label Propagation model
    lp_model = LabelSpreading()
    print("Label Spreading model initialized.")

    for t in itertools.count():
        env.reset()
        print(f"Environment reset during replay memory population at iteration {t + 1}")
        data = np.array(env.states_list)
        print(f"Collected data shape from env.states_list: {data.shape}")

        if data.ndim == 2:
            data = data.reshape(1, data.shape[0], data.shape[1])
            print(f"Reshaped data to: {data.shape}")
        elif data.ndim == 3:
            print(f"Data already has correct shape: {data.shape}")
        else:
            raise ValueError(f"Unexpected env.states_list shape: {data.shape}")

        # Extract last time step's features for anomaly detection
        data_last = data[:, -1, :]  # Shape: (samples, features)
        print(f"Data for IsolationForest decision_function: {data_last.shape}")
        anomaly_score = model.decision_function(data_last)  # e.g., [-0.5, 0.5]
        print(f"Anomaly scores: {anomaly_score}")

        pred_score = [-1 * s + 0.5 for s in anomaly_score]  # Transform to [0, 1]
        print(f"Transformed anomaly scores (pred_score): {pred_score}")

        # Select top and bottom samples based on anomaly scores
        warm_samples = np.argsort(pred_score)[:5]
        warm_samples = np.append(warm_samples, np.argsort(pred_score)[-5:])
        print(f"Selected warm_samples indices: {warm_samples}")

        # Initialize label list with unlabeled (-1)
        state_list = np.array(env.states_list)
        state_list = state_list.reshape(state_list.shape[0], -1)  # Flatten if necessary
        label_list = [-1] * len(state_list)  # Remove labels initially

        for sample in warm_samples:
            if sample >= len(env.states_list):
                print(f"Sample index {sample} out of bounds. Skipping.")
                continue
            # Pick a state from warm_up samples
            state = env.states_list[sample]
            # Update the cursor
            env.timeseries_curser = sample + n_steps
            action_probs = policy(state, epsilons[min(total_t, epsilon_decay_steps - 1)])
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            print(f"Selected action {action} for sample {sample}")

            # Assign the true label to the current position
            true_label = env.timeseries['anomaly'][env.timeseries_curser]
            env.timeseries['label'][env.timeseries_curser] = true_label
            num_label += 1
            print(f"Assigned true label {true_label} to position {env.timeseries_curser}")

            # Retrieve label for propagation
            label_list[sample] = int(true_label)

            # Take a step in the environment
            next_state, reward, done, _ = env.step(action)
            print(f"Added transition to replay memory: reward={reward}, done={done}")

            # Add experience to replay memory
            replay_memory.append(Transition(state, reward, next_state, done))
            print(f"Replay memory size: {len(replay_memory)}")

        # Label propagation main process:
        unlabeled_indices = [i for i, e in enumerate(label_list) if e == -1]
        print(f"Unlabeled indices before label propagation: {unlabeled_indices}")
        label_list = np.array(label_list)
        lp_model.fit(state_list, label_list)
        print("Label Spreading model fitted.")

        pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)
        print(f"Predicted entropies: {pred_entropies}")

        # Select up to N samples that are most certain about
        certainty_index = np.argsort(pred_entropies)
        certainty_index = certainty_index[np.in1d(certainty_index, unlabeled_indices)][:num_LabelPropagation]
        print(f"Selected certainty_index for pseudo-labeling: {certainty_index}")

        # Assign pseudo labels
        for index in certainty_index:
            pseudo_label = lp_model.transduction_[index]
            env.timeseries['label'][index + n_steps] = pseudo_label
            print(f"Assigned pseudo label {pseudo_label} to position {index + n_steps}")

        # Break the loop if enough experiences are collected
        if len(replay_memory) >= replay_memory_init_size:
            print(f"Replay memory initialized with {len(replay_memory)} transitions.")
            break

    popu_time = time.time() - popu_time
    print("Populating replay memory completed in {:.2f} seconds".format(popu_time))

    # 3. Start the main training loop
    for i_episode in range(num_episodes):
        # Save the current checkpoint periodically
        if i_episode % 50 == 49:
            print(f"Save checkpoint in episode {i_episode + 1}/{num_episodes}")
            saver.save(sess, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

        per_loop_time1 = time.time()

        # Reset the environment
        state = env.reset()
        print(f"Episode {i_episode + 1} started.")
        while env.datasetidx > env.datasetrng * validation_separate_ratio:
            env.reset()
            print('Double reset due to dataset index exceeding validation separation ratio.')

        # Active learning:
        # Index of already labeled samples of this TS
        labeled_index = [i for i, e in enumerate(env.timeseries['label']) if e != -1]
        # Transform to match state_list
        labeled_index = [item for item in labeled_index if item >= n_steps]
        labeled_index = [item - n_steps for item in labeled_index]
        print(f"Already labeled indices: {labeled_index}")

        # Initialize active learning
        al = active_learning(env=env, N=num_active_learning, strategy='margin_sampling',
                             estimator=qlearn_estimator, already_selected=labeled_index)
        # Find the samples that need to be labeled by human
        al_samples = al.get_samples()
        print(f'Labeling samples: {al_samples} in env {env.datasetidx}')
        # Add the new labeled samples
        labeled_index.extend(al_samples)
        num_label += len(al_samples)
        print(f"Total labeled samples: {num_label}")

        # Retrieve input for label propagation
        state_list = np.array(env.states_list)
        state_list = state_list.reshape(state_list.shape[0], -1)  # Flatten if necessary
        label_list = np.array(env.timeseries['label'][n_steps:])

        for new_sample in al_samples:
            # Assign true label from environment
            label_list[new_sample] = env.timeseries['anomaly'][n_steps + new_sample]
            env.timeseries['label'][n_steps + new_sample] = env.timeseries['anomaly'][n_steps + new_sample]
            print(f"Assigned true label {env.timeseries['anomaly'][n_steps + new_sample]} to sample {new_sample}")

        for samples in labeled_index:
            env.timeseries_curser = samples + n_steps
            # Epsilon for this time step
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
            print(f"Training step {total_t}: using epsilon={epsilon}")

            state = env.states_list[samples]
            print(f"Selected action for sample {samples}: state shape {state.shape}")

            # Choose an action to take
            action_probs = policy(state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            print(f"Chosen action: {action}")

            # Take a step
            next_state, reward, done, _ = env.step(action)
            print(f"Step taken: reward={reward}, done={done}")

            # Control replay memory size
            if len(replay_memory) == replay_memory_size:
                replay_memory.popleft()
                print("Replay memory full. Removed oldest transition.")

            # Add experience to replay memory
            replay_memory.append(Transition(state, reward, next_state, done))
            print(f"Added transition to replay memory. Current size: {len(replay_memory)}")

        # Label propagation main process:
        unlabeled_indices = [i for i, e in enumerate(label_list) if e == -1]
        print(f"Unlabeled indices before label propagation: {unlabeled_indices}")
        label_list = np.array(label_list)
        lp_model.fit(state_list, label_list)
        print("Label Spreading model fitted.")

        pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)
        print(f"Predicted entropies: {pred_entropies}")

        # Select up to N samples that are most certain about
        certainty_index = np.argsort(pred_entropies)
        certainty_index = certainty_index[np.in1d(certainty_index, unlabeled_indices)][:num_LabelPropagation]
        print(f"Selected certainty_index for pseudo-labeling: {certainty_index}")

        # Assign pseudo labels
        for index in certainty_index:
            pseudo_label = lp_model.transduction_[index]
            env.timeseries['label'][index + n_steps] = pseudo_label
            print(f"Assigned pseudo label {pseudo_label} to position {index + n_steps}")

        per_loop_time2 = time.time()

        # Update the model
        for i_epoch in range(num_epoches):
            # Add epsilon to TensorBoard (optional)
            if qlearn_estimator.summary_writer:
                episode_summary = tf.compat.v1.Summary()
                qlearn_estimator.summary_writer.add_summary(episode_summary, total_t)

            # Update the target estimator periodically
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, qlearn_estimator, target_estimator)
                print("\nCopied model parameters to target network.\n")

            # Sample a minibatch from the replay memory
            if len(replay_memory) < batch_size:
                batch_size_current = len(replay_memory)
            else:
                batch_size_current = batch_size
            samples = random.sample(replay_memory, batch_size_current)
            states_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
            print(f"Training on a batch of size {batch_size_current}")

            if discount_factor > 0:
                # Calculate Q values and targets
                # Assuming next_states_batch contains possible next states for actions
                try:
                    next_states_split = np.split(next_states_batch, 2, axis=1)
                    next_states_batch0 = next_states_split[0].reshape(-1, n_steps, n_input_dim)
                    next_states_batch1 = next_states_split[1].reshape(-1, n_steps, n_input_dim)
                    print(f"Split next_states_batch into {next_states_batch0.shape} and {next_states_batch1.shape}")
                except ValueError as e:
                    print(f"Error splitting next_states_batch: {e}")
                    continue

                q_values_next0 = target_estimator.predict(state=next_states_batch0)
                q_values_next1 = target_estimator.predict(state=next_states_batch1)
                print(f"Predicted Q-values for next states.")

                # Compute the maximum Q-value for next states
                max_q_next = np.maximum(np.amax(q_values_next0, axis=1),
                                        np.amax(q_values_next1, axis=1))
                print(f"Computed max Q-values for next states.")

                # Compute target Q-values
                targets_batch = reward_batch + (discount_factor * max_q_next.reshape(-1, 1))
                print(f"Computed target Q-values.")
            else:
                targets_batch = reward_batch

            # Perform gradient descent update
            qlearn_estimator.update(state=states_batch, target=targets_batch.astype(np.float32))
            print(f"Updated Q-estimator.")

            total_t += 1

        per_loop_time_updt = time.time() - per_loop_time2
        print("Global step {} @ Episode {}/{}, time: {:.2f} + {:.2f} seconds"
              .format(total_t, i_episode + 1, num_episodes, per_loop_time2 - per_loop_time1, per_loop_time_updt))
    return


# ========================== Evaluation Metrics ============================
def evaluate_model(y_true, y_pred):
    """
    Compute various classification metrics.

    Args:
        y_true (list): Ground truth labels (0 = Normal, 1 = Anomaly)
        y_pred (list): Predicted labels (0 = Normal, 1 = Anomaly)

    Returns:
        dict: Dictionary with evaluation metrics.
    """
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Compute existing metrics
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)

    # Compute additional metrics
    mcc = matthews_corrcoef(y_true, y_pred)  # Matthews Correlation Coefficient
    balanced_acc = (recall + (tn / (tn + fp))) / 2  # Balanced Accuracy
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

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
    Validate the trained model using multiple evaluation metrics.

    Args:
        env: Environment.
        estimator: Trained model.
        num_episodes (int): Number of validation episodes.
        record_dir (str, optional): Directory to record performance. Defaults to None.
        plot (int, optional): Flag to plot results. Defaults to 1.

    Returns:
        dict: Averaged evaluation metrics.
    """
    y_true_all = []
    y_pred_all = []

    for i_episode in range(num_episodes):
        state_rec = []
        action_rec = []

        # Define a greedy policy (epsilon=0)
        policy = make_epsilon_greedy_policy(estimator, env.action_space_n)
        state = env.reset()
        while env.datasetidx < env.datasetrng * validation_separate_ratio:
            state = env.reset()

        for t in itertools.count():
            action_probs = policy(state, 0)  # Use greedy policy
            action = np.argmax(action_probs)

            y_true_all.append(env.timeseries['anomaly'][env.timeseries_curser])  # True label
            y_pred_all.append(action)  # Predicted label

            next_state, reward, done, _ = env.step(action)
            if done:
                break
            state = next_state[action]

    # Compute evaluation metrics
    results = evaluate_model(y_true_all, y_pred_all)

    # Print metrics
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    return results


# ========================== Active Learning ===============================
class active_learning(object):
    """
    Active Learning class for selecting samples to label.
    """

    def __init__(self, env, N, strategy, estimator, already_selected):
        self.env = env
        self.N = N
        self.strategy = strategy
        self.estimator = estimator
        self.already_selected = already_selected

    def get_samples(self):
        """
        Select samples based on the margin sampling strategy.

        Returns:
            list: Indices of selected samples.
        """
        states_list = self.env.states_list
        distances = []
        for state in states_list:
            q = self.estimator.predict(state=[state])[0]
            distance = abs(q[0] - q[1])
            distances.append(distance)
        distances = np.array(distances)

        # Margin sampling: select samples with smallest distance (most uncertain)
        min_margin = distances
        rank_ind = np.argsort(min_margin)
        rank_ind = [i for i in rank_ind if i not in self.already_selected]
        active_samples = rank_ind[:self.N]
        print(f"Selected active samples for labeling: {active_samples}")
        return active_samples

    def get_samples_by_score(self, threshold):
        """
        Select samples based on a score threshold.

        Args:
            threshold (float): Threshold for selection.

        Returns:
            list: Indices of selected samples.
        """
        states_list = self.env.states_list
        distances = []
        for state in states_list:
            q = self.estimator.predict(state=[state])[0]
            distance = abs(q[0] - q[1])
            distances.append(distance)
        distances = np.array(distances)

        # Select samples with distance below threshold
        rank_ind = np.argsort(distances)
        rank_ind = [i for i in rank_ind if i not in self.already_selected]
        active_samples = [t for t in rank_ind if distances[t] < threshold]
        print(f"Selected samples by score threshold {threshold}: {active_samples}")
        return active_samples

    def label(self, active_samples):
        """
        Manual labeling of selected samples.

        Args:
            active_samples (list): Indices of samples to label.

        Returns:
            None
        """
        for sample in active_samples:
            print('AL finds one of the most confused samples:')
            print(self.env.timeseries['value'].iloc[sample:sample + n_steps])
            print('Please label the last timestamp based on your knowledge')
            print('0 for non-anomaly; 1 for anomaly')
            label = input()
            self.env.timeseries.loc[sample + n_steps - 1, 'anomaly'] = label
        return


# ========================== Visualization Function =========================
def plot_tau_evolution():
    """
    Plot the evolution of tau during training.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(tau_values, label="Tau over time", alpha=0.8)
    plt.xlabel("Training Steps")
    plt.ylabel("Tau Value")
    plt.title("Evolution of Tau during Training")
    plt.legend()
    plot_path = os.path.join(current_dir, 'tau_evolution.png')
    plt.savefig(plot_path)  # Save the plot
    print(f"Tau evolution plot saved at {plot_path}")
    plt.close()  # Close the figure to free memory


# ========================== Training Function =============================
def train(num_LP, num_AL, discount_factor, learn_tau=True):
    """
    Train the RL agent with learned tau using DRO.

    Args:
        num_LP (int): Number of Label Propagation samples.
        num_AL (int): Number of Active Learning samples.
        discount_factor (float): Discount factor for RL.
        learn_tau (bool): Whether to learn tau dynamically using DRO.

    Returns:
        dict: Optimization metrics (e.g., F1-score).
    """
    # Load the dataset
    data_directory = os.path.join(current_dir, 'normal-data')
    print(f"Loading normal data from {data_directory}...")
    x_train_sequences = load_normal_data(data_directory, n_steps=n_steps)
    x_train = x_train_sequences[:, -1, :]  # Shape: (samples, 3)
    print(f"x_train shape: {x_train.shape}")

    # Train a Variational Autoencoder (VAE)
    print("Training VAE...")
    vae, _ = build_vae(original_dim, latent_dim, intermediate_dim)
    vae.fit(x_train, epochs=50, batch_size=32)
    vae_save_path = os.path.join(current_dir, 'vae_model.h5')
    vae.save(vae_save_path)
    print(f"VAE model saved at {vae_save_path}")

    # Load pretrained VAE
    vae = load_model(vae_save_path, custom_objects={'Sampling': Sampling}, compile=False)
    print("Loaded pretrained VAE model.")

    # Define experiment directories
    exp_relative_dir = ['RLVAL with DRO and Adaptive Scaling']
    dataset_dir = [os.path.join(current_dir, 'ydata-labeled-time-series-anomalies-v1_0', 'A1Benchmark')]

    for i in range(len(dataset_dir)):
        env = EnvTimeSeriesfromRepo(dataset_dir[i])
        env.statefnc = RNNBinaryStateFuc

        # Assign the appropriate reward function
        if learn_tau:
            env.rewardfnc = lambda timeseries, timeseries_curser, action: RNNBinaryRewardFuc(
                timeseries, timeseries_curser, action, vae
            )
        else:
            env.rewardfnc = RNNBinaryRewardFuc  # Without adaptive tau

        # Initialize environment parameters
        env.timeseries_curser_init = n_steps
        env.datasetfix = DATAFIXED
        env.datasetidx = 0

        # Testing environment
        env_test = env
        env_test.rewardfnc = RNNBinaryRewardFucTest

        # Define experiment directory
        experiment_dir = os.path.abspath(os.path.join(current_dir, "exp", exp_relative_dir[i]))
        print(f"Experiment directory: {experiment_dir}")

        # Reset TensorFlow graph and initialize global step
        tf.compat.v1.reset_default_graph()
        global_step = tf.compat.v1.train.get_or_create_global_step()

        # Initialize TensorFlow session
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(global_step.assign(0))  # Initialize global_step to 0
        print("TensorFlow session initialized.")

        # Initialize Q-estimator and Target estimator
        qlearn_estimator = Q_Estimator_Nonlinear(scope="qlearn", summaries_dir=experiment_dir, learning_rate=0.0003)
        target_estimator = Q_Estimator_Nonlinear(scope="target")
        print("Initialized Q-estimator and Target estimator.")

        with sess.as_default():
            # Start Q-Learning
            q_learning(env,
                       sess=sess,
                       qlearn_estimator=qlearn_estimator,
                       target_estimator=target_estimator,
                       num_episodes=EPISODES,
                       num_epoches=10,
                       experiment_dir=experiment_dir,
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
                       vae_model=vae
                       )
            # Validate the trained model
            print("Validating the trained model...")
            optimization_metric = q_learning_validator(env_test, qlearn_estimator,
                                                       int(env.datasetsize * (1 - validation_separate_ratio)),
                                                       experiment_dir)
        return optimization_metric


# ========================== Main Execution ================================
if __name__ == "__main__":
    try:
        # Run training with learned tau using Adaptive Preference Scaling
        print("Starting training with num_LP=100, num_AL=30, discount_factor=0.92...")
        train(100, 30, 0.92, learn_tau=True)

        print("Starting training with num_LP=150, num_AL=50, discount_factor=0.94...")
        train(150, 50, 0.94, learn_tau=True)

        print("Starting training with num_LP=200, num_AL=100, discount_factor=0.96...")
        train(200, 100, 0.96, learn_tau=True)

        # Plot the evolution of tau
        plot_tau_evolution()

    except Exception as e:
        print(f"An error occurred: {e}")
