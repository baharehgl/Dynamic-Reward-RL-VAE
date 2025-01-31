# asp-DRO.py

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
from environment.time_series_repo import EnvTimeSeriesfromRepo

# Specify CUDA devices for GPU acceleration (adjust as needed)
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

# ========================== Hyperparameters ==========================
# Training Parameters
DATAFIXED = 0  # Whether target is at a single time series dataset
EPISODES = 500  # Number of episodes for training
DISCOUNT_FACTOR = 0.99  # Reward discount factor [0,1]
EPSILON_START = 1.0  # Starting value of epsilon
EPSILON_END = 0.1  # Ending value of epsilon
EPSILON_DECAY_STEPS = 500000  # Steps over which epsilon decays

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
            if 'value' not in df.columns or 'anomaly' not in df.columns:
                print(f"Skipping {file}: Missing 'value' or 'anomaly' columns.")
                continue
            df = df[['value', 'anomaly']].dropna()
            if df.shape[0] < n_steps:
                print(f"Skipping {file} as it has less than {n_steps} rows after cleaning.")
                continue
            data_list.append(df)
            print(f"Loaded dataset {file} with {df.shape[0]} rows.")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not data_list:
        raise ValueError(
            f"No valid data found in {data_path}. Ensure CSV files have at least {n_steps} rows and contain 'value' and 'anomaly' columns.")

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
        current_state = np.array([current_time_step])  # Shape: (1, 1)

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
        true_label = timeseries['anomaly'][timeseries_curser]

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
               num_episodes=500,
               num_epoches=1000,
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
        print(f"Loading model checkpoint {latest_checkpoint}...")
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
            # Filter out any None states
            valid_states = [state for state in env.states_list if state is not None]
            data_train.extend(valid_states)
            print(f"Extended data_train with {len(valid_states)} states.")

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

    # Initialize Label Spreading model
    lp_model = LabelSpreading()
    print("Label Spreading model initialized.")

    # Populate replay memory
    while len(replay_memory) < replay_memory_init_size:
        env.reset()
        print(f"Environment reset during replay memory population. Current replay memory size: {len(replay_memory)}")
        states = env.states_list

        # Shuffle states to ensure diversity
        random.shuffle(states)

        for state in states:
            if len(replay_memory) >= replay_memory_init_size:
                break
            # Select action using policy
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
            action_probs = policy(state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            # Take a step in the environment
            next_state, reward, done, _ = env.step(action)

            # Add experience to replay memory
            replay_memory.append(Transition(state, reward, next_state, done))
            total_t += 1

            if done:
                env.reset()

    popu_time = time.time() - popu_time
    print("Populating replay memory completed in {:.2f} seconds".format(popu_time))
    print(f"Replay memory size after warm-up: {len(replay_memory)}")

    # ========================== Main Q-Learning Training Loop ==========================
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done and step < num_epoches:
            # Select action using epsilon-greedy policy
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
            action_probs = policy(state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            # Take action in the environment
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            total_t += 1

            # Add experience to replay memory
            replay_memory.append(Transition(state, reward, next_state, done))

            # Sample a minibatch from replay memory
            minibatch = random.sample(replay_memory, batch_size)
            states_mb, rewards_mb, next_states_mb, dones_mb = zip(*minibatch)

            # Prepare target Q-values
            target_q = qlearn_estimator.predict(states_mb, sess)
            next_q = target_estimator.predict(next_states_mb, sess) if not any(
                s is None for s in next_states_mb) else np.zeros_like(target_q)
            targets = np.copy(target_q)

            for i in range(len(minibatch)):
                if dones_mb[i]:
                    targets[i][action] = rewards_mb[i]
                else:
                    targets[i][action] = rewards_mb[i] + discount_factor * np.max(next_q[i])

            # Train the Q-network
            qlearn_estimator.update(states_mb, targets, sess)

            # Update target estimator periodically
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, qlearn_estimator, target_estimator)
                print(f"Updated target estimator at step {total_t}.")

            # Move to the next state
            state = next_state
            step += 1

        print(f"Episode {episode + 1}/{num_episodes} completed with total reward: {episode_reward}")

        # Optionally, save the model checkpoint
        if (episode + 1) % 10 == 0:
            saver.save(sess, checkpoint_path, global_step=episode + 1)
            print(f"Model checkpoint saved at episode {episode + 1}.")

    print("Training completed.")
