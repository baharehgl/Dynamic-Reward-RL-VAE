import matplotlib

matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import random
import sys
import os
import time
import zipfile
from scipy import stats
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.models import load_model
from mpl_toolkits.mplot3d import axes3d
from collections import deque, namedtuple

# Disable eager execution for TensorFlow 1.x compatibility
tf.compat.v1.disable_eager_execution()

# Add parent directory to sys.path for custom environment
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from environment.time_series_repo_ext import EnvTimeSeriesfromRepo  # Ensure this path is correct

# Set CUDA devices (modify as per your setup)
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

# ============================ Hyperparameters ============================
# Q-Learning Parameters
DATAFIXED = 0  # Whether target at a single time series dataset
EPISODES = 500  # Number of episodes for training
DISCOUNT_FACTOR = 0.5  # Reward discount factor [0,1]
EPSILON = 0.5  # Epsilon-greedy method parameter for action selection
EPSILON_DECAY = 1.00  # Epsilon-greedy method decay parameter

# Action Definitions
NOT_ANOMALY = 0
ANOMALY = 1
action_space = [NOT_ANOMALY, ANOMALY]
action_space_n = len(action_space)

# LSTM Parameters
n_steps = 25  # Size of the sliding window
n_input_dim = 2  # Dimension of the input for an LSTM cell
n_hidden_dim = 128  # Dimension of the hidden state in LSTM cell

# Reward Values
TP_Value = 5
TN_Value = 1
FP_Value = -1
FN_Value = -5

validation_separate_ratio = 0.9


# ============================ VAE Setup ============================

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae(original_dim, latent_dim=2, intermediate_dim=64):
    inputs = layers.Input(shape=(original_dim,))
    h = layers.Dense(intermediate_dim, activation='relu')(inputs)
    h = layers.Dense(intermediate_dim, activation='relu')(h)
    h = layers.Dense(intermediate_dim, activation='relu')(h)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    z = Sampling()([z_mean, z_log_var])

    decoder_h = layers.Dense(intermediate_dim, activation='relu')(z)
    decoder_h = layers.Dense(intermediate_dim, activation='relu')(decoder_h)
    decoder_h = layers.Dense(intermediate_dim, activation='relu')(decoder_h)
    decoder_mean = layers.Dense(original_dim, activation='sigmoid')(decoder_h)

    encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    vae = models.Model(inputs, decoder_mean, name='vae')

    # Define VAE loss
    reconstruction_loss = losses.binary_crossentropy(inputs, decoder_mean) * original_dim
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return vae, encoder


# ============================ Data Loading ============================

def unzip_file(zip_path, extract_to):
    """Unzip a zip file to the specified directory."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def load_normal_data_kpi(data_path):
    """
    Load and preprocess KPI training data from CSV.

    Args:
        data_path (str): Path to the KPI_train.csv file.

    Returns:
        np.ndarray: Scaled training data.
    """
    # Read the CSV file
    data = pd.read_csv(data_path)

    # Select relevant columns (adjust based on your CSV structure)
    # Example assumes 'metric1', 'metric2', 'metric3'
    # Modify as per your actual column names
    selected_columns = ['metric1', 'metric2', 'metric3']  # Replace with actual column names
    data = data[selected_columns]

    # Handle missing values if any
    data = data.fillna(method='ffill').fillna(method='bfill')

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values)

    return scaled_data


def load_test_data_kpi(data_path):
    """
    Load and preprocess KPI test data from HDF5.

    Args:
        data_path (str): Path to the KPI_ground_truth.hdf file.

    Returns:
        pd.DataFrame: Test data with ground truth labels.
    """
    with h5py.File(data_path, 'r') as hdf:
        # Explore the structure of the HDF5 file
        # Adjust keys based on actual structure
        # Example assumes 'data' and 'labels' datasets
        data = hdf['data'][:]  # Replace 'data' with actual dataset name
        labels = hdf['labels'][:]  # Replace 'labels' with actual dataset name

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['metric1', 'metric2', 'metric3'])  # Adjust as needed
    df['anomaly'] = labels  # Add anomaly labels

    return df


# ============================ Environment Setup ============================

def create_environment_kpi(train_data, test_data):
    """
    Create and configure the environment for KPI data.

    Args:
        train_data (np.ndarray): Scaled training data for VAE.
        test_data (pd.DataFrame): Test data with ground truth labels.

    Returns:
        EnvTimeSeriesfromRepo: Configured environment.
    """
    env = EnvTimeSeriesfromRepo()

    # Assuming EnvTimeSeriesfromRepo has methods to set training and test data
    env.set_train_data(train_data)
    env.set_test_data(test_data)

    # Configure state and reward functions (to be set outside based on training/testing)
    env.statefnc = RNNBinaryStateFuc
    env.rewardfnc = RNNBinaryRewardFuc  # Set later based on mode

    env.timeseries_curser_init = n_steps
    env.datasetfix = DATAFIXED
    env.datasetidx = 0

    return env


# ============================ Reward Functions ============================

def kl_divergence(p, q):
    """Compute the KL divergence KL(p || q)."""
    p = np.clip(p, 1e-10, 1)  # Avoid log(0)
    q = np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))


def adaptive_scaling_factor_dro(preference_strength, tau_min=0.1, tau_max=5.0, rho=0.5, max_iter=10):
    """
    Learn tau dynamically using DRO (KL-constrained optimization).

    Args:
    - preference_strength: Probability of preference (sigmoid of reconstruction error).
    - tau_min, tau_max: Bounds for tau.
    - rho: Regularization term for DRO.
    - max_iter: Iterations for optimizing tau.

    Returns:
    - Optimized tau.
    """
    tau = 1.0  # Initialize tau

    for _ in range(max_iter):
        # Compute KL divergence constraint
        kl_term = kl_divergence([preference_strength, 1 - preference_strength], [0.5, 0.5])

        # Gradient of the DRO loss w.r.t. tau
        grad = -np.log(1 + np.exp(-preference_strength / tau)) + rho - kl_term

        # Hessian (second derivative)
        hess = (preference_strength ** 2 * np.exp(-preference_strength / tau)) / (
                tau ** 3 * (1 + np.exp(-preference_strength / tau)) ** 2
        )

        # Newton's update step for tau
        tau = tau - grad / (hess + 1e-8)

        # Project tau to be within valid bounds
        tau = np.clip(tau, tau_min, tau_max)

    return tau


tau_values = []


def RNNBinaryRewardFuc(timeseries, timeseries_curser, action=0, vae=None, scale_factor=10):
    """
    Reward function incorporating VAE reconstruction error and adaptive scaling via DRO.

    Args:
        timeseries (pd.DataFrame): The timeseries data.
        timeseries_curser (int): Current cursor position in the timeseries.
        action (int): Action taken.
        vae (tf.keras.Model): Trained VAE model.
        scale_factor (float): Scaling factor for the VAE penalty.

    Returns:
        list: Rewards for each possible action.
    """
    if timeseries_curser >= n_steps:
        # Extract the current window for VAE
        current_state = np.array([timeseries['value'][timeseries_curser - n_steps:timeseries_curser]])
        vae_reconstruction = vae.predict(current_state)
        reconstruction_error = np.mean(np.square(vae_reconstruction - current_state))

        vae_penalty = -scale_factor * reconstruction_error

        # Compute preference strength using sigmoid
        preference_strength = np.clip(1 / (1 + np.exp(-reconstruction_error)), 0.05, 0.95)

        # Adaptively scale rewards using DRO
        tau = adaptive_scaling_factor_dro(preference_strength)
        tau_values.append(tau)  # Store tau for visualization

        # Assign rewards based on true label and action
        if timeseries['label'][timeseries_curser] == 0:
            return [tau * (TN_Value + vae_penalty), tau * (FP_Value + vae_penalty)]
        if timeseries['label'][timeseries_curser] == 1:
            return [tau * (FN_Value + vae_penalty), tau * (TP_Value + vae_penalty)]
    else:
        return [0, 0]


def RNNBinaryRewardFucTest(timeseries, timeseries_curser, action=0):
    """
    Reward function for testing (without VAE and adaptive scaling).

    Args:
        timeseries (pd.DataFrame): The timeseries data.
        timeseries_curser (int): Current cursor position in the timeseries.
        action (int): Action taken.

    Returns:
        list: Rewards for each possible action.
    """
    if timeseries_curser >= n_steps:
        if timeseries['anomaly'][timeseries_curser] == 0:
            return [TN_Value, FP_Value]
        if timeseries['anomaly'][timeseries_curser] == 1:
            return [FN_Value, TP_Value]
    else:
        return [0, 0]


# ============================ State Function ============================

def RNNBinaryStateFuc(timeseries, timeseries_curser, previous_state=[], action=None):
    """
    State function that returns the current state as a sliding window.

    Args:
        timeseries (pd.DataFrame): The timeseries data.
        timeseries_curser (int): Current cursor position in the timeseries.
        previous_state (list): Previous state.
        action (int): Action taken.

    Returns:
        np.ndarray: Current state(s).
    """
    if timeseries_curser == n_steps:
        state = []
        for i in range(timeseries_curser):
            state.append([timeseries['value'][i], 0])

        state.pop(0)
        state.append([timeseries['value'][timeseries_curser], 1])

        return np.array(state, dtype='float32')

    if timeseries_curser > n_steps:
        state0 = np.concatenate((previous_state[1:n_steps],
                                 [[timeseries['value'][timeseries_curser], 0]]))
        state1 = np.concatenate((previous_state[1:n_steps],
                                 [[timeseries['value'][timeseries_curser], 1]]))

        return np.array([state0, state1], dtype='float32')


# ============================ Q-Learning Components ============================

class Q_Estimator_Nonlinear():
    """
    Action-Value Function Approximator Q(s,a) with Tensorflow RNN.
    Note: The Recurrent Neural Network is used here!
    """

    def __init__(self, learning_rate=np.float32(0.01), scope="Q_Estimator_Nonlinear", summaries_dir=None):
        self.scope = scope
        self.summary_writer = None

        with tf.compat.v1.variable_scope(scope):
            # Graph input
            self.state = tf.compat.v1.placeholder(shape=[None, n_steps, n_input_dim],
                                                  dtype=tf.float32, name="state")
            self.target = tf.compat.v1.placeholder(shape=[None, action_space_n],
                                                   dtype=tf.float32, name="target")

            # Define weights and biases
            self.weights = {
                'out': tf.Variable(tf.compat.v1.random_normal([n_hidden_dim, action_space_n]))
            }
            self.biases = {
                'out': tf.Variable(tf.compat.v1.random_normal([action_space_n]))
            }

            self.state_unstack = tf.unstack(self.state, n_steps, 1)

            # Define an LSTM cell with TensorFlow
            lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(n_hidden_dim, forget_bias=1.0)

            # Get LSTM cell output
            self.outputs, self.states = tf.compat.v1.nn.static_rnn(lstm_cell,
                                                                   self.state_unstack,
                                                                   dtype=tf.float32)

            # Linear activation, using RNN inner loop last output
            self.action_values = tf.matmul(self.outputs[-1], self.weights['out']) + self.biases['out']

            # Loss and train op
            self.losses = tf.compat.v1.squared_difference(self.action_values, self.target)
            self.loss = tf.reduce_mean(self.losses)

            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

            # Define global_step variable if not already defined
            if not tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="global_step"):
                self.global_step = tf.Variable(0, name="global_step", trainable=False)
            else:
                self.global_step = \
                tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="global_step")[0]

            self.train_op = self.optimizer.minimize(self.loss,
                                                    global_step=self.global_step)

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
                self.summary_writer = tf.compat.v1.summary.FileWriter(summary_dir)

    def predict(self, state, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        return sess.run(self.action_values, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        summaries, global_step_val, _ = sess.run([self.summaries,
                                                  self.global_step,
                                                  self.train_op], feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step_val)
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


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns Q-values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """

    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype='float32') * epsilon / nA
        q_values = estimator.predict(state=[observation])
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


# ============================ Q-Learning Algorithm ============================

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
        qlearn_estimator: Q-Learning estimator.
        target_estimator: Target network estimator.
        num_episodes (int): Number of episodes to run for.
        num_epoches (int): Number of epochs per episode.
        replay_memory_size (int): Maximum size of replay memory.
        replay_memory_init_size (int): Initial size of replay memory.
        experiment_dir (str): Directory for experiment logs.
        update_target_estimator_every (int): Steps after which to update target network.
        discount_factor (float): Discount factor for RL.
        epsilon_start (float): Starting value of epsilon for epsilon-greedy.
        epsilon_end (float): Final value of epsilon after decay.
        epsilon_decay_steps (int): Number of steps over which epsilon decays.
        batch_size (int): Batch size for training.
        num_LabelPropagation (int): Number of samples for label propagation.
        num_active_learning (int): Number of samples for active learning.
        test (int): Whether to run in test mode.
        vae_model (tf.keras.Model): Trained VAE model.

    Returns:
        None
    """
    # Define a named tuple for storing experiences
    Transition = namedtuple("Transition", ["state", "reward", "next_state", "done"])

    # Initialize replay memory
    replay_memory = []

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Saver for saving checkpoints
    saver = tf.compat.v1.train.Saver()

    # Load a previous checkpoint if available
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
        if test:
            return

    # Get the current global step
    total_t = sess.run(qlearn_estimator.global_step)

    # Epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # Define the policy
    policy = make_epsilon_greedy_policy(qlearn_estimator, env.action_space_n)

    num_label = 0

    # 1. Populate the replay memory with initial experience using Warm-Up
    popu_time = time.time()

    # Warm up with active learning
    print('Warm up starting...')

    outliers_fraction = 0.01
    data_train = []
    for num in range(env.datasetsize):
        env.reset()
        data_train.extend(env.states_list)

    # Warm-Up using Isolation Forest
    model = WarmUp().warm_up_isolation_forest(outliers_fraction, np.array(data_train))

    # Label Propagation model
    lp_model = LabelSpreading()

    for t in itertools.count():
        env.reset()
        data = np.array(env.states_list).transpose(2, 0, 1).reshape(2, -1)[0].reshape(-1, n_steps)[:, -1].reshape(-1, 1)
        anomaly_score = model.decision_function(data)  # Score range depends on the model
        pred_score = [-1 * s + 0.5 for s in anomaly_score]  # Adjust as needed
        warm_samples = np.argsort(pred_score)[:5]
        warm_samples = np.append(warm_samples, np.argsort(pred_score)[-5:])

        # Initialize labels
        label_list = [-1] * len(env.states_list)

        for sample in warm_samples:
            # Select a state from warm-up samples
            state = env.states_list[sample]
            # Update the cursor
            env.timeseries_curser = sample + n_steps
            action_probs = policy(state, epsilons[min(total_t, epsilon_decay_steps - 1)])
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            # Mark the sample as labeled
            env.timeseries.at[env.timeseries_curser, 'label'] = env.timeseries.at[env.timeseries_curser, 'anomaly']
            num_label += 1

            # Assign label for propagation
            label_list[sample] = int(env.timeseries.at[env.timeseries_curser, 'anomaly'])

            next_state, reward, done, _ = env.step(action)

            replay_memory.append(Transition(state, reward, next_state, done))

        # Label Propagation main process
        unlabeled_indices = [i for i, e in enumerate(label_list) if e == -1]
        label_list = np.array(label_list)
        lp_model.fit(np.array(env.states_list), label_list)
        pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)

        # Select most certain samples
        certainty_index = np.argsort(pred_entropies)
        certainty_index = certainty_index[np.isin(certainty_index, unlabeled_indices)][:num_LabelPropagation]

        # Assign pseudo labels
        for index in certainty_index:
            pseudo_label = lp_model.transduction_[index]
            env.timeseries.at[index + n_steps, 'label'] = pseudo_label

        if len(replay_memory) >= replay_memory_init_size:
            break

    popu_time = time.time() - popu_time
    print("Populating replay memory completed in {:.2f} seconds.".format(popu_time))

    # 2. Start the main Q-Learning loop
    for i_episode in range(num_episodes):
        # Save the current checkpoint periodically
        if i_episode % 50 == 49:
            print("Saving checkpoint at episode {}/{}".format(i_episode + 1, num_episodes))
            saver.save(sess, checkpoint_path)

        per_loop_time1 = time.time()

        # Reset the environment
        state = env.reset()
        while env.datasetidx > env.datasetrng * validation_separate_ratio:
            env.reset()
            print('Double reset due to dataset index exceeding validation ratio.')

        # Active Learning:
        # Identify already labeled samples
        labeled_index = [i for i, e in enumerate(env.timeseries['label']) if e != -1]
        labeled_index = [item for item in labeled_index if item >= n_steps]
        labeled_index = [item - n_steps for item in labeled_index]

        # Initialize active learning
        al = active_learning(env=env, N=num_active_learning, strategy='margin_sampling',
                             estimator=qlearn_estimator, already_selected=labeled_index)
        # Get samples to label
        al_samples = al.get_samples()
        print('Labeling samples: {} in env {}'.format(al_samples, env.datasetidx))

        # Label the selected samples
        for sample in al_samples:
            # Assign true labels
            env.timeseries.at[sample + n_steps, 'label'] = env.timeseries.at[sample + n_steps, 'anomaly']
            num_label += 1

            # Take a step and add to replay memory
            next_state, reward, done, _ = env.step(action=0)  # Action is arbitrary here
            replay_memory.append(Transition(state, reward, next_state, done))

        # Add labeled samples to replay memory
        for sample in al_samples:
            if len(replay_memory) >= replay_memory_size:
                replay_memory.pop(0)
            replay_memory.append(Transition(state, reward, next_state, done))

        # Label Propagation main process:
        state_list = np.array(env.states_list)
        label_list = np.array(env.timeseries['label'][n_steps:])
        unlabeled_indices = [i for i, e in enumerate(label_list) if e == -1]

        if len(unlabeled_indices) > 0:
            lp_model.fit(state_list, label_list)
            pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)

            # Select most certain samples
            certainty_index = np.argsort(pred_entropies)
            certainty_index = certainty_index[np.isin(certainty_index, unlabeled_indices)][:num_LabelPropagation]

            # Assign pseudo labels
            for index in certainty_index:
                pseudo_label = lp_model.transduction_[index]
                env.timeseries.at[index + n_steps, 'label'] = pseudo_label

        per_loop_time2 = time.time()

        # Update the model
        for i_epoch in range(num_epoches):
            # Update target network periodically
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, qlearn_estimator, target_estimator)
                print("\nCopied model parameters to target network.\n")

            # Sample a minibatch from replay memory
            samples = random.sample(replay_memory, batch_size)
            states_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            if discount_factor > 0:
                # Calculate Q values and targets
                next_states_batch = np.squeeze(np.split(next_states_batch, 2, axis=1))
                next_states_batch0 = next_states_batch[0]
                next_states_batch1 = next_states_batch[1]

                q_values_next0 = target_estimator.predict(state=next_states_batch0, sess=sess)
                q_values_next1 = target_estimator.predict(state=next_states_batch1, sess=sess)

                targets_batch = reward_batch + (discount_factor *
                                                np.stack((np.amax(q_values_next0, axis=1),
                                                          np.amax(q_values_next1, axis=1)),
                                                         axis=-1))
            else:
                targets_batch = reward_batch

            # Perform gradient descent update
            qlearn_estimator.update(state=states_batch, target=targets_batch.astype(np.float32), sess=sess)

            total_t += 1

        per_loop_time_popu = per_loop_time2 - per_loop_time1
        per_loop_time_updt = time.time() - per_loop_time2
        print("Global step {} @ Episode {}/{}, time: {:.2f} + {:.2f}".format(
            total_t, i_episode + 1, num_episodes, per_loop_time_popu, per_loop_time_updt))

    return


# ============================ Evaluation Metrics ============================

def evaluate_model(y_true, y_pred):
    """
    Compute various classification metrics.

    Args:
    - y_true (list): Ground truth labels (0 = Normal, 1 = Anomaly)
    - y_pred (list): Predicted labels (0 = Normal, 1 = Anomaly)

    Returns:
    - dict: Dictionary with evaluation metrics.
    """
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Compute existing metrics
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)

    # Compute new metrics
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
        env (EnvTimeSeriesfromRepo): Environment.
        estimator (Q_Estimator_Nonlinear): Trained model.
        num_episodes (int): Number of validation episodes.
        record_dir (str): Directory to save performance records.
        plot (int): Whether to plot results.

    Returns:
        dict: Averaged evaluation metrics.
    """
    y_true_all = []
    y_pred_all = []

    policy = make_epsilon_greedy_policy(estimator, env.action_space_n)

    for i_episode in range(num_episodes):
        print("Validation Episode {}/{}".format(i_episode + 1, num_episodes))
        state = env.reset()
        while env.datasetidx < env.datasetrng * validation_separate_ratio:
            env.reset()
            print('Double reset during validation.')

        for t in itertools.count():
            action_probs = policy(state, 0)  # Greedy policy
            action = np.argmax(action_probs)

            y_true_all.append(env.timeseries.at[env.timeseries_curser, 'anomaly'])  # True label
            y_pred_all.append(action)  # Predicted label

            next_state, reward, done, _ = env.step(action)
            if done:
                break
            state = next_state[action]

    # Compute evaluation metrics
    results = evaluate_model(y_true_all, y_pred_all)

    # Save metrics to a text file if record_dir is provided
    if record_dir:
        with open(os.path.join(record_dir, 'performance.txt'), 'w') as rec_file:
            for metric, value in results.items():
                rec_file.write(f"{metric}: {value:.4f}\n")

    # Plot the metrics and save as PNG
    if plot:
        metrics = list(results.keys())
        values = list(results.values())
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics, values, color='skyblue')
        plt.xlabel('Metrics')
        plt.ylabel('Scores')
        plt.title('Validation Metrics')
        plt.ylim(0, 1)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(record_dir, 'validation_metrics.png'))
        plt.close()
        print("Validation metrics plot saved as 'validation_metrics.png'.")

    # Print metrics
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    return results


# ============================ Active Learning and Warm-Up ============================

class active_learning(object):
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
            distance = abs(q[0] - q[1])
            distances.append(distance)
        distances = np.array(distances)
        if len(distances.shape) < 2:
            min_margin = abs(distances)
        else:
            sort_distances = np.sort(distances, axis=1)[:, -2:]
            min_margin = sort_distances[:, 1] - sort_distances[:, 0]
        rank_ind = np.argsort(min_margin)
        rank_ind = [i for i in rank_ind if i not in self.already_selected]
        active_samples = rank_ind[:self.N]
        return active_samples

    def get_samples_by_score(self, threshold):
        states_list = self.env.states_list
        distances = []
        for state in states_list:
            q = self.estimator.predict(state=[state])[0]
            distance = abs(q[0] - q[1])
            distances.append(distance)
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
        for sample in active_samples:
            print('Active Learning found a sample:')
            print(self.env.timeseries['value'].iloc[sample:sample + n_steps])
            print('Please label the last timestamp based on your knowledge:')
            print('0 for non-anomaly; 1 for anomaly')
            # For automated scripts, replace input() with predefined labels or skip
            label = int(input())  # Replace with automatic labeling if needed
            self.env.timeseries.at[sample + n_steps - 1, 'anomaly'] = label
        return


class WarmUp(object):
    def warm_up_SVM(self, outliers_fraction, data):
        """
        Warm-up using One-Class SVM.

        Args:
            outliers_fraction (float): Fraction of outliers.
            data (np.ndarray): Training data.

        Returns:
            OneClassSVM: Trained One-Class SVM model.
        """
        model = OneClassSVM(gamma='auto', nu=0.95 * outliers_fraction)
        model.fit(data)
        return model

    def warm_up_isolation_forest(self, outliers_fraction, X_train):
        """
        Warm-up using Isolation Forest.

        Args:
            outliers_fraction (float): Fraction of outliers.
            X_train (np.ndarray): Training data.

        Returns:
            IsolationForest: Trained Isolation Forest model.
        """
        from sklearn.ensemble import IsolationForest
        data = np.array(X_train).reshape(-1, X_train.shape[-1])
        clf = IsolationForest(contamination=outliers_fraction)
        clf.fit(data)
        return clf


# ============================ Training Function ============================

def train(num_LP, num_AL, discount_factor, learn_tau=True):
    """
    Train the RL agent with learned tau using DRO on KPI dataset.

    Args:
        num_LP (int): Number of Label Propagation samples.
        num_AL (int): Number of Active Learning samples.
        discount_factor (float): Discount factor for RL.
        learn_tau (bool): Whether to learn tau dynamically using DRO.

    Returns:
        float: Optimization metric (e.g., F1-score).
    """
    # Paths to your KPI data
    #kpi_train_zip = r'path_to_your_data/KPI_train.csv.zip'
    #kpi_test_zip = r'path_to_your_data/KPI_ground_truth.hdf.zip'
    kpi_train_zip = r'KPI_train.csv.zip'
    kpi_test_zip = r'KPI_ground_truth.hdf.zip'

    # Extract the zip files
    unzip_file(kpi_train_zip, 'KPI_data/train')
    unzip_file(kpi_test_zip, 'KPI_data/test')

    # Paths to the extracted files
    kpi_train_csv = os.path.join('KPI_data', 'train', 'KPI_train.csv')  # Adjust if different
    kpi_test_hdf = os.path.join('KPI_data', 'test', 'KPI_ground_truth.hdf')  # Adjust if different

    # Load and preprocess training data
    x_train = load_normal_data_kpi(kpi_train_csv)

    # Build and train the VAE
    original_dim = x_train.shape[1]  # Number of features, e.g., 3
    latent_dim = 10
    intermediate_dim = 64

    vae, encoder = build_vae(original_dim, latent_dim, intermediate_dim)
    vae.fit(x_train, x_train, epochs=50, batch_size=32, validation_split=0.1)
    vae.save('vae_model_kpi.h5')

    # Load the trained VAE
    vae = load_model('vae_model_kpi.h5', custom_objects={'Sampling': Sampling}, compile=False)

    # Load and preprocess test data
    df_test = load_test_data_kpi(kpi_test_hdf)

    # Create the environment with KPI data
    env = create_environment_kpi(x_train, df_test)

    # Configure reward function based on whether to learn tau
    if learn_tau:
        env.rewardfnc = lambda timeseries, timeseries_curser, action: RNNBinaryRewardFuc(
            timeseries, timeseries_curser, action, vae
        )
    else:
        env.rewardfnc = RNNBinaryRewardFuc  # Without adaptive tau

    # Create the testing environment
    env_test = create_environment_kpi(x_train, df_test)
    env_test.rewardfnc = RNNBinaryRewardFucTest

    # Define experiment directories
    exp_relative_dir = ['RLVAL with DRO and Adaptive Scaling']
    experiment_dir = os.path.abspath("./exp/{}".format(exp_relative_dir[0]))

    # Reset TensorFlow graph
    tf.compat.v1.reset_default_graph()
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Initialize Q-Learning estimators
    qlearn_estimator = Q_Estimator_Nonlinear(scope="qlearn", summaries_dir=experiment_dir, learning_rate=0.0003)
    target_estimator = Q_Estimator_Nonlinear(scope="target")

    # Start TensorFlow session
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    with sess.as_default():
        # Populate replay memory and train
        q_learning(
            env=env,
            sess=sess,
            qlearn_estimator=qlearn_estimator,
            target_estimator=target_estimator,
            num_episodes=300,
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
        optimization_metric = q_learning_validator(
            env=env_test,
            estimator=qlearn_estimator,
            num_episodes=int(env.datasetsize * (1 - validation_separate_ratio)),
            record_dir=experiment_dir
        )

    return optimization_metric


# ============================ Plotting Function ============================

def plot_tau_evolution(experiment_dir):
    """
    Plot the evolution of tau during training and save as PNG.

    Args:
        experiment_dir (str): Directory where tau_values are stored.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    plt.plot(tau_values, label="Tau over time", alpha=0.8)
    plt.xlabel("Training Steps")
    plt.ylabel("Tau Value")
    plt.title("Evolution of Tau during Training")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(experiment_dir, 'tau_evolution.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Tau evolution plot saved as '{plot_path}'.")


# ============================ Main Execution ============================

if __name__ == "__main__":
    # Example training runs with different parameters
    # Update 'path_to_your_data' with the actual path to your data files

    # First Training Run
    metric1 = train(num_LP=100, num_AL=30, discount_factor=0.92, learn_tau=True)
    # Assuming experiment_dir is consistent, otherwise modify to capture inside train()
    plot_tau_evolution(experiment_dir='./exp/RLVAL with DRO and Adaptive Scaling')

    # Second Training Run
    metric2 = train(num_LP=150, num_AL=50, discount_factor=0.94, learn_tau=True)
    plot_tau_evolution(experiment_dir='./exp/RLVAL with DRO and Adaptive Scaling')

    # Third Training Run
    metric3 = train(num_LP=200, num_AL=100, discount_factor=0.96, learn_tau=True)
    plot_tau_evolution(experiment_dir='./exp/RLVAL with DRO and Adaptive Scaling')

    # Print F1 Scores from different training runs
    print(f"F1 Scores from different training runs: {metric1}, {metric2}, {metric3}")

    # Optionally, save F1 scores to a file or handle them as needed
