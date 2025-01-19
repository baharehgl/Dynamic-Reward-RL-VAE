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
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.models import load_model
from mpl_toolkits.mplot3d import axes3d
from collections import deque, namedtuple
import glob

# --------------------------- Custom Environment Import ---------------------------

# Import the custom environment
# Ensure that the path to the custom environment is correct
# Modify the path as necessary
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
try:
    from environment.time_series_repo_ext import EnvTimeSeriesfromRepo  # Ensure this path is correct
except ImportError as e:
    print("Error importing EnvTimeSeriesfromRepo:", e)
    sys.exit(1)

# --------------------------- CUDA Configuration ---------------------------

# Set CUDA devices if using GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

# --------------------------- Hyperparameters ---------------------------

DATAFIXED = 0  # Whether target at a single time series dataset

EPISODES = 500  # Number of episodes for training
DISCOUNT_FACTOR = 0.5  # Reward discount factor [0,1]
EPSILON = 0.5  # Epsilon-greedy method parameter for action selection
EPSILON_DECAY = 1.00  # Epsilon-greedy method decay parameter

NOT_ANOMALY = 0
ANOMALY = 1

action_space = [NOT_ANOMALY, ANOMALY]
action_space_n = len(action_space)

n_steps = 25  # Size of the sliding window for SLIDE_WINDOW state and reward functions
n_input_dim = 2  # Dimension of the input for an LSTM cell
n_hidden_dim = 128  # Dimension of the hidden state in LSTM cell

# Reward Values
TP_Value = 5
TN_Value = 1
FP_Value = -1
FN_Value = -5

validation_separate_ratio = 0.9

# --------------------------- Data Extraction ---------------------------

def unzip_file(zip_path, extract_to):
    """
    Unzip a zip file to a specified directory.

    Args:
        zip_path (str): Path to the zip file.
        extract_to (str): Directory to extract the contents to.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

# Relative paths to your zip files (assuming they are in the same directory as the script)
kpi_train_zip = os.path.join(current_dir, "KPI_train.csv.zip")
kpi_test_zip = os.path.join(current_dir, "KPI_ground_truth.hdf.zip")

# Extraction directories
train_extract_dir = os.path.join(current_dir, 'KPI_data', 'train')
test_extract_dir = os.path.join(current_dir, 'KPI_data', 'test')

# Create directories if they don't exist
os.makedirs(train_extract_dir, exist_ok=True)
os.makedirs(test_extract_dir, exist_ok=True)

# Extract the zip files
unzip_file(kpi_train_zip, train_extract_dir)
unzip_file(kpi_test_zip, test_extract_dir)

# --------------------------- Data Loading ---------------------------

def find_file(directory, pattern):
    """
    Find a file in a directory matching the given pattern.

    Args:
        directory (str): Directory to search in.
        pattern (str): Pattern to match file names.

    Returns:
        str: Path to the first matching file.

    Raises:
        FileNotFoundError: If no matching file is found.
    """
    search_path = os.path.join(directory, pattern)
    files = glob.glob(search_path)
    if not files:
        raise FileNotFoundError(f"No file matching pattern '{pattern}' found in directory '{directory}'")
    return files[0]

def list_hdf_keys(hdf_path):
    """
    List all available keys in the HDF5 file.

    Args:
        hdf_path (str): Path to the HDF5 file.
    """
    with pd.HDFStore(hdf_path, 'r') as store:
        print("\nAvailable keys in the HDF5 file:")
        print(store.keys())

def load_normal_data_kpi(data_path, exclude_columns=None):
    """
    Load and preprocess KPI training data from CSV.

    Args:
        data_path (str): Path to the KPI_train.csv file.
        exclude_columns (list, optional): Columns to exclude from scaling.

    Returns:
        np.ndarray: Scaled training data.
    """
    # Read the CSV file
    data = pd.read_csv(data_path)

    # Display columns and their data types for debugging
    print("\nColumns in the CSV and their data types:")
    print(data.dtypes)

    # Exclude specified columns if provided
    if exclude_columns:
        metric_columns = [col for col in data.columns if col not in exclude_columns]
    else:
        # Automatically select only numeric columns
        metric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

    print(f"\nSelected metric columns for scaling: {metric_columns}")

    # Ensure that the selected columns are indeed numeric
    data = data[metric_columns].apply(pd.to_numeric, errors='coerce')

    # Handle missing values if any
    data = data.fillna(method='ffill').fillna(method='bfill')

    # Verify that all data is numeric
    if data.isnull().values.any():
        missing_cols = data.columns[data.isnull().any()].tolist()
        raise ValueError(f"Data contains non-numeric values in columns: {missing_cols}. Please check the CSV file.")

    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values)

    return scaled_data

def load_test_data_kpi_pandas(data_path, key='data'):
    """
    Load and preprocess KPI test data from HDF5 using Pandas.

    Args:
        data_path (str): Path to the KPI_ground_truth.hdf file.
        key (str): The key of the dataset to read.

    Returns:
        pd.DataFrame: Test data with ground truth labels.
    """
    try:
        df = pd.read_hdf(data_path, key=key)
        print(f"\nSuccessfully loaded data from key '{key}'.")
    except Exception as e:
        print(f"\nError reading HDF5 file with Pandas: {e}")
        sys.exit(1)

    # Verify the DataFrame
    print("\nLoaded DataFrame:")
    print(df.head())

    # Check if 'anomaly' column exists
    if 'anomaly' not in df.columns:
        print("Error: 'anomaly' column not found in the test data.")
        sys.exit(1)

    # Create a 'value' column as the mean of all metrics (modify as per your data)
    metric_columns = [col for col in df.columns if col not in ['timestamp', 'label', 'KPI ID', 'anomaly']]
    if not metric_columns:
        print("Error: No metric columns found for creating the 'value' column.")
        sys.exit(1)

    df['value'] = df[metric_columns].mean(axis=1)
    print("Created 'value' column as the mean of metric columns.")

    return df

def load_test_data_kpi(data_path):
    """
    Load and preprocess KPI test data from HDF5 using Pandas.

    Args:
        data_path (str): Path to the KPI_ground_truth.hdf file.

    Returns:
        pd.DataFrame: Test data with ground truth labels.
    """
    return load_test_data_kpi_pandas(data_path, key='data')  # Ensure 'data' is the correct key

# Dynamically locate the CSV and HDF5 files after extraction
try:
    kpi_train_csv = find_file(train_extract_dir, '*.csv')
    kpi_test_hdf = find_file(test_extract_dir, '*.hdf')
except FileNotFoundError as e:
    print(e)
    sys.exit(1)

# Optional: List available keys in the HDF5 file to verify the correct key
list_hdf_keys(kpi_test_hdf)

# Load and scale the training data
# Exclude 'timestamp', 'label', and 'KPI ID' columns
x_train = load_normal_data_kpi(kpi_train_csv, exclude_columns=['timestamp', 'label', 'KPI ID'])
print(f"\nLoaded and scaled training data from {kpi_train_csv}")

# Load the test data
df_test = load_test_data_kpi(kpi_test_hdf)
print(f"\nLoaded test data from {kpi_test_hdf}")

# --------------------------- VAE Components ---------------------------

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(original_dim, latent_dim=2, intermediate_dim=64):
    """
    Build a Variational Autoencoder (VAE) model.

    Args:
        original_dim (int): Dimensionality of the input data.
        latent_dim (int): Dimensionality of the latent space.
        intermediate_dim (int): Dimensionality of the intermediate dense layers.

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
    encoder = models.Model(inputs, [z_mean, z_log_var, z], name="encoder")
    vae = models.Model(inputs, decoder_mean, name="vae")

    # Loss
    reconstruction_loss = losses.binary_crossentropy(inputs, decoder_mean) * original_dim
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    return vae, encoder

# --------------------------- Reward Function ---------------------------

def kl_divergence(p, q):
    """Compute the KL divergence KL(p || q)."""
    p = np.clip(p, 1e-10, 1)  # Avoid log(0)
    q = np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))

def adaptive_scaling_factor_dro(preference_strength, tau_min=0.1, tau_max=5.0, rho=0.5, max_iter=10):
    """
    Learn tau dynamically using DRO (KL-constrained optimization).

    Args:
        preference_strength (float): Probability of preference (sigmoid of reconstruction error).
        tau_min (float): Minimum value for tau.
        tau_max (float): Maximum value for tau.
        rho (float): Regularization term for DRO.
        max_iter (int): Maximum iterations for optimization.

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
    Reward function for the RNN-based binary action space.

    Args:
        timeseries (pd.DataFrame): The time series data.
        timeseries_curser (int): Current cursor position in the time series.
        action (int): Action taken (0 for NOT_ANOMALY, 1 for ANOMALY).
        vae (keras.Model): Trained VAE model.
        scale_factor (int): Scaling factor for the VAE penalty.

    Returns:
        list: Rewards for each possible action.
    """
    if timeseries_curser >= n_steps:
        # Extract the current window of data
        current_state = np.array([timeseries['value'][timeseries_curser - n_steps:timeseries_curser]])
        vae_reconstruction = vae.predict(current_state)
        reconstruction_error = np.mean(np.square(vae_reconstruction - current_state))

        vae_penalty = -scale_factor * reconstruction_error

        # Calculate preference strength using sigmoid of reconstruction error
        preference_strength = np.clip(1 / (1 + np.exp(-reconstruction_error)), 0.05, 0.95)

        # Dynamically adjust tau using DRO
        tau = adaptive_scaling_factor_dro(preference_strength)
        tau_values.append(tau)  # Store tau for visualization

        if timeseries['anomaly'][timeseries_curser] == 0:
            return [tau * (TN_Value + vae_penalty), tau * (FP_Value + vae_penalty)]
        if timeseries['anomaly'][timeseries_curser] == 1:
            return [tau * (FN_Value + vae_penalty), tau * (TP_Value + vae_penalty)]
    else:
        return [0, 0]

def RNNBinaryRewardFucTest(timeseries, timeseries_curser, action=0):
    """
    Reward function for testing.

    Args:
        timeseries (pd.DataFrame): The time series data.
        timeseries_curser (int): Current cursor position in the time series.
        action (int): Action taken (0 for NOT_ANOMALY, 1 for ANOMALY).

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

# --------------------------- Q-Learning Components ---------------------------

class Q_Estimator_Nonlinear():
    """
    Action-Value Function Approximator Q(s,a) with TensorFlow RNN.
    Note: The Recurrent Neural Network is used here !
    """

    def __init__(self, learning_rate=np.float32(0.01), scope="Q_Estimator_Nonlinear", summaries_dir=None):
        self.scope = scope
        self.summary_writer = None

        with tf.compat.v1.variable_scope(scope):
            # tf Graph input
            self.state = tf.compat.v1.placeholder(shape=[None, n_steps, n_input_dim],
                                                 dtype=tf.float32, name="state")
            self.target = tf.compat.v1.placeholder(shape=[None, action_space_n],
                                                  dtype=tf.float32, name="target")

            # Define weights
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

            # Define a global step variable if not already defined
            if not tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_STEP):
                self.global_step = tf.Variable(0, name="global_step", trainable=False)
            else:
                self.global_step = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_STEP)[0]

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
                os.makedirs(summary_dir, exist_ok=True)
                self.summary_writer = tf.compat.v1.summary.FileWriter(summary_dir)

    def predict(self, state, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        return sess.run(self.action_values, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        summaries, global_step, _ = sess.run([self.summaries,
                                              self.global_step,
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

def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns Q values for a given state
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

# --------------------------- Q-Learning Algorithm ---------------------------

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
        replay_memory_size (int): Maximum size of the replay memory.
        replay_memory_init_size (int): Initial size to populate the replay memory.
        experiment_dir (str): Directory for experiment logs.
        update_target_estimator_every (int): Frequency to update the target estimator.
        discount_factor (float): Discount factor for RL.
        epsilon_start (float): Starting value of epsilon.
        epsilon_end (float): Final value of epsilon.
        epsilon_decay_steps (int): Number of steps over which epsilon decays.
        batch_size (int): Batch size for training.
        num_LabelPropagation (int): Number of samples for label propagation.
        num_active_learning (int): Number of samples for active learning.
        test (int): If set, run in test mode.
        vae_model (keras.Model): Trained VAE model.

    Returns:
        None
    """
    # Define Transition tuple
    Transition = namedtuple("Transition", ["state", "reward", "next_state", "done"])

    # Initialize replay memory
    replay_memory = []

    # Create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Saver for TensorFlow
    saver = tf.compat.v1.train.Saver()

    # Load a previous checkpoint if available
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print(f"\nLoading model checkpoint {latest_checkpoint}...\n")
        saver.restore(sess, latest_checkpoint)
        if test:
            return

    # Get the current step
    try:
        total_t = sess.run(qlearn_estimator.global_step)
    except Exception as e:
        print("Error retrieving global_step:", e)
        total_t = 0

    # Epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # Define the policy
    policy = make_epsilon_greedy_policy(qlearn_estimator, env.action_space_n)

    num_label = 0

    # 2. Populate the replay memory with initial experience by SVM
    popu_time = time.time()

    # Warm up with active learning
    print('\nWarm up starting...')

    outliers_fraction = 0.01
    data_train = []
    for num in range(env.datasetsize):
        env.reset()
        # Remove time window
        data_train.extend(env.states_list)
    # Isolation Forest model
    model = WarmUp().warm_up_isolation_forest(outliers_fraction, np.array(data_train))

    # Label Spreading model
    lp_model = LabelSpreading()

    for t in itertools.count():
        env.reset()
        data = np.array(env.states_list).reshape(-1, n_steps)[:, -1].reshape(-1, 1)
        anomaly_score = model.decision_function(data)  # Typically scores indicating anomaly
        pred_score = [-1 * s + 0.5 for s in anomaly_score]  # Adjust scores as needed
        warm_samples = np.argsort(pred_score)[:5]
        warm_samples = np.append(warm_samples, np.argsort(pred_score)[-5:])

        # Retrieve input for label propagation
        state_list = np.array(env.states_list)
        label_list = [-1] * len(state_list)  # Initialize labels as -1 (unlabeled)

        for sample in warm_samples:
            # Pick up a state from warm_up samples
            state = env.states_list[sample]
            # Update the cursor
            env.timeseries_curser = sample + n_steps
            action_probs = policy(state, epsilons[min(total_t, epsilon_decay_steps - 1)])
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            # Mark the sample as labeled
            env.timeseries.at[env.timeseries_curser, 'label'] = env.timeseries.at[env.timeseries_curser, 'anomaly']
            num_label += 1

            # Retrieve label for propagation
            label_list[sample] = int(env.timeseries.at[env.timeseries_curser, 'anomaly'])

            next_state, reward, done, _ = env.step(action)
            replay_memory.append(Transition(state, reward, next_state, done))

        # Label propagation main process:
        unlabeled_indices = [i for i, e in enumerate(label_list) if e == -1]
        label_list = np.array(label_list)
        lp_model.fit(state_list, label_list)
        pred_entropies = stats.entropy(lp_model.label_distributions_.T)
        # Select up to N samples that are most certain
        certainty_index = np.argsort(pred_entropies)[:num_LabelPropagation]
        # Assign pseudo labels
        for index in certainty_index:
            if index in unlabeled_indices:
                pseudo_label = lp_model.transduction_[index]
                env.timeseries.at[index + n_steps, 'label'] = pseudo_label

        if len(replay_memory) >= replay_memory_init_size:
            break

    popu_time = time.time() - popu_time
    print(f"\nPopulating replay memory took {popu_time:.2f} seconds")

    # 3. Start the main training loop
    for i_episode in range(num_episodes):
        # Save the current checkpoint periodically
        if i_episode % 50 == 49:
            print(f"\nSaving checkpoint at episode {i_episode + 1}/{num_episodes}")
            saver.save(sess, checkpoint_path)

        loop_start_time = time.time()

        # Reset the environment
        state = env.reset()
        while env.datasetidx > env.datasetrng * validation_separate_ratio:
            env.reset()
            print('Double reset due to validation separation')

        # Active Learning
        # Find already labeled samples
        labeled_index = [i for i, e in enumerate(env.timeseries['label']) if e != -1]
        labeled_index = [item for item in labeled_index if item >= n_steps]
        labeled_index = [item - n_steps for item in labeled_index]

        # Initialize Active Learning
        al = active_learning(env=env, N=num_active_learning, strategy='margin_sampling',
                             estimator=qlearn_estimator, already_selected=labeled_index)
        al_samples = al.get_samples()
        print(f'\nLabeling samples: {al_samples} in env {env.datasetidx}')

        # Assign labels to active samples
        for sample in al_samples:
            env.timeseries.at[sample + n_steps, 'label'] = env.timeseries.at[sample + n_steps, 'anomaly']
            num_label += 1

            next_state, reward, done, _ = env.step(action=np.random.choice(action_space_n))
            replay_memory.append(Transition(state, reward, next_state, done))

        # Append to replay memory and handle label propagation
        # Retrieve input for label propagation
        state_list = np.array(env.states_list)
        label_list = np.array(env.timeseries['label'])

        # Label Propagation
        unlabeled_indices = [i for i, e in enumerate(label_list) if e == -1]
        lp_model.fit(state_list, label_list)
        pred_entropies = stats.entropy(lp_model.label_distributions_.T)
        # Select up to N samples that are most certain
        certainty_index = np.argsort(pred_entropies)[:num_LabelPropagation]
        # Assign pseudo labels
        for index in certainty_index:
            if index in unlabeled_indices:
                pseudo_label = lp_model.transduction_[index]
                env.timeseries.at[index, 'label'] = pseudo_label

        # Take actions for labeled samples
        for samples in labeled_index:
            env.timeseries_curser = samples + n_steps
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
            state = env.states_list[samples]

            # Choose an action to take
            action_probs = policy(state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            # Take a step
            next_state, reward, done, _ = env.step(action)

            # Control replay memory
            if len(replay_memory) >= replay_memory_size:
                replay_memory.pop(0)

            replay_memory.append(Transition(state, reward, next_state, done))

        # Update the model
        for i_epoch in range(num_epoches):
            # Add epsilon to TensorBoard
            if qlearn_estimator.summary_writer:
                episode_summary = tf.compat.v1.Summary()
                qlearn_estimator.summary_writer.add_summary(episode_summary, total_t)

            # Update the target estimator periodically
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, qlearn_estimator, target_estimator)
                print("\nCopied model parameters to target network.\n")

            # Sample a minibatch from the replay memory
            if len(replay_memory) < batch_size:
                continue  # Skip if not enough samples

            samples = random.sample(replay_memory, batch_size)
            states_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            if discount_factor > 0:
                # Calculate Q values and targets
                q_values_next = target_estimator.predict(state=next_states_batch)
                targets_batch = reward_batch + (discount_factor *
                                                np.amax(q_values_next, axis=1))
            else:
                targets_batch = reward_batch

            # Perform gradient descent update
            qlearn_estimator.update(state=states_batch, target=targets_batch.astype(np.float32))

            total_t += 1

        # Print out the training progress
        loop_end_time = time.time()
        per_loop_time = loop_end_time - loop_start_time
        print(f"Global step {total_t} @ Episode {i_episode + 1}/{num_episodes}, time: {per_loop_time:.2f} seconds")

    return

# --------------------------- Evaluation Metrics ---------------------------

def evaluate_model(y_true, y_pred):
    """
    Compute various classification metrics.

    Args:
    - y_true (list or np.ndarray): Ground truth labels (0 = Normal, 1 = Anomaly)
    - y_pred (list or np.ndarray): Predicted labels (0 = Normal, 1 = Anomaly)

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
        env: Environment.
        estimator: Trained model.
        num_episodes (int): Number of validation episodes.
        record_dir (str): Directory to save performance records.
        plot (int): If set, plot the results.

    Returns:
        dict: Dictionary with averaged evaluation metrics.
    """
    y_true_all = []
    y_pred_all = []

    for i_episode in range(num_episodes):
        print(f"\nValidation Episode {i_episode + 1}/{num_episodes}")

        policy = make_epsilon_greedy_policy(estimator, env.action_space_n)
        state = env.reset()
        while env.datasetidx < env.datasetrng * validation_separate_ratio:
            state = env.reset()

        for t in itertools.count():
            action_probs = policy(state, 0)  # Use greedy policy
            action = np.argmax(action_probs)

            y_true_all.append(env.timeseries.at[env.timeseries_curser, 'anomaly'])  # True label
            y_pred_all.append(action)  # Predicted label

            next_state, reward, done, _ = env.step(action)
            if done:
                break
            state = next_state[action]

    # Compute evaluation metrics
    results = evaluate_model(y_true_all, y_pred_all)

    # Print metrics
    print("\nValidation Metrics:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    # Optionally, save results to a file
    if record_dir:
        performance_path = os.path.join(record_dir, 'performance_metrics.txt')
        with open(performance_path, 'w') as rec_file:
            for metric, value in results.items():
                rec_file.write(f"{metric}: {value:.4f}\n")
        print(f"\nPerformance metrics saved to {performance_path}")

    # Plotting (if enabled)
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(tau_values, label="Tau over time", alpha=0.8)
        plt.xlabel("Training Steps")
        plt.ylabel("Tau Value")
        plt.title("Evolution of Tau during Training")
        plt.legend()
        if record_dir:
            tau_plot_path = os.path.join(record_dir, 'tau_evolution.png')
        else:
            tau_plot_path = 'tau_evolution.png'
        plt.savefig(tau_plot_path)
        plt.close()
        print(f"Tau evolution plot saved to {tau_plot_path}")

    return results

# --------------------------- Active Learning ---------------------------

class active_learning(object):
    def __init__(self, env, N, strategy, estimator, already_selected):
        """
        Active Learning class to select samples for labeling.

        Args:
            env: Environment.
            N (int): Number of samples to select.
            strategy (str): Strategy for selecting samples.
            estimator: Q-learning estimator.
            already_selected (list): List of already selected sample indices.
        """
        self.env = env
        self.N = N
        self.strategy = strategy
        self.estimator = estimator
        self.already_selected = already_selected

    def get_samples(self):
        """
        Get samples based on the specified strategy.

        Returns:
            list: List of selected sample indices.
        """
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
        """
        Get samples based on a threshold score.

        Args:
            threshold (float): Threshold for selecting samples.

        Returns:
            list: List of selected sample indices.
        """
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
        """
        Label the selected samples manually.

        Args:
            active_samples (list): List of sample indices to label.
        """
        for sample in active_samples:
            print('\nActive Learning found a confusing sample:')
            print(self.env.timeseries['value'].iloc[sample:sample + n_steps])
            print('Please label the last timestamp based on your knowledge:')
            print('0 for non-anomaly; 1 for anomaly')
            try:
                label = int(input())
                if label not in [0, 1]:
                    print("Invalid label. Defaulting to 0.")
                    label = 0
            except:
                print("Invalid input. Defaulting to 0.")
                label = 0
            self.env.timeseries.at[sample + n_steps - 1, 'anomaly'] = label
        return

# --------------------------- Warm-Up Classes ---------------------------

class WarmUp(object):
    def warm_up_SVM(self, outliers_fraction, X_train):
        """
        Warm-up using One-Class SVM.

        Args:
            outliers_fraction (float): Fraction of outliers.
            X_train (np.ndarray): Training data.

        Returns:
            OneClassSVM: Trained One-Class SVM model.
        """
        model = OneClassSVM(gamma='auto', nu=0.95 * outliers_fraction)
        model.fit(X_train)
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
        clf = IsolationForest(contamination=outliers_fraction)
        clf.fit(X_train)
        return clf

# --------------------------- State Function ---------------------------

def RNNBinaryStateFuc(timeseries, timeseries_curser):
    """
    Example state function for the environment.
    Modify this function based on your actual state representation.

    Args:
        timeseries (pd.DataFrame): The time series data.
        timeseries_curser (int): Current cursor position in the time series.

    Returns:
        list: List of states.
    """
    # Example: Return the last 'n_steps' values as the state
    state = timeseries['value'][timeseries_curser - n_steps:timeseries_curser].tolist()
    return state

# --------------------------- Training Function ---------------------------

def train(num_LP, num_AL, discount_factor, learn_tau=True):
    """
    Train the RL agent with learned tau using DRO on KPI dataset.

    Args:
        num_LP (int): Number of Label Propagation samples.
        num_AL (int): Number of Active Learning samples.
        discount_factor (float): Discount factor for RL.
        learn_tau (bool): Whether to learn tau dynamically using DRO.

    Returns:
        dict: Evaluation metrics.
    """
    # Paths to your KPI data
    kpi_train_csv = find_file(train_extract_dir, '*.csv')  # Already set earlier
    kpi_test_hdf = find_file(test_extract_dir, '*.hdf')  # Already set earlier

    # Load and preprocess training data
    x_train = load_normal_data_kpi(kpi_train_csv, exclude_columns=['timestamp', 'label', 'KPI ID'])

    # Train the VAE
    original_dim = x_train.shape[1]  # Number of features, e.g., 1 ('value')
    latent_dim = 10
    intermediate_dim = 64

    vae, encoder = build_vae(original_dim, latent_dim, intermediate_dim)
    print("\nTraining VAE...")
    vae.fit(x_train, x_train, epochs=50, batch_size=32, validation_split=0.1)
    vae_save_path = os.path.join(current_dir, 'vae_model_kpi.h5')
    vae.save(vae_save_path)
    print(f"VAE trained and saved to {vae_save_path}")

    # Load the trained VAE
    vae = load_model(vae_save_path, custom_objects={'Sampling': Sampling}, compile=False)
    print("Loaded trained VAE.")

    # Load and preprocess test data
    df_test = load_test_data_kpi(kpi_test_hdf)

    # Create the environment with KPI data
    env = EnvTimeSeriesfromRepo()
    env.set_train_data(x_train)
    env.set_test_data(df_test)

    # Configure reward function based on whether to learn tau
    if learn_tau:
        env.rewardfnc = lambda timeseries, timeseries_curser, action: RNNBinaryRewardFuc(
            timeseries, timeseries_curser, action, vae
        )
    else:
        env.rewardfnc = RNNBinaryRewardFuc  # Without adaptive tau

    # Set the state function
    env.statefnc = RNNBinaryStateFuc  # Ensure this function is defined

    env.timeseries_curser_init = n_steps
    env.datasetfix = DATAFIXED
    env.datasetidx = 0

    # Create the testing environment
    env_test = EnvTimeSeriesfromRepo()
    env_test.set_train_data(x_train)
    env_test.set_test_data(df_test)
    env_test.rewardfnc = RNNBinaryRewardFucTest

    # Define experiment directories
    exp_relative_dir = ['RLVAL_with_DRO_and_Adaptive_Scaling']
    experiment_dir = os.path.abspath("./exp/{}".format(exp_relative_dir[0]))
    os.makedirs(experiment_dir, exist_ok=True)

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
            record_dir=experiment_dir,
            plot=1
        )

    return optimization_metric

# --------------------------- Plotting Function ---------------------------

def plot_tau_evolution(record_dir=None):
    """
    Plot the evolution of tau during training and save as PNG.

    Args:
        record_dir (str): Directory to save the plot. If None, saves in the current directory.

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    plt.plot(tau_values, label="Tau over time", alpha=0.8)
    plt.xlabel("Training Steps")
    plt.ylabel("Tau Value")
    plt.title("Evolution of Tau during Training")
    plt.legend()
    if record_dir:
        tau_plot_path = os.path.join(record_dir, 'tau_evolution.png')
    else:
        tau_plot_path = 'tau_evolution.png'
    plt.savefig(tau_plot_path)
    plt.close()
    print(f"Tau evolution plot saved to {tau_plot_path}")

# --------------------------- Main Execution ---------------------------

if __name__ == "__main__":
    # Example training runs with different parameters
    print("\nStarting training with num_LP=100, num_AL=30, discount_factor=0.92")
    try:
        metric1 = train(num_LP=100, num_AL=30, discount_factor=0.92, learn_tau=True)
        print(f"Training run 1 completed with metrics: {metric1}")
    except Exception as e:
        print(f"Error during Training run 1: {e}")

    print("\nStarting training with num_LP=150, num_AL=50, discount_factor=0.94")
    try:
        metric2 = train(num_LP=150, num_AL=50, discount_factor=0.94, learn_tau=True)
        print(f"Training run 2 completed with metrics: {metric2}")
    except Exception as e:
        print(f"Error during Training run 2: {e}")

    print("\nStarting training with num_LP=200, num_AL=100, discount_factor=0.96")
    try:
        metric3 = train(num_LP=200, num_AL=100, discount_factor=0.96, learn_tau=True)
        print(f"Training run 3 completed with metrics: {metric3}")
    except Exception as e:
        print(f"Error during Training run 3: {e}")

    # Optionally, plot tau evolution for the last training run
    # plot_tau_evolution(record_dir='path_to_save_plot')  # Uncomment and set path if needed
