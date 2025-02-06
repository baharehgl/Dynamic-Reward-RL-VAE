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
from collections import deque, namedtuple
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.semi_supervised import LabelSpreading

tf.compat.v1.disable_eager_execution()


# Get the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Define data paths
NORMAL_DATA_PATH = os.path.join(project_root, 'normal-data')
ANOMALY_DATA_PATH = os.path.join(project_root, 'ydata-labeled-time-series-anomalies-v1_0')
# ================== ADAPTIVE SCALING CLASS ==================
class AdaptiveScaling:
    """Dynamically adjusts reward coefficients α (extrinsic) and β (intrinsic)"""

    def __init__(self, alpha=0.5, beta=0.5, decay=0.95, min_val=0.1, max_val=1.0):
        self.alpha = alpha
        self.beta = beta
        self.decay = decay
        self.min = min_val
        self.max = max_val
        self.r1_history = deque(maxlen=100)
        self.r2_history = deque(maxlen=100)

    def update(self, r1_batch, r2_batch):
        # Update moving averages
        self.r1_history.extend(r1_batch)
        self.r2_history.extend(r2_batch)

        # Calculate variance-based adjustment
        var_r1 = np.var(list(self.r1_history)) if self.r1_history else 1.0
        var_r2 = np.var(list(self.r2_history)) if self.r2_history else 1.0

        # Adjust coefficients with stability controls
        new_alpha = np.clip(var_r2 / (var_r1 + var_r2 + 1e-8), self.min, self.max)
        new_beta = 1.0 - new_alpha

        # Apply exponential smoothing
        self.alpha = self.decay * self.alpha + (1 - self.decay) * new_alpha
        self.beta = self.decay * self.beta + (1 - self.decay) * new_beta

    def get_coefficients(self):
        return self.alpha, self.beta


# Global adaptive scaler
adaptive_scaler = AdaptiveScaling(alpha=0.7, beta=0.3, decay=0.9)
# ============================================================

# Constants and hyperparameters
EPISODES = 500
DISCOUNT_FACTOR = 0.5
EPSILON = 0.5
EPSILON_DECAY = 1.00
n_steps = 25
n_input_dim = 2
n_hidden_dim = 128

# Reward Values
TP_Value = 5
TN_Value = 1
FP_Value = -1
FN_Value = -5


class Sampling(layers.Layer):
    """VAE sampling layer"""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae(original_dim, latent_dim=10, intermediate_dim=64):
    inputs = layers.Input(shape=(original_dim,))
    h = layers.Dense(intermediate_dim, activation='relu')(inputs)
    h = layers.Dense(intermediate_dim, activation='relu')(h)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    z = Sampling()([z_mean, z_log_var])

    decoder_h = layers.Dense(intermediate_dim, activation='relu')(z)
    decoder_h = layers.Dense(intermediate_dim, activation='relu')(decoder_h)
    x_decoded = layers.Dense(original_dim, activation='sigmoid')(decoder_h)

    vae = models.Model(inputs, x_decoded)

    reconstruction_loss = losses.mse(inputs, x_decoded) * original_dim
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return vae


# Modified Reward Function with Adaptive Scaling
def RNNBinaryRewardFuc(timeseries, timeseries_curser, action=0, vae=None):
    global adaptive_scaler

    if timeseries_curser >= n_steps:
        # Get current state
        current_state = np.array([timeseries['value'][timeseries_curser - n_steps:timeseries_curser]])

        # Extrinsic reward (environment-based)
        if timeseries['label'][timeseries_curser] == 0:
            r1 = [TN_Value, FP_Value]
        else:
            r1 = [FN_Value, TP_Value]

        # Intrinsic reward (VAE reconstruction error)
        vae_reconstruction = vae.predict(current_state, verbose=0)
        reconstruction_error = np.mean(np.square(vae_reconstruction - current_state))
        r2 = [-reconstruction_error] * 2  # Penalize both actions equally

        # Get adaptive coefficients
        alpha, beta = adaptive_scaler.get_coefficients()

        # Combine rewards
        combined_reward = [
            alpha * r1[0] + beta * r2[0],
            alpha * r1[1] + beta * r2[1]
        ]

        return combined_reward
    else:
        return [0, 0]


class Q_Estimator_Nonlinear:
    def __init__(self, learning_rate=0.01, scope="Q_Estimator_Nonlinear", summaries_dir=None):
        self.scope = scope
        with tf.compat.v1.variable_scope(scope):
            self.state = tf.compat.v1.placeholder(shape=[None, n_steps, n_input_dim], dtype=tf.float32)
            self.target = tf.compat.v1.placeholder(shape=[None, 2], dtype=tf.float32)

            # LSTM Network
            lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(n_hidden_dim)
            self.outputs, self.states = tf.compat.v1.nn.static_rnn(
                lstm_cell,
                tf.unstack(self.state, n_steps, 1),
                dtype=tf.float32
            )

            # Output layer
            self.action_values = layers.Dense(2)(self.outputs[-1])

            # Optimization
            self.loss = tf.reduce_mean(tf.square(self.action_values - self.target))
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)


def q_learning(env, sess, qlearn_estimator, target_estimator, num_episodes,
               num_epoches, vae_model=None, **kwargs):
    Transition = namedtuple("Transition", ["state", "reward", "next_state", "done"])
    replay_memory = []

    # Training loop
    for i_episode in range(num_episodes):
        state = env.reset()
        episode_r1, episode_r2 = [], []

        # Active learning and environment interaction
        # ... (maintain existing active learning logic)

        # Batch processing
        for i_epoch in range(num_epoches):
            if len(replay_memory) < batch_size:
                continue

            samples = random.sample(replay_memory, batch_size)
            states_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            # Calculate rewards for adaptation
            batch_r1, batch_r2 = [], []
            for transition in samples:
                # Calculate intrinsic reward
                vae_recon = vae_model.predict(np.array([transition.state]), verbose=0)
                recon_error = np.mean(np.square(vae_recon - transition.state))
                batch_r2.append(-recon_error)

                # Store extrinsic reward
                batch_r1.append(np.max(transition.reward))

            # Update adaptive scaling
            adaptive_scaler.update(batch_r1, batch_r2)

            # Q-learning update
            targets = reward_batch + DISCOUNT_FACTOR * np.amax(target_estimator.predict(next_states_batch), axis=1)
            qlearn_estimator.update(states_batch, targets)

        # Print coefficients
        if i_episode % 10 == 0:
            alpha, beta = adaptive_scaler.get_coefficients()
            print(f"Episode {i_episode}: α={alpha:.2f}, β={beta:.2f}")


def train(num_LP, num_AL, discount_factor):
    # VAE initialization
    original_dim = 3
    vae = build_vae(original_dim)

    # Load and preprocess data
    data_directory = 'path/to/normal-data'
    x_train = load_normal_data(data_directory)
    vae.fit(x_train, epochs=50, batch_size=32)

    # Environment setup
    env = EnvTimeSeriesfromRepo('path/to/dataset')
    env.rewardfnc = lambda ts, tc, a: RNNBinaryRewardFuc(ts, tc, a, vae)

    # Initialize Q-networks
    tf.compat.v1.reset_default_graph()
    qlearn_estimator = Q_Estimator_Nonlinear()
    target_estimator = Q_Estimator_Nonlinear()

    # Training session
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        q_learning(env, sess, qlearn_estimator, target_estimator,
                   num_episodes=300, num_epoches=10, vae_model=vae)


if __name__ == "__main__":
    train(num_LP=100, num_AL=30, discount_factor=0.95)