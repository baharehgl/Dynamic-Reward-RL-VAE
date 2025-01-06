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
from scipy.optimize import minimize
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.ensemble import IsolationForest
from collections import namedtuple

tf.compat.v1.disable_eager_execution()
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

# Constants
EPISODES = 500
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0  # Initial Epsilon for Exploration
EPSILON_DECAY = 0.995  # Decay Rate for Epsilon
MIN_EPSILON = 0.1  # Minimum value for Epsilon
NOT_ANOMALY = 0
ANOMALY = 1
action_space = [NOT_ANOMALY, ANOMALY]
n_steps = 25
n_input_dim = 2
n_hidden_dim = 128
TP_Value = 5
TN_Value = 1
FP_Value = -1
FN_Value = -5
validation_separate_ratio = 0.9

# Adaptive Scaling Parameters
rho = 0.01
tau_min = 0.1
tau_max = 5.0

########################### Environment for Time-Series RL #####################
class EnvTimeSeries:
    def __init__(self, data_path, n_steps=25):
        """
        Initializes the RL environment for time-series anomaly detection.
        """
        self.data_path = data_path
        self.n_steps = n_steps
        self.timeseries = self.load_data()
        self.cursor = n_steps  # Start cursor after collecting enough data points
        self.done = False  # Flag to check if episode is done

    def load_data(self):
        """
        Loads and preprocesses multiple CSV files from the specified directory.
        """
        if not os.path.isdir(self.data_path):
            raise NotADirectoryError(f"Provided path is not a directory: {self.data_path}")

        all_files = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path) if f.endswith('.csv')]

        if not all_files:
            raise FileNotFoundError(f"No CSV files found in directory: {self.data_path}")

        # Read and combine all CSV files
        data_list = [pd.read_csv(file) for file in all_files]
        data = pd.concat(data_list, ignore_index=True)

        # Normalize 'value' column if it exists
        scaler = StandardScaler()
        if 'value' in data.columns:
            data['value'] = scaler.fit_transform(data[['value']])
        else:
            raise KeyError("Column 'value' not found in dataset!")

        return data

    def reset(self):
        """
        Resets the environment and returns the initial state.
        """
        self.cursor = self.n_steps
        self.done = False
        return self.get_state()

    def get_state(self):
        """
        Returns the current state (sliding window of time-series data).
        """
        state = self.timeseries.iloc[self.cursor - self.n_steps:self.cursor][['value']].values
        return np.expand_dims(state, axis=0)  # Reshape for LSTM input

    def step(self, action):
        """
        Takes an action and returns next state, reward, and done flag.
        """
        reward, tau = RNNBinaryRewardFuc(self.timeseries, self.cursor, action, vae)

        self.cursor += 1  # Move to the next time step
        if self.cursor >= len(self.timeseries) - 1:
            self.done = True

        next_state = self.get_state()
        return next_state, reward, self.done, tau


########################### VAE #####################
def load_normal_data(data_path):
    all_files = [os.path.join(data_path, fname) for fname in os.listdir(data_path) if fname.endswith('.csv')]
    data_list = [pd.read_csv(file) for file in all_files]
    data = pd.concat(data_list, axis=0, ignore_index=True)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values)
    return scaled_data

data_directory = r'C:\Users\Asus\Documents\GitHub\Adaptive-Reward-Scaling-Reinforcement-Learning\normal-data'

x_train = load_normal_data(data_directory)

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=(tf.shape(z_mean)[0], tf.shape(z_mean)[1]))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(original_dim, latent_dim=10, intermediate_dim=64):
    inputs = tf.keras.layers.Input(shape=(original_dim,))
    h = tf.keras.layers.Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = tf.keras.layers.Dense(latent_dim)(h)
    z_log_var = tf.keras.layers.Dense(latent_dim)(h)
    z = Sampling()([z_mean, z_log_var])
    decoder_h = tf.keras.layers.Dense(intermediate_dim, activation='relu')
    decoder_mean = tf.keras.layers.Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)
    encoder = tf.keras.models.Model(inputs, [z_mean, z_log_var, z])
    vae = tf.keras.models.Model(inputs, x_decoded_mean)
    reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, x_decoded_mean) * original_dim
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return vae, encoder

vae, encoder = build_vae(original_dim=3)

########################### Adaptive Preference Scaling #####################
def optimize_tau(r1, r2, rho=0.01, tau_min=0.1, tau_max=5.0):
    def loss(tau):
        sigma = 1 / (1 + np.exp(-(r1 - r2) / tau))
        return -tau * np.log(sigma) + rho * tau
    result = minimize(loss, x0=1.0, bounds=[(tau_min, tau_max)], method='L-BFGS-B')
    return result.x[0]

def RNNBinaryRewardFuc(timeseries, timeseries_curser, action=0, vae=None, rho=0.01, tau_min=0.1, tau_max=5.0):
    if timeseries_curser >= n_steps:
        current_state = np.array([timeseries['value'][timeseries_curser - n_steps:timeseries_curser]])
        vae_reconstruction = vae.predict(current_state)
        reconstruction_error = np.mean(np.square(vae_reconstruction - current_state))
        r1 = TP_Value if timeseries['label'][timeseries_curser] == 1 else TN_Value
        r2 = -reconstruction_error
        tau = optimize_tau(r1, r2, rho, tau_min, tau_max)
        adaptive_reward = tau * r1 + (1 - tau) * r2
        return [adaptive_reward, adaptive_reward], tau
    else:
        return [0, 0], 0

########################### RL: Q-learning with LSTM #####################
class QLearningAgent:
    def __init__(self, learning_rate=0.01):
        self.state = tf.compat.v1.placeholder(shape=[None, n_steps, n_input_dim], dtype=tf.float32, name="state")
        self.target = tf.compat.v1.placeholder(shape=[None, len(action_space)], dtype=tf.float32, name="target")
        lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(n_hidden_dim, forget_bias=1.0)
        self.outputs, _ = tf.compat.v1.nn.dynamic_rnn(lstm_cell, self.state, dtype=tf.float32)
        self.q_values = tf.keras.layers.Dense(len(action_space))(self.outputs[:, -1, :])
        self.loss = tf.reduce_mean(tf.compat.v1.losses.mean_squared_error(self.q_values, self.target))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def predict(self, sess, state):
        return sess.run(self.q_values, feed_dict={self.state: state})

    def update(self, sess, state, target):
        sess.run(self.optimizer, feed_dict={self.state: state, self.target: target})

########################### Active Learning #####################
class ActiveLearning:
    def __init__(self, env, num_samples, estimator):
        self.env = env
        self.num_samples = num_samples
        self.estimator = estimator

    def select_samples(self):
        states_list = self.env.states_list
        distances = []
        for state in states_list:
            q = self.estimator.predict(sess, [state])[0]
            distance = abs(q[0] - q[1])
            distances.append(distance)
        return np.argsort(distances)[:self.num_samples]

    def label_samples(self, samples):
        for sample in samples:
            print(f"Label sample: {sample} (0 = normal, 1 = anomaly)")
            label = input()
            self.env.timeseries.loc[sample + n_steps - 1, 'anomaly'] = int(label)

########################### Training Loop #####################
#env = EnvTimeSeries(data_path=r'C:\Users\Asus\Documents\GitHub\Adaptive-Reward-Scaling-Reinforcement-Learning\time-series.csv')
env = EnvTimeSeries(data_path=r'C:\Users\Asus\Documents\GitHub\Adaptive-Reward-Scaling-Reinforcement-Learning\ydata-labeled-time-series-anomalies-v1_0\A1Benchmark')

sess = tf.compat.v1.Session()
agent = QLearningAgent()
sess.run(tf.compat.v1.global_variables_initializer())

tau_values = []
step_values = []
epsilon = EPSILON
num_active_learning_samples = 5

for step in range(EPISODES):
    state = env.reset()  # ✅ FIXED: Properly initializing the environment
    done = False

    while not done:  # Run episode until termination
        action_probs = agent.predict(sess, [state])[0]
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        next_state, reward, done, tau = env.step(action)  # ✅ FIXED: Now using environment

        tau_values.append(tau)
        step_values.append(step)

        # Perform Active Learning every 50 steps
        if step % 50 == 0:
            al = ActiveLearning(env, num_active_learning_samples, estimator=agent)
            selected_samples = al.select_samples()
            al.label_samples(selected_samples)

        # Update RL model
        target_q_values = reward + DISCOUNT_FACTOR * np.max(agent.predict(sess, [next_state])[0])
        agent.update(sess, [state], [target_q_values])

        state = next_state  # Move to next state

    # Update epsilon for exploration
    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

########################### Visualization #####################
plt.figure(figsize=(10, 5))
plt.plot(step_values, tau_values, label="Adaptive Scaling Factor (τ)", color='b')
plt.xlabel("Training Steps")
plt.ylabel("Tau (τ)")
plt.title("Evolution of Adaptive Scaling Factor (τ) Over Training")
plt.legend()
plt.grid()
plt.show()
