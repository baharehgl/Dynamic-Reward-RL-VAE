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
from sklearn.metrics import pairwise_distances

tf.compat.v1.disable_eager_execution()

from mpl_toolkits.mplot3d import axes3d
from collections import deque, namedtuple
from tensorflow.keras import layers, models, losses, optimizers
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Append current directory to sys.path so local modules can be imported.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import the environment.
from env import EnvTimeSeriesfromRepo
from sklearn.svm import OneClassSVM
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

############################
# Macros and Hyperparameters.
DATAFIXED = 0  # whether target is fixed to a single time series
EPISODES = 500  # Increased training episodes
DISCOUNT_FACTOR = 0.9  # Adjusted discount factor for long-term rewards
EPSILON = 1.0  # Initial exploration rate
EPSILON_DECAY = 0.99  # Added epsilon decay
MIN_EPSILON = 0.01

# Extrinsic reward values (heuristic):
TN_Value = 2  # True Negative
TP_Value = 10  # True Positive
FP_Value = -5  # False Positive
FN_Value = -8  # False Negative

NOT_ANOMALY = 0
ANOMALY = 1
action_space = [NOT_ANOMALY, ANOMALY]
action_space_n = len(action_space)

n_steps = 30  # sliding window length
n_input_dim = 3  # dimension of input to LSTM (value and action indicator)
n_hidden_dim = 256  # hidden dimension

validation_separate_ratio = 0.85


########################### Enhanced VAE #####################
class EnhancedSampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_enhanced_vae(original_dim, latent_dim=64, intermediate_dim=128):
    inputs = layers.Input(shape=(original_dim,))

    # Encoder
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)

    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z = EnhancedSampling()([z_mean, z_log_var])

    # Decoder
    d = layers.Dense(intermediate_dim)(z)
    d = layers.Reshape((-1, 64))(d)
    d = layers.UpSampling1D(2)(d)
    d = layers.Conv1DTranspose(64, 3, activation='relu', padding='same')(d)
    outputs = layers.Conv1D(1, 3, activation='sigmoid', padding='same')(d)

    vae = models.Model(inputs, outputs)
    reconstruction_loss = losses.mse(inputs, outputs) * original_dim
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    vae_loss = reconstruction_loss + kl_loss
    vae.add_loss(vae_loss)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            0.001, decay_steps=1000, decay_rate=0.9
        )
    )
    vae.compile(optimizer=optimizer)
    return vae


########################### Enhanced Q-Network #####################
class EnhancedQEstimator:
    def __init__(self, learning_rate=0.001, scope="EnhancedQEstimator"):
        self.scope = scope
        with tf.compat.v1.variable_scope(scope):
            self.state = tf.compat.v1.placeholder(shape=[None, n_steps, n_input_dim], dtype=tf.float32)
            self.target = tf.compat.v1.placeholder(shape=[None, action_space_n], dtype=tf.float32)

            # Enhanced architecture
            x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(self.state)
            x = layers.Dropout(0.3)(x)
            x = layers.Bidirectional(layers.LSTM(128))(x)
            x = layers.Dense(128, activation='relu')(x)
            self.action_values = layers.Dense(action_space_n)(x)

            self.loss = tf.reduce_mean(tf.square(self.action_values - self.target))
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)

    def predict(self, state, sess):
        return sess.run(self.action_values, {self.state: state})

    def update(self, state, target, sess):
        sess.run(self.train_op, feed_dict={self.state: state, self.target: target})


########################### Enhanced Active Learning #####################
class EnhancedActiveLearning:
    def __init__(self, env, estimator, strategy="uncertainty",
                 initial_samples=5, max_samples=100):
        self.env = env
        self.estimator = estimator
        self.strategy = strategy
        self.labeled_indices = set()
        self.max_samples = max_samples
        self.initial_samples = initial_samples

    def _calculate_uncertainty(self, states):
        uncertainties = []
        for state in states:
            q_values = self.estimator.predict([state])
            uncertainty = 1 - (np.max(q_values) - np.min(q_values))
            uncertainties.append(uncertainty)
        return np.array(uncertainties)

    def select_samples(self, batch_size=5):
        states = self.env.get_states()
        if len(self.labeled_indices) < self.initial_samples:
            return random.sample(range(len(states)), self.initial_samples)

        if self.strategy == "uncertainty":
            uncertainties = self._calculate_uncertainty(states)
            candidates = np.argsort(uncertainties)[-batch_size * 3:]
            return [i for i in candidates if i not in self.labeled_indices][:batch_size]

        elif self.strategy == "diversity":
            # Implement diversity sampling using VAE latent space
            latent_states = self.env.vae.encoder.predict(states)
            distances = pairwise_distances(latent_states)
            diverse_indices = np.argpartition(distances.mean(axis=1), -batch_size)[-batch_size:]
            return [i for i in diverse_indices if i not in self.labeled_indices]

        return random.sample(range(len(states)), batch_size)

    def label_samples(self, indices):
        new_labels = []
        for idx in indices:
            if idx in self.labeled_indices:
                continue
            print(f"Labeling sample {idx}:")
            print(self.env.timeseries.iloc[idx:idx + n_steps])
            label = input("Enter label (0-normal/1-anomaly): ")
            self.env.timeseries.loc[idx + n_steps - 1, 'label'] = int(label)
            self.labeled_indices.add(idx)
            new_labels.append((idx, int(label)))
        return new_labels

    def update_learning(self, session, batch_size=32):
        if len(self.labeled_indices) < self.initial_samples:
            return

        # Get labeled data
        labeled_states = [self.env.states[i] for i in self.labeled_indices]
        labels = [self.env.timeseries.loc[i + n_steps - 1, 'label']
                  for i in self.labeled_indices]

        # Semi-supervised learning
        lp_model = LabelSpreading(kernel='knn', n_neighbors=5)
        lp_model.fit(labeled_states, labels)
        pseudo_labels = lp_model.transduction_

        # Update Q-network with pseudo-labels
        for i in range(0, len(pseudo_labels), batch_size):
            batch_states = self.env.states[i:i + batch_size]
            batch_labels = pseudo_labels[i:i + batch_size]
            targets = self.estimator.predict(batch_states, session)
            targets[np.arange(len(batch_labels)), batch_labels] += 1.0
            self.estimator.update(batch_states, targets, session)


########################### Training Enhancements #####################
def improved_reward_function(timeseries, cursor, action, vae, dynamic_coef=1.0):
    if cursor >= n_steps:
        window = timeseries['value'][cursor - n_steps:cursor].values.reshape(1, -1)
        reconstructed = vae.predict(window)

        # Enhanced reconstruction error calculation
        mse = np.mean(np.square(window - reconstructed))
        dynamic_penalty = dynamic_coef * mse

        # Temporal consistency check
        prev_window = timeseries['value'][cursor - n_steps - 5:cursor - 5].values.reshape(1, -1)
        if len(prev_window[0]) == n_steps:
            prev_reconstructed = vae.predict(prev_window)
            temporal_diff = np.abs(reconstructed - prev_reconstructed).mean()
            dynamic_penalty += temporal_diff * 0.5

        if timeseries['label'][cursor] == 0:
            return [TN_Value - dynamic_penalty, FP_Value - dynamic_penalty]
        else:
            return [FN_Value + dynamic_penalty, TP_Value + dynamic_penalty]
    return [0, 0]


def enhanced_active_learning(env, estimator, samples=5):
    states = env.get_states()
    uncertainties = []

    for state in states:
        q_values = estimator.predict([state])
        uncertainty = np.abs(q_values[0][0] - q_values[0][1])
        uncertainties.append(uncertainty)

    uncertain_indices = np.argsort(uncertainties)[-samples:]
    return uncertain_indices


def train_vae(vae, data, epochs=50, batch_size=128):
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    vae.fit(data, epochs=epochs, batch_size=batch_size,
            validation_split=0.1, callbacks=[early_stop])


def copy_model_parameters(sess, estimator1, estimator2):
    e1_params = sorted([t for t in tf.compat.v1.trainable_variables() if t.name.startswith(estimator1.scope)],
                       key=lambda v: v.name)
    e2_params = sorted([t for t in tf.compat.v1.trainable_variables() if t.name.startswith(estimator2.scope)],
                       key=lambda v: v.name)
    for e1_v, e2_v in zip(e1_params, e2_params):
        sess.run(e2_v.assign(e1_v))


def make_epsilon_greedy_policy(estimator, nA, sess):
    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype='float32') * epsilon / nA
        q_values = estimator.predict(state=[observation], sess=sess)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


# Proportional update for dynamic coefficient.
def update_dynamic_coef_proportional(current_coef, episode_reward, target_reward=100.0, alpha=0.005, min_coef=0.1,
                                     max_coef=10.0):
    new_coef = current_coef + alpha * (target_reward - episode_reward)
    return max(min(new_coef, max_coef), min_coef)


# --- Active Learning class remains unchanged.
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
            distances.append(abs(q[0] - q[1]))
        distances = np.array(distances)
        rank_ind = np.argsort(distances)
        rank_ind = [i for i in rank_ind if i < len(states_list) and i not in self.already_selected]
        return rank_ind[:self.N]

    def get_samples_by_score(self, threshold):
        states_list = self.env.states_list
        distances = []
        for state in states_list:
            q = self.estimator.predict(state=[state])[0]
            distances.append(abs(q[0] - q[1]))
        distances = np.array(distances)
        rank_ind = np.argsort(distances)
        rank_ind = [i for i in rank_ind if i < len(states_list) and i not in self.already_selected]
        return [t for t in rank_ind if distances[t] < threshold]

    def label(self, active_samples):
        for sample in active_samples:
            print('AL finds one of the most confused samples:')
            print(self.env.timeseries['value'].iloc[sample:sample + n_steps])
            print('Please label the last timestamp (0 for normal, 1 for anomaly):')
            label = input()
            self.env.timeseries.loc[sample + n_steps - 1, 'anomaly'] = float(label)
        return


class WarmUp(object):
    def warm_up_SVM(self, outliers_fraction, N):
        states_list = self.env.get_states_list()
        data = np.array(states_list).transpose(2, 0, 1).reshape(2, -1)[0].reshape(-1, n_steps)[:, -1].reshape(-1, 1)
        model = OneClassSVM(gamma='auto', nu=0.95 * outliers_fraction)
        model.fit(data)
        distances = model.decision_function(data)
        rank_ind = np.argsort(np.abs(distances))
        samples = rank_ind[0:N]
        return samples

    def warm_up_isolation_forest(self, outliers_fraction, X_train):
        from sklearn.ensemble import IsolationForest
        X_train_arr = np.array(X_train)
        data = X_train_arr[:, -1].reshape(-1, 1)
        clf = IsolationForest(contamination=outliers_fraction)
        clf.fit(data)
        return clf


def q_learning(env, sess, qlearn_estimator, target_estimator, num_episodes, num_epoches,
               replay_memory_size=500000, replay_memory_init_size=50000, experiment_dir='./log/',
               update_target_estimator_every=10000, discount_factor=0.99,
               epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=500000, batch_size=256,
               num_LabelPropagation=20, num_active_learning=5, test=0, vae_model=None):
    Transition = namedtuple("Transition", ["state", "reward", "next_state", "done"])
    replay_memory = []
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
    global_step_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="global_step")
    if len(global_step_list) == 0:
        raise ValueError("global_step variable not found!")
    total_t = sess.run(global_step_list[0])
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    policy = make_epsilon_greedy_policy(qlearn_estimator, env.action_space_n, sess)
    num_label = 0
    print('Warm up starting...')
    outliers_fraction = 0.01
    max_warmup_samples = 10000
    data_train = []
    for num in range(env.datasetsize):
        env.reset()
        env.states_list = [s for s in env.states_list if s is not None]
        data_train.extend(env.states_list)
        if len(data_train) >= max_warmup_samples:
            data_train = data_train[:max_warmup_samples]
            break
    model_warm = WarmUp().warm_up_isolation_forest(outliers_fraction, data_train)
    lp_model = LabelSpreading()
    for t in itertools.count():
        env.reset()
        env.states_list = [s for s in env.states_list if s is not None]
        env.states = [s for s in env.states if s is not None]
        state = env.reset()
        state = state[0]

        # Active learning sampling
        if len(replay_memory) % 100 == 0:
            samples = active_learner.select_samples()
            new_labels = active_learner.label_samples(samples)
            if new_labels:
                active_learner.update_learning(sess)

        for time_steps in range(num_epoches):
            epsilon = epsilons[
                min(total_t, int(epsilon_decay_steps) - 1)] if total_t < epsilon_decay_steps else epsilon_end
            if np.random.rand() < epsilon:
                action = np.random.choice(action_space)
            else:
                q_values = qlearn_estimator.predict([state], sess)
                action = np.argmax(q_values)

            next_state, reward, done = env.step(action)
            next_state = next_state[0]
            replay_memory.append(Transition(state=state, reward=reward, next_state=next_state, done=done))

            if len(replay_memory) > replay_memory_size:
                replay_memory.pop(0)

            if total_t > replay_memory_init_size:
                samples = random.sample(replay_memory, batch_size)
                states_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))
                q_values_next = target_estimator.predict(next_states_batch, sess)
                best_actions = np.argmax(q_values_next, axis=1)
                q_values_next_target = qlearn_estimator.predict(next_states_batch, sess)
                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32) * discount_factor * \
                                q_values_next_target[np.arange(batch_size), best_actions]

                states_batch = np.array(states_batch)
                targets_batch = np.array(targets_batch)
                q_values_batch = qlearn_estimator.predict(states_batch, sess)
                q_values_batch[np.arange(batch_size), action] = targets_batch
                qlearn_estimator.update(states_batch, q_values_batch, sess)
                if total_t % update_target_estimator_every == 0:
                    copy_model_parameters(sess, qlearn_estimator, target_estimator)
                    print("\nCopied model parameters to target network.")
            state = next_state

            if done:
                break
            total_t += 1
    return qlearn_estimator


def enhanced_q_learning(env, sess, q_estimator, target_estimator,
                        num_episodes=EPISODES, vae=None):
    # Initialize active learning module
    active_learner = EnhancedActiveLearning(env, q_estimator)

    # Initialize replay memory
    replay_memory = deque(maxlen=100000)

    # Training loop
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        dynamic_coef = 1.0

        while True:
            # Epsilon-greedy action selection
            if np.random.rand() < EPSILON:
                action = np.random.choice(action_space)
            else:
                q_values = q_estimator.predict([state], sess)
                action = np.argmax(q_values)

            # Take action and get next state
            next_state, reward, done = env.step(action)

            # Store transition in replay memory
            replay_memory.append((state, action, reward, next_state, done))

            # Active learning sampling
            if len(replay_memory) % 100 == 0:
                samples = active_learner.select_samples()
                new_labels = active_learner.label_samples(samples)
                if new_labels:
                    active_learner.update_learning(sess)

            # Experience replay
            if len(replay_memory) >= batch_size:
                batch = random.sample(replay_memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                # Double Q-learning update
                next_q_values = q_estimator.predict(next_states, sess)
                best_actions = np.argmax(next_q_values, axis=1)
                next_q_values_target = target_estimator.predict(next_states, sess)
                targets = rewards + (1 - dones) * DISCOUNT_FACTOR * \
                          next_q_values_target[np.arange(batch_size), best_actions]

                # Update Q-network
                current_q = q_estimator.predict(states, sess)
                current_q[np.arange(batch_size), actions] = targets
                q_estimator.update(states, current_q, sess)

            # Update dynamic coefficient
            dynamic_coef = update_dynamic_coef_proportional(
                dynamic_coef, episode_reward,
                target_reward=TP_Value * 10,
                alpha=0.005
            )

            # Update target network
            if episode % update_target_every == 0:
                copy_model_parameters(sess, q_estimator, target_estimator)

            if done:
                break

        # Epsilon decay
        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    return q_estimator


# Main training loop remains similar but with enhanced components
def test_and_visualize(qlearn_estimator, env, data_dir, test_size=0.33, visualize=False):
    # --- Load and prepare test data.
    test_file_list = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    test_data_list = [os.path.join(data_dir, file) for file in test_file_list]
    test_timeseries = pd.read_csv(test_data_list[0], sep=',', header=None)
    test_timeseries.columns = ['value'] + ['sensor_' + str(i) for i in range(1, test_timeseries.shape[1])]
    test_timeseries['anomaly'] = test_timeseries.apply(lambda row: 1 if row['sensor_1'] == 1 else 0, axis=1)
    test_timeseries = test_timeseries[['value', 'anomaly']]
    test_states = []
    test_states_list = []
    for i in range(n_steps, len(test_timeseries)):
        state = RNNBinaryStateFuc(test_timeseries, i)
        if state is not None:
            test_states.append(state[0])
            test_states_list.append(state)
    test_states = np.array(test_states, dtype='float32')
    if visualize:
        x = [test_timeseries['value'].iloc[i] for i in range(len(test_timeseries))]
        pre_action = [np.argmax(qlearn_estimator.predict(state=[test_states[i]])) for i in range(len(test_states))]
        action_change_ix = [i for i in range(1, len(pre_action)) if pre_action[i] != pre_action[i - 1]]
        action_change_value = [test_timeseries['value'].iloc[i] for i in action_change_ix]
        plt.figure(figsize=(15, 4))
        plt.plot(x, color='blue', label='Value')
        plt.scatter(action_change_ix, action_change_value, color='red', label='Action Changes')
        plt.title('Time Series Visualization with Action Changes')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
        plt.savefig('result_test.png')

    y_true = test_timeseries['anomaly'][n_steps:].values
    y_pred = [np.argmax(qlearn_estimator.predict(state=[test_states[i]])) for i in range(len(test_states))]
    precision, recall, fscore, support = precision_recall_fscore_support(y_true[:len(y_pred)], y_pred, average='binary')
    au_pr = average_precision_score(y_true[:len(y_pred)], y_pred)
    print(f"Precision:{precision}, Recall:{recall}, F1-score:{fscore}, AU-PR:{au_pr}")


if __name__ == '__main__':
    # --- Setup:
    SMD_ROOT = './datasets/SMD'
    machine_id = 'machine-1-1'
    data_dir = os.path.join(SMD_ROOT, 'train', machine_id)
    data_dir_test = os.path.join(SMD_ROOT, 'test', machine_id)
    normal_data = load_normal_data(data_dir, n_steps)
    time_series_file = os.path.join(SMD_ROOT, 'test', machine_id + '.txt')

    # --- Load in time series, pre-label and split into training and test sets.
    timeseries = pd.read_csv(time_series_file, sep=',', header=None)
    timeseries.columns = ['value'] + ['sensor_' + str(i) for i in range(1, timeseries.shape[1])]
    timeseries['label'] = timeseries.apply(lambda row: 1 if row['sensor_1'] == 1 else 0, axis=1)
    timeseries = timeseries[['value', 'label']]

    # --- Define training/testing split:
    train_separate_index = int(len(timeseries) * validation_separate_ratio)
    timeseries_train = timeseries[:train_separate_index]
    timeseries_test = timeseries[train_separate_index:]

    # --- Initialize environment.
    env = EnvTimeSeriesfromRepo(time_series=timeseries_train)
    env.reset()
    env.states_list = [s for s in env.states_list if s is not None]
    env.states = [s for s in env.states if s is not None]
    states_list = env.states_list

    # Initialize VAE, Q-estimators, and target Q-estimator
    vae = build_enhanced_vae(original_dim=n_steps, latent_dim=latent_dim, intermediate_dim=intermediate_dim)
    train_vae(vae, normal_data, epochs=10)
    qlearn_estimator = EnhancedQEstimator(learning_rate=0.001, scope="qlearn")
    target_estimator = EnhancedQEstimator(learning_rate=0.001, scope="target")
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # --- Session setup.
    tf_config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
    with tf.compat.v1.Session(config=tf_config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        # Initialize active learning module
        active_learner = EnhancedActiveLearning(env, qlearn_estimator)

        # Train the Q-network
        enhanced_q_learning(env, sess, qlearn_estimator, target_estimator, vae=vae)
        # train_vae(vae, normal_data, epochs=10)

        # Test and visualize the results.
        test_and_visualize(qlearn_estimator, env, data_dir_test)
