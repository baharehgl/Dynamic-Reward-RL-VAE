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
from tensorflow.keras import layers, models, losses, optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from collections import deque, namedtuple

tf.compat.v1.disable_eager_execution()

# ------------------------------------------------------------------
# 0) Set up relative paths to data.
current_dir = os.path.dirname(os.path.abspath(__file__))

normal_data_dir = os.path.join(current_dir, "normal-data")
benchmark_dir = os.path.join(
    current_dir, "ydata-labeled-time-series-anomalies-v1_0", "A1Benchmark"
)
# ------------------------------------------------------------------

# Append parent directory so we can import local modules (e.g. environment folder)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from environment.time_series_repo import EnvTimeSeriesfromRepo

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

# Macros for running Q-learning
DATAFIXED = 0  # whether target at a single time series dataset

EPISODES = 500
DISCOUNT_FACTOR = 0.5
EPSILON = 0.5
EPSILON_DECAY = 1.00

NOT_ANOMALY = 0
ANOMALY = 1

action_space = [NOT_ANOMALY, ANOMALY]
action_space_n = len(action_space)

n_steps = 50
n_input_dim = 2
n_hidden_dim = 128

# Reward Values
TP_Value = 5
TN_Value = 1
FP_Value = -1
FN_Value = -5

validation_separate_ratio = 0.9

########################### VAE #####################
def load_normal_data(data_path):
    """Load all CSVs from `data_path`, concatenate, and scale."""
    all_files = [
        os.path.join(data_path, fname)
        for fname in os.listdir(data_path)
        if fname.endswith('.csv')
    ]
    data_list = [pd.read_csv(file) for file in all_files]
    data = pd.concat(data_list, axis=0, ignore_index=True)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.values)
    return scaled_data

x_train = load_normal_data(normal_data_dir)

class Sampling(layers.Layer):
    """Sampling layer for VAE."""
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

    decoder_h = layers.Dense(intermediate_dim, activation='relu')
    decoder_h = layers.Dense(intermediate_dim, activation='relu')
    decoder_h = layers.Dense(intermediate_dim, activation='relu')
    decoder_mean = layers.Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    encoder = models.Model(inputs, [z_mean, z_log_var, z])
    vae = models.Model(inputs, x_decoded_mean)

    reconstruction_loss = losses.binary_crossentropy(inputs, x_decoded_mean) * original_dim
    kl_loss = -0.5 * tf.reduce_sum(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
    )
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return vae, encoder


original_dim = 3
latent_dim = 10
intermediate_dim = 64

vae, encoder = build_vae(original_dim, latent_dim, intermediate_dim)

###################### State / Reward Functions #####################
def RNNBinaryStateFuc(timeseries, timeseries_curser, previous_state=[], action=None):
    """
    Returns a vector (or 2 possible states) for our RNN.
    """
    if timeseries_curser == n_steps:
        state = []
        for i in range(timeseries_curser):
            state.append([timeseries['value'][i], 0])
        state.pop(0)
        state.append([timeseries['value'][timeseries_curser], 1])
        return np.array(state, dtype='float32')

    if timeseries_curser > n_steps:
        state0 = np.concatenate((
            previous_state[1:n_steps],
            [[timeseries['value'][timeseries_curser], 0]]
        ))
        state1 = np.concatenate((
            previous_state[1:n_steps],
            [[timeseries['value'][timeseries_curser], 1]]
        ))
        return np.array([state0, state1], dtype='float32')


tau_values = []

def kl_divergence(p, q):
    """Compute KL(p || q)."""
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))

def adaptive_scaling_factor_dro(preference_strength, tau_min=0.1, tau_max=5.0, rho=0.5, max_iter=10):
    """
    Learn tau dynamically using a KL-constrained approach (DRO).
    """
    tau = 1.0
    for _ in range(max_iter):
        kl_term = kl_divergence([preference_strength, 1 - preference_strength], [0.5, 0.5])
        grad = -np.log(1 + np.exp(-preference_strength / tau)) + rho - kl_term
        hess = (preference_strength**2 * np.exp(-preference_strength / tau) /
                (tau**3 * (1 + np.exp(-preference_strength / tau))**2))
        tau = tau - grad / (hess + 1e-8)
        tau = np.clip(tau, tau_min, tau_max)
    return tau

def RNNBinaryRewardFuc(timeseries, timeseries_curser, action=0, vae=None, scale_factor=10):
    """
    Reward function that includes a penalty from VAE reconstruction error
    and an adaptive scaling factor (tau).
    """
    if timeseries_curser >= n_steps:
        current_state = np.array([timeseries['value'][timeseries_curser - n_steps:timeseries_curser]])
        vae_reconstruction = vae.predict(current_state)
        reconstruction_error = np.mean(np.square(vae_reconstruction - current_state))

        vae_penalty = -scale_factor * reconstruction_error
        preference_strength = np.clip(
            1 / (1 + np.exp(-reconstruction_error)), 0.05, 0.95
        )

        tau = adaptive_scaling_factor_dro(preference_strength)
        tau_values.append(tau)

        if timeseries['label'][timeseries_curser] == 0:
            return [tau * (TN_Value + vae_penalty),
                    tau * (FP_Value + vae_penalty)]
        if timeseries['label'][timeseries_curser] == 1:
            return [tau * (FN_Value + vae_penalty),
                    tau * (TP_Value + vae_penalty)]
    else:
        return [0, 0]

def RNNBinaryRewardFucTest(timeseries, timeseries_curser, action=0):
    """
    Testing reward function using 'anomaly' column to see if action
    was correct or not (for evaluation).
    """
    if timeseries_curser >= n_steps:
        if timeseries['anomaly'][timeseries_curser] == 0:
            return [TN_Value, FP_Value]
        if timeseries['anomaly'][timeseries_curser] == 1:
            return [FN_Value, TP_Value]
    else:
        return [0, 0]


###################### Q Estimator (RNN) #####################
class Q_Estimator_Nonlinear():
    """
    Action-Value Function Approximator Q(s,a) using a TensorFlow RNN.
    """
    def __init__(self, learning_rate=np.float32(0.01), scope="Q_Estimator_Nonlinear",
                 summaries_dir=None, global_step=None):
        """
        Pass in a global_step variable (tf.Variable) so that the optimizer
        can increment it with each training step.
        """
        self.scope = scope
        self.summary_writer = None
        self.global_step = global_step

        with tf.compat.v1.variable_scope(scope):
            self.state = tf.compat.v1.placeholder(
                shape=[None, n_steps, n_input_dim],
                dtype=tf.float32,
                name="state"
            )
            self.target = tf.compat.v1.placeholder(
                shape=[None, action_space_n],
                dtype=tf.float32,
                name="target"
            )

            self.weights = {
                'out': tf.Variable(tf.compat.v1.random_normal([n_hidden_dim, action_space_n]))
            }
            self.biases = {
                'out': tf.Variable(tf.compat.v1.random_normal([action_space_n]))
            }

            self.state_unstack = tf.unstack(self.state, n_steps, 1)

            # LSTMCell (TF1 style). Deprecation warning is normal in TF2.
            lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(n_hidden_dim, forget_bias=1.0)

            self.outputs, self.states = tf.compat.v1.nn.static_rnn(
                lstm_cell, self.state_unstack, dtype=tf.float32
            )

            self.action_values = (
                tf.matmul(self.outputs[-1], self.weights['out']) + self.biases['out']
            )

            self.losses = tf.compat.v1.squared_difference(self.action_values, self.target)
            self.loss = tf.reduce_mean(self.losses)

            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

            # IMPORTANT: Use 'global_step=self.global_step' so it increments each update
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

            self.summaries = tf.compat.v1.summary.merge([
                tf.compat.v1.summary.histogram("loss_hist", self.losses),
                tf.compat.v1.summary.scalar("loss", self.loss),
                tf.compat.v1.summary.histogram("q_values_hist", self.action_values),
                tf.compat.v1.summary.scalar("q_value", tf.reduce_max(self.action_values))
            ])

            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, f"summaries_{scope}")
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.compat.v1.summary.FileWriter(summary_dir)

    def predict(self, state, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        return sess.run(self.action_values, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        feed_dict = {self.state: state, self.target: target}
        summaries, _, _ = sess.run(
            [self.summaries, self.global_step, self.train_op],
            feed_dict
        )
        if self.summary_writer:
            # We can retrieve the current global step to write summaries
            current_gstep = sess.run(self.global_step)
            self.summary_writer.add_summary(summaries, current_gstep)
        return


def copy_model_parameters(sess, estimator1, estimator2):
    """
    Copies the model parameters of one estimator to another.
    """
    e1_params = [
        t for t in tf.compat.v1.trainable_variables()
        if t.name.startswith(estimator1.scope)
    ]
    e1_params = sorted(e1_params, key=lambda v: v.name)

    e2_params = [
        t for t in tf.compat.v1.trainable_variables()
        if t.name.startswith(estimator2.scope)
    ]
    e2_params = sorted(e2_params, key=lambda v: v.name)

    update_ops = []
    for e1_v, e2_v in zip(e1_params, e2_params):
        op = e2_v.assign(e1_v)
        update_ops.append(op)

    sess.run(update_ops)


def make_epsilon_greedy_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on Q-function approximator and epsilon.
    """
    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype='float32') * epsilon / nA
        q_values = estimator.predict(state=[observation])
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


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
               vae_model=None,
               global_step=None):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    """
    Transition = namedtuple("Transition", ["state", "reward", "next_state", "done"])
    replay_memory = []

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver = tf.compat.v1.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print(f"Loading model checkpoint {latest_checkpoint} ...")
        saver.restore(sess, latest_checkpoint)
        # If test=1, you might just exit after loading
        if test:
            return

    # Instead of get_global_step, we do:
    total_t = sess.run(global_step)  # current value of the global_step

    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    policy = make_epsilon_greedy_policy(qlearn_estimator, env.action_space_n)

    num_label = 0

    # -------------------------------------------------------------------
    # Warm up the replay memory with initial experience
    popu_time = time.time()
    print('Warm up starting...')

    outliers_fraction = 0.01
    data_train = []
    for num in range(env.datasetsize):
        env.reset()
        data_train.extend(env.states_list)

    model = WarmUp().warm_up_isolation_forest(outliers_fraction, data_train)
    lp_model = LabelSpreading()

    for t in itertools.count():
        env.reset()
        data = np.array(env.states_list).transpose(2, 0, 1).reshape(2, -1)[0].reshape(-1, n_steps)[:, -1].reshape(-1, 1)
        anomaly_score = model.decision_function(data)
        pred_score = [-1 * s + 0.5 for s in anomaly_score]
        warm_samples = np.argsort(pred_score)[:5]
        warm_samples = np.append(warm_samples, np.argsort(pred_score)[-5:])

        state_list = np.array(env.states_list).transpose(2, 0, 1)[0]
        label_list = [-1] * len(state_list)

        # Label the warm_samples
        for sample in warm_samples:
            state = env.states_list[sample]
            env.timeseries_curser = sample + n_steps
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
            action_probs = policy(state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            env.timeseries['label'][env.timeseries_curser] = env.timeseries['anomaly'][env.timeseries_curser]
            num_label += 1

            next_state, reward, done, _ = env.step(action)
            replay_memory.append(Transition(state, reward, next_state, done))
            label_list[sample] = int(env.timeseries['anomaly'][env.timeseries_curser])

        unlabeled_indices = [i for i, e in enumerate(label_list) if e == -1]
        label_list = np.array(label_list)
        lp_model.fit(state_list, label_list)
        pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)
        certainty_index = np.argsort(pred_entropies)
        certainty_index = certainty_index[np.in1d(certainty_index, unlabeled_indices)][:num_LabelPropagation]
        for index in certainty_index:
            pseudo_label = lp_model.transduction_[index]
            env.timeseries['label'][index + n_steps] = pseudo_label

        if len(replay_memory) >= replay_memory_init_size:
            break

    popu_time = time.time() - popu_time
    print(f"Populating replay memory took {popu_time:.2f} seconds.")

    # -------------------------------------------------------------------
    # Main training loop
    dict_labeled = {}
    for i_episode in range(num_episodes):
        if i_episode % 50 == 49:
            print(f"Save checkpoint in episode {i_episode + 1}/{num_episodes}")
            saver.save(sess, checkpoint_path)

        per_loop_time1 = time.time()
        state = env.reset()
        while env.datasetidx > env.datasetrng * validation_separate_ratio:
            env.reset()
            print('double reset')

        # Active learning
        labeled_index = [i for i, e in enumerate(env.timeseries['label']) if e != -1]
        labeled_index = [item for item in labeled_index if item >= n_steps]
        labeled_index = [item - n_steps for item in labeled_index]

        al = active_learning(env=env,
                             N=num_active_learning,
                             strategy='margin_sampling',
                             estimator=qlearn_estimator,
                             already_selected=labeled_index)
        al_samples = al.get_samples()
        labeled_index.extend(al_samples)
        num_label += len(al_samples)

        state_list = np.array(env.states_list).transpose(2, 0, 1)[0]
        label_list = np.array(env.timeseries['label'][n_steps:])

        for new_sample in al_samples:
            label_list[new_sample] = env.timeseries['anomaly'][n_steps + new_sample]
            env.timeseries['label'][n_steps + new_sample] = env.timeseries['anomaly'][n_steps + new_sample]

        for samples in labeled_index:
            env.timeseries_curser = samples + n_steps
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]
            state = env.states_list[samples]

            action_probs = policy(state, epsilon)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)

            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)
            replay_memory.append(Transition(state, reward, next_state, done))

        unlabeled_indices = [i for i, e in enumerate(label_list) if e == -1]
        label_list = np.array(label_list)
        lp_model.fit(state_list, label_list)
        pred_entropies = stats.distributions.entropy(lp_model.label_distributions_.T)
        certainty_index = np.argsort(pred_entropies)
        certainty_index = certainty_index[np.in1d(certainty_index, unlabeled_indices)][:num_LabelPropagation]
        for index in certainty_index:
            pseudo_label = lp_model.transduction_[index]
            env.timeseries['label'][index + n_steps] = pseudo_label

        per_loop_time2 = time.time()

        # Update the model
        for i_epoch in range(num_epoches):
            # Optionally add a summary for epsilon, etc.
            if qlearn_estimator.summary_writer:
                episode_summary = tf.compat.v1.Summary()
                # Add your custom scalars if needed
                current_gstep = sess.run(global_step)
                qlearn_estimator.summary_writer.add_summary(episode_summary, current_gstep)

            # Update target estimator if needed
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, qlearn_estimator, target_estimator)
                print("\nCopied model parameters to target network.\n")

            samples = random.sample(replay_memory, batch_size)
            states_batch, reward_batch, next_states_batch, done_batch = map(np.array, zip(*samples))

            if discount_factor > 0:
                # next_states_batch has shape [batch_size, 2, n_steps, n_input_dim]
                next_states_batch = np.squeeze(np.split(next_states_batch, 2, axis=1))
                next_states_batch0 = next_states_batch[0]
                next_states_batch1 = next_states_batch[1]

                q_values_next0 = target_estimator.predict(state=next_states_batch0)
                q_values_next1 = target_estimator.predict(state=next_states_batch1)

                targets_batch = reward_batch + (
                    discount_factor * np.stack(
                        (np.amax(q_values_next0, axis=1),
                         np.amax(q_values_next1, axis=1)),
                        axis=-1
                    )
                )
            else:
                targets_batch = reward_batch

            qlearn_estimator.update(
                state=states_batch,
                target=targets_batch.astype(np.float32),
                sess=sess
            )

            # Re-fetch the global_step to keep track in Python
            total_t = sess.run(global_step)

        per_loop_time_popu = per_loop_time2 - per_loop_time1
        per_loop_time_updt = time.time() - per_loop_time2
        print(f"Global step {total_t} @ Episode {i_episode+1}/{num_episodes},"
              f" time: {per_loop_time_popu:.2f} + {per_loop_time_updt:.2f}")

    return


def evaluate_model(y_true, y_pred):
    """
    Compute classification metrics and return as a dictionary.
    """
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

def q_learning_validator(env, estimator, num_episodes, record_dir=None, plot=1):
    """
    Validate the trained model using multiple evaluation metrics.
    """
    y_true_all = []
    y_pred_all = []

    for i_episode in range(num_episodes):
        policy = make_epsilon_greedy_policy(estimator, env.action_space_n)
        state = env.reset()
        while env.datasetidx < env.datasetrng * validation_separate_ratio:
            state = env.reset()

        for t in itertools.count():
            action_probs = policy(state, 0)  # greedy
            action = np.argmax(action_probs)
            y_true_all.append(env.timeseries['anomaly'][env.timeseries_curser])
            y_pred_all.append(action)
            next_state, reward, done, _ = env.step(action)
            if done:
                break
            state = next_state[action]

    results = evaluate_model(y_true_all, y_pred_all)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    return results

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
            sort_distances = np.sort(distances, 1)[:, -2:]
            min_margin = sort_distances[:, 1] - sort_distances[:, 0]
        rank_ind = np.argsort(min_margin)
        rank_ind = [i for i in rank_ind if i not in self.already_selected]
        active_samples = rank_ind[0:self.N]
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
            sort_distances = np.sort(distances, 1)[:, -2:]
            min_margin = sort_distances[:, 1] - sort_distances[:, 0]
        rank_ind = np.argsort(min_margin)
        rank_ind = [i for i in rank_ind if i not in self.already_selected]
        active_samples = [t for t in rank_ind if distances[t] < threshold]
        return active_samples

    def label(self, active_samples):
        # Example of interactive labeling in terminal (if needed).
        for sample in active_samples:
            print('Please label the last timestamp (0 for normal, 1 for anomaly):')
            label = input()
            self.env.timeseries.loc[sample + n_steps - 1, 'anomaly'] = label
        return


class WarmUp(object):
    """
    Used to 'warm up' your replay memory with outlier-based detection
    (OneClassSVM, IsolationForest, etc.)
    """
    def warm_up_SVM(self, outliers_fraction, N):
        states_list = self.env.get_states_list()
        data = (
            np.array(states_list)
            .transpose(2, 0, 1)
            .reshape(2, -1)[0]
            .reshape(-1, n_steps)[:, -1]
            .reshape(-1, 1)
        )
        model = OneClassSVM(gamma='auto', nu=0.95 * outliers_fraction)
        model.fit(data)
        distances = model.decision_function(data)
        if len(distances.shape) < 2:
            min_margin = abs(distances)
        else:
            sort_distances = np.sort(distances, 1)[:, -2:]
            min_margin = sort_distances[:, 1] - sort_distances[:, 0]
        rank_ind = np.argsort(min_margin)
        samples = rank_ind[0:N]
        return samples

    def warm_up_isolation_forest(self, outliers_fraction, X_train):
        from sklearn.ensemble import IsolationForest
        data = (
            np.array(X_train)
            .transpose(2, 0, 1)
            .reshape(2, -1)[0]
            .reshape(-1, n_steps)[:, -1]
            .reshape(-1, 1)
        )
        clf = IsolationForest(contamination=outliers_fraction)
        clf.fit(data)
        return clf


def train(num_LP, num_AL, discount_factor, learn_tau=True):
    """
    Train the RL agent. Also shows how we create an integer `global_step`.
    """
    # If you want to train a VAE from scratch for your environment:
    vae, _ = build_vae(original_dim, latent_dim, intermediate_dim)
    vae.fit(x_train, epochs=50, batch_size=32)
    vae.save('vae_model.h5')

    # Or load a pretrained model:
    # from tensorflow.keras.models import load_model
    # vae = load_model('vae_model.h5', custom_objects={'Sampling': Sampling}, compile=False)

    exp_relative_dir = ['RLVAL_with_DRO_and_Adaptive_Scaling']
    dataset_dir = [benchmark_dir]  # relative path to your anomalies

    for i in range(len(dataset_dir)):
        env = EnvTimeSeriesfromRepo(dataset_dir[i])
        env.statefnc = RNNBinaryStateFuc
        if learn_tau:
            env.rewardfnc = lambda ts, cur, a: RNNBinaryRewardFuc(ts, cur, a, vae)
        else:
            env.rewardfnc = RNNBinaryRewardFuc
        env.timeseries_curser_init = n_steps
        env.datasetfix = DATAFIXED
        env.datasetidx = 0

        env_test = env
        env_test.rewardfnc = RNNBinaryRewardFucTest

        experiment_dir = os.path.abspath(
            "./exp/{}".format(exp_relative_dir[i])
        )

        tf.compat.v1.reset_default_graph()

        # Create a global_step variable with int type
        global_step = tf.compat.v1.get_variable(
            name="global_step",
            shape=[],
            dtype=tf.int32,
            initializer=tf.zeros_initializer(),
            trainable=False
        )

        qlearn_estimator = Q_Estimator_Nonlinear(
            scope="qlearn",
            summaries_dir=experiment_dir,
            learning_rate=0.0003,
            global_step=global_step
        )
        target_estimator = Q_Estimator_Nonlinear(
            scope="target",
            global_step=global_step
        )

        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())

        with sess.as_default():
            q_learning(
                env,
                sess=sess,
                qlearn_estimator=qlearn_estimator,
                target_estimator=target_estimator,
                num_episodes=300,
                num_epoches=10,
                experiment_dir=experiment_dir,
                replay_memory_size=500000,
                replay_memory_init_size=1500,
                update_target_estimator_every=10,
                epsilon_start=1.0,
                epsilon_end=0.1,
                epsilon_decay_steps=500000,
                discount_factor=discount_factor,
                batch_size=256,
                num_LabelPropagation=num_LP,
                num_active_learning=num_AL,
                test=0,
                vae_model=vae,
                global_step=global_step
            )
            # Evaluate
            optimization_metric = q_learning_validator(
                env_test,
                qlearn_estimator,
                int(env.datasetsize * (1 - validation_separate_ratio)),
                experiment_dir
            )
        return optimization_metric


def plot_tau_evolution():
    plt.figure(figsize=(10, 5))
    plt.plot(tau_values, label="Tau over time", alpha=0.8)
    plt.xlabel("Training Steps")
    plt.ylabel("Tau Value")
    plt.title("Evolution of Tau during Training")
    plt.legend()
    plt.show()



plot_tau_evolution()
train(100, 30, 0.92, learn_tau=True)
train(150, 50, 0.94, learn_tau=True)
train(200, 100, 0.96, learn_tau=True)
