# train_wadi.py
import os
import time
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from collections import namedtuple
from tensorflow.keras import layers, models, losses
from tensorflow.keras.models import load_model
from env_wadi import EnvTimeSeriesWaDi

# Disable eager execution for TF1.x style
tf.compat.v1.disable_eager_execution()

# Hyperparameters & globals
EPISODES       = 3
n_steps        = 25
n_input_dim    = 2
n_hidden_dim   = 128
DISCOUNT_FACTOR= 0.5

# Extrinsic reward values
TN_Value = 1
TP_Value = 10
FP_Value = -1
FN_Value = -10

action_space_n = 2

# =============================
# VAE Setup
# =============================
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim   = tf.shape(z_mean)[1]
        eps   = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * eps


def build_vae(original_dim, latent_dim=10, intermediate_dim=64):
    inputs = layers.Input(shape=(original_dim,))
    h = layers.Dense(intermediate_dim, activation='relu')(inputs)
    h = layers.Dense(intermediate_dim, activation='relu')(h)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
    z = Sampling()([z_mean, z_log_var])

    decoder_h    = layers.Dense(intermediate_dim, activation='relu')
    decoder_mean = layers.Dense(original_dim, activation='sigmoid')
    h_decoded    = decoder_h(z)
    outputs      = decoder_mean(h_decoded)

    vae = models.Model(inputs, outputs)
    recon_loss = losses.mse(inputs, outputs) * original_dim
    kl_loss    = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    vae.add_loss(tf.reduce_mean(recon_loss + kl_loss))
    vae.compile(optimizer='adam')
    encoder = models.Model(inputs, z_mean)
    return vae, encoder

# =============================
# State & Reward Functions
# =============================
def RNNBinaryStateFuc(timeseries, timeseries_curser, previous_state=None, action=None):
    if timeseries_curser == n_steps:
        state = [[timeseries['value'].iat[i], 0] for i in range(n_steps)]
        state.pop(0)
        state.append([timeseries['value'].iat[n_steps], 1])
        return np.array(state, dtype='float32')
    if timeseries_curser > n_steps:
        prev = previous_state
        s0 = np.concatenate((prev[1:], [[timeseries['value'].iat[timeseries_curser], 0]]))
        s1 = np.concatenate((prev[1:], [[timeseries['value'].iat[timeseries_curser], 1]]))
        return np.array([s0, s1], dtype='float32')
    return None


def RNNBinaryRewardFuc(timeseries, timeseries_curser, action, vae=None,
                       dynamic_coef=1.0, include_vae_penalty=True):
    if timeseries_curser < n_steps:
        return [0,0]
    vae_penalty = 0.0
    if include_vae_penalty and vae is not None:
        window = timeseries['value'].values[timeseries_curser-n_steps:timeseries_curser].reshape(1, -1)
        recon  = vae.predict(window)
        err    = np.mean((recon - window)**2)
        vae_penalty = dynamic_coef * err
    lbl = timeseries['label'].iat[timeseries_curser]
    if lbl == 0:
        return [TN_Value + vae_penalty, FP_Value + vae_penalty]
    else:
        return [FN_Value + vae_penalty, TP_Value + vae_penalty]


def RNNBinaryRewardFucTest(timeseries, timeseries_curser, action):
    if timeseries_curser < n_steps:
        return [0,0]
    lbl = timeseries['anomaly'].iat[timeseries_curser]
    if lbl == 0:
        return [TN_Value, FP_Value]
    else:
        return [FN_Value, TP_Value]

# =============================
# Q-value Estimator
# =============================
class Q_Estimator_Nonlinear:
    def __init__(self, learning_rate=np.float32(0.01), scope="Q_Estimator", summaries_dir=None):
        self.scope = scope
        with tf.compat.v1.variable_scope(scope):
            self.state  = tf.compat.v1.placeholder(shape=[None, n_steps, n_input_dim], dtype=tf.float32)
            self.target = tf.compat.v1.placeholder(shape=[None, action_space_n], dtype=tf.float32)

            self.weights = {'out': tf.Variable(tf.random.normal([n_hidden_dim, action_space_n]))}
            self.biases  = {'out': tf.Variable(tf.random.normal([action_space_n]))}

            state_unstack = tf.compat.v1.unstack(self.state, n_steps, axis=1)
            lstm_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(n_hidden_dim)
            outputs, _ = tf.compat.v1.nn.static_rnn(lstm_cell, state_unstack, dtype=tf.float32)
            self.action_values = tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']

            self.losses = tf.compat.v1.square(self.action_values - self.target)
            self.loss   = tf.reduce_mean(self.losses)
            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
            global_step_var = tf.compat.v1.get_variable("global_step", shape=[], trainable=False,
                                                       initializer=tf.zeros_initializer())
            self.train_op = self.optimizer.minimize(self.loss, global_step=global_step_var)

    def predict(self, state, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        return sess.run(self.action_values, {self.state: state})

    def update(self, state, target, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        sess.run(self.train_op, {self.state: state, self.target: target})



# =============================
# Helpers
# =============================
def copy_model_parameters(sess, estimator1, estimator2):
    e1_params = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith(estimator1.scope)]
    e2_params = [v for v in tf.compat.v1.trainable_variables() if v.name.startswith(estimator2.scope)]
    for v1, v2 in zip(sorted(e1_params, key=lambda x: x.name),
                      sorted(e2_params, key=lambda x: x.name)):
        sess.run(v2.assign(v1))


def make_epsilon_greedy_policy(estimator, nA, sess):
    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype='float32') * epsilon / nA
        q_vals = estimator.predict([observation], sess)[0]
        best = np.argmax(q_vals)
        A[best] += (1.0 - epsilon)
        return A
    return policy_fn


def update_dynamic_coef_proportional(current_coef, episode_reward,
                                     target_reward=0.0, alpha=0.001,
                                     min_coef=0.1, max_coef=10.0):
    new_coef = current_coef + alpha*(target_reward - episode_reward)
    return max(min(new_coef, max_coef), min_coef)

# =============================
# Active Learning
# =============================
class active_learning:
    def __init__(self, env, N, strategy, estimator, already_selected):
        self.env = env
        self.N = N
        self.strategy = strategy
        self.estimator = estimator
        self.already_selected = already_selected

    def get_samples(self):
        distances = []
        for state in self.env.states_list:
            q = self.estimator.predict([state])[0]
            distances.append(abs(q[0] - q[1]))
        rank_ind = np.argsort(distances)
        rank_ind = [i for i in rank_ind if i not in self.already_selected]
        return rank_ind[:self.N]

    def get_samples_by_score(self, threshold):
        distances = []
        for state in self.env.states_list:
            q = self.estimator.predict([state])[0]
            distances.append(abs(q[0] - q[1]))
        rank_ind = np.argsort(distances)
        return [i for i in rank_ind if distances[i] < threshold and i not in self.already_selected]

    def label(self, active_samples):
        for sample in active_samples:
            print("Active sample:", sample)
            label = float(input("Label last timestamp (0 normal, 1 anomaly): "))
            self.env.timeseries.loc[sample + n_steps - 1, 'label'] = label

# =============================
# WarmUp
# =============================
class WarmUp:
    def warm_up_isolation_forest(self, outliers_fraction, data_train):
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(contamination=outliers_fraction)
        clf.fit(data_train)
        return clf

# =============================
# Q-Learning
# =============================
def q_learning(env, sess, qlearn_estimator, target_estimator, num_episodes, num_epoches,
               replay_memory_size=500000, replay_memory_init_size=50000,
               experiment_dir='./exp/WaDi/', update_target_estimator_every=10000,
               epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=500000,
               batch_size=256, num_LabelPropagation=20, num_active_learning=10,
               test=0, vae_model=None, include_vae_penalty=True):
    Transition = namedtuple("Transition", ["state", "reward", "next_state", "done"])
    replay_memory = []
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    os.makedirs(checkpoint_dir, exist_ok=True)
    saver = tf.compat.v1.train.Saver()
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    if latest:
        saver.restore(sess, checkpoint_path)
        if test:
            return [], []

    global_step_var = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="global_step")[0]
    total_t = sess.run(global_step_var)
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
    policy = make_epsilon_greedy_policy(qlearn_estimator, action_space_n, sess)

    # Warm up
    data_train = []
    for s in env.states_list:
        data_train.append(s[-1][0])
        if len(data_train) >= replay_memory_init_size:
            break
    data_train = np.array(data_train).reshape(-1,1)
    model_warm = WarmUp().warm_up_isolation_forest(0.01, data_train)
    lp_model = LabelSpreading()

    # Initialize labels via Active Learning & LP
    env_states = env.states_list
    for sample in list(model_warm.predict(data_train).argsort()[:5]):
        env.timeseries.loc[sample + n_steps, 'label'] = env.timeseries['anomaly'].iat[sample + n_steps]

    # Main Q-learning loop
    dynamic_coef = 20.0
    episode_rewards = []
    coef_history = []
    for i in range(num_episodes):
        env.rewardfnc = lambda ts, tc, a: RNNBinaryRewardFuc(ts, tc, a,
                               vae_model, dynamic_coef, include_vae_penalty)
        episode_reward = 0.0
        for t in itertools.count():
            state = env.reset() if t==0 else next_state[action]
            action_probs = policy(state, epsilons[min(total_t, epsilon_decay_steps-1)])
            action = np.random.choice(np.arange(action_space_n), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward[action]
            replay_memory.append(Transition(state, reward, next_state, done))
            if len(replay_memory) > replay_memory_size:
                replay_memory.pop(0)
            if done:
                break
            total_t += 1
            if total_t % update_target_estimator_every == 0:
                copy_model_parameters(sess, qlearn_estimator, target_estimator)
        # Train for num_epoches
        for _ in range(num_epoches):
            batch = random.sample(replay_memory, batch_size)
            states_b, rewards_b, next_b, dones_b = map(np.array, zip(*batch))
            q_next0 = target_estimator.predict(next_b[:,0], sess)
            q_next1 = target_estimator.predict(next_b[:,1], sess)
            targets = rewards_b + DISCOUNT_FACTOR * np.stack((np.max(q_next0,1),
                                                             np.max(q_next1,1)),1)
            qlearn_estimator.update(states_b, targets, sess)
        # Update dynamic coef
        dynamic_coef = update_dynamic_coef_proportional(dynamic_coef, episode_reward)
        episode_rewards.append(episode_reward)
        coef_history.append(dynamic_coef)
        # Save
        saver.save(sess, checkpoint_path)

    return episode_rewards, coef_history

# =============================
# Q-Learning Validator
# =============================
def q_learning_validator(env, estimator, num_episodes, record_dir=None, plot=1):
    precision_all, recall_all, f1_all, aupr_all = [], [], [], []
    for i in range(num_episodes):
        policy = make_epsilon_greedy_policy(estimator, action_space_n, tf.compat.v1.get_default_session())
        state = env.reset()
        preds, gts, ts_vals = [], [], []
        for t in itertools.count():
            action = np.argmax(policy(state,0))
            preds.append(action)
            gt = env.timeseries['anomaly'].iat[env.timeseries_curser]
            gts.append(gt)
            ts_vals.append(state[-1][0] if isinstance(state,np.ndarray) else state)
            state, _, done, _ = env.step(action)
            if done: break
        preds = np.array(preds)
        gts   = np.array(gts)
        precision, recall, f1, _ = precision_recall_fscore_support(gts, preds, average='binary')
        aupr = average_precision_score(gts, preds)
        precision_all.append(precision)
        recall_all.append(recall)
        f1_all.append(f1)
        aupr_all.append(aupr)
        if plot and record_dir:
            p_dir = os.path.join(record_dir, f"val_ep_{i+1}.png")
            fig, axes = plt.subplots(4,1,sharex=True)
            axes[0].plot(ts_vals); axes[0].set_title('TS')
            axes[1].plot(preds); axes[1].set_title('Pred')
            axes[2].plot(gts);   axes[2].set_title('GT')
            axes[3].plot([aupr]*len(ts_vals)); axes[3].set_title('AUPR')
            plt.savefig(p_dir); plt.close(fig)
    return np.mean(f1_all), np.mean(aupr_all)

# =============================
# Plotting
# =============================
def save_plots(experiment_dir, episode_rewards, coef_history):
    os.makedirs(experiment_dir, exist_ok=True)
    plt.figure(); plt.plot(episode_rewards); plt.title('Reward'); plt.savefig(os.path.join(experiment_dir,'reward.png')); plt.close()
    plt.figure(); plt.plot(coef_history); plt.title('Coef'); plt.savefig(os.path.join(experiment_dir,'coef.png')); plt.close()

# =============================
# Training Wrapper
# =============================
def train_wrapper(num_LP, num_AL, discount_factor):
    # 1) VAE training on WaDi normal windows
    df = pd.read_csv("WaDi/WADI_14days_new.csv")
    vals = df['TOTAL_CONS_REQUIRED_FLOW'].values.astype(float)
    X = np.array([vals[i:i+n_steps] for i in range(len(vals)-n_steps)])
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    vae, _ = build_vae(n_steps, latent_dim=10, intermediate_dim=64)
    vae.fit(Xs, epochs=2, batch_size=32)
    vae.save('vae_wadi.h5')

    # 2) Env instantiation
    env = EnvTimeSeriesWaDi(
        sensor_csv="WaDi/WADI_14days_new.csv",
        label_csv ="WaDi/WADI_attackdataLABLE.csv",
        n_steps   =n_steps
    )
    env.statefnc  = RNNBinaryStateFuc
    env.rewardfnc = lambda ts, tc, a: RNNBinaryRewardFuc(ts, tc, a, vae, dynamic_coef=20.0, include_vae_penalty=True)

    # 3) TF & estimators
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()
    from tensorflow.compat.v1.keras import backend as K; K.set_session(sess)
    global_step = tf.Variable(0, name="global_step", trainable=False)
    qlearn = Q_Estimator_Nonlinear(scope="qlearn", learning_rate=3e-4)
    target = Q_Estimator_Nonlinear(scope="target")
    sess.run(tf.compat.v1.global_variables_initializer())

    # 4) Training & Validation
    exp_dir = os.path.abspath(f"./exp/WaDi_LP{num_LP}_AL{num_AL}_d{discount_factor}")
    rewards, coefs = q_learning(env, sess, qlearn, target,
                                num_episodes=EPISODES, num_epoches=10,
                                experiment_dir=exp_dir,
                                update_target_estimator_every=1000,
                                epsilon_start=1.0, epsilon_end=0.1,
                                epsilon_decay_steps=50000,
                                batch_size=256,
                                num_LabelPropagation=num_LP,
                                num_active_learning=num_AL,
                                test=0,
                                vae_model=vae,
                                include_vae_penalty=True)

    final_f1, final_aupr = q_learning_validator(env, qlearn,
                                                num_episodes=1,
                                                record_dir=exp_dir)
    save_plots(exp_dir, rewards, coefs)
    print(f"Results LP={num_LP} AL={num_AL}: F1={final_f1:.4f}, AUPR={final_aupr:.4f}")

if __name__ == "__main__":
    train_wrapper(200, 1000, 0.96)
    train_wrapper(200, 5000, 0.96)
    train_wrapper(200,10000,0.96)
