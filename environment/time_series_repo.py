# environment/time_series_repo.py

import pandas as pd
import numpy as np
import os
import sklearn.preprocessing

# Action Definitions
NOT_ANOMALY = 0
ANOMALY = 1

# Reward Values
REWARD_CORRECT = 1
REWARD_INCORRECT = -1

action_space = [NOT_ANOMALY, ANOMALY]

# Hyperparameters
n_steps = 50  # Number of steps in each sequence

def defaultStateFuc(timeseries, timeseries_curser, previous_state=None, action=None):
    """
    Generate the current state for the RL agent based on the time series.

    Args:
        timeseries (DataFrame): The time series data.
        timeseries_curser (int): Current position in the time series.
        previous_state (np.ndarray or None): Previous state (if any).
        action (int or None): Previous action taken (if any).

    Returns:
        np.ndarray or None: Current state representation or None if insufficient data.
    """
    if timeseries_curser < n_steps:
        return None  # Not enough data to form a state
    elif timeseries_curser == n_steps:
        # Initial state: first n_steps with action flags set to 0
        state = []
        for i in range(timeseries_curser):
            state.append([timeseries['value'][i], 0])  # Action flag set to 0
        state.pop(0)  # Remove the first element to maintain window size
        state.append([timeseries['value'][timeseries_curser], 1])  # Latest data point with action flag
        return np.array(state, dtype='float32')
    else:
        # Generate two possible next states based on the current action
        if previous_state is None:
            return None
        # Action 0: NOT_ANOMALY
        state0 = np.concatenate((previous_state[1:n_steps],
                                 [[timeseries['value'][timeseries_curser], 0]]))
        # Action 1: ANOMALY
        state1 = np.concatenate((previous_state[1:n_steps],
                                 [[timeseries['value'][timeseries_curser], 1]]))
        return np.array([state0, state1], dtype='float32')

def defaultRewardFuc(timeseries, timeseries_curser, action):
    """
    Compute the reward based on the action taken.

    Args:
        timeseries (DataFrame): The time series data.
        timeseries_curser (int): Current position in the time series.
        action (int): Action taken (0 or 1).

    Returns:
        int: Reward value.
    """
    if action == timeseries['anomaly'][timeseries_curser]:
        return REWARD_CORRECT
    else:
        return REWARD_INCORRECT

class EnvTimeSeriesfromRepo():
    """
    Environment class for Time Series Anomaly Detection using Reinforcement Learning.
    """

    def __init__(self, repodir='normal-data/', n_steps=50):
        """
        Initialize the environment with the dataset directory and sequence length.

        Args:
            repodir (str): Path to the dataset directory containing CSV files.
            n_steps (int): Number of steps in each sequence.
        """
        self.repodir = repodir
        self.repodirext = []
        self.n_steps = n_steps

        # Traverse the repository directory to find all CSV files
        for subdir, dirs, files in os.walk(self.repodir):
            for file in files:
                if file.endswith('.csv'):
                    self.repodirext.append(os.path.join(subdir, file))

        self.action_space_n = len(action_space)

        self.timeseries = None
        self.timeseries_curser = -1
        self.timeseries_curser_init = 0
        self.timeseries_states = None

        self.statefnc = defaultStateFuc
        self.rewardfnc = defaultRewardFuc

        self.timeseries_repo = []
        self.states_list = []

        # Load datasets
        for i, filepath in enumerate(self.repodirext):
            try:
                # Read CSV with specified columns
                ts = pd.read_csv(filepath, usecols=[1, 2], header=0, names=['value', 'anomaly'])
                # Add 'label' column initialized to -1
                ts['label'] = -1
                # Convert to float32
                ts = ts.astype(np.float32)
                # Scale 'value' column
                scaler = sklearn.preprocessing.MinMaxScaler()
                scaler.fit(ts['value'].values.reshape(-1, 1))
                ts['value'] = scaler.transform(ts['value'].values.reshape(-1, 1))
                # Check if dataset has enough rows
                if ts.shape[0] < self.n_steps:
                    print(f"Skipping {filepath}: Not enough rows ({ts.shape[0]}) for n_steps={self.n_steps}.")
                    continue
                # Append to repository
                self.timeseries_repo.append(ts)
                print(f"Loaded dataset {i+1}/{len(self.repodirext)}: {filepath} with {ts.shape[0]} rows.")
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        self.datasetsize = len(self.timeseries_repo)
        print(f"Total valid datasets loaded: {self.datasetsize}")

        if self.datasetsize == 0:
            raise ValueError(f"No valid datasets found in directory {self.repodir}. Ensure CSV files have at least {self.n_steps} rows and contain 'value' and 'anomaly' columns.")

    def reset(self):
        """
        Reset the environment to start a new episode.

        Returns:
            np.ndarray: The initial state.
        """
        # Select a new dataset
        if self.datasetsize == 0:
            raise ValueError("No datasets loaded. Cannot reset environment.")
        self.datasetidx = (self.datasetidx + 1) % self.datasetsize if hasattr(self, 'datasetidx') else 0
        self.timeseries = self.timeseries_repo[self.datasetidx]
        self.timeseries_curser = self.timeseries_curser_init

        # Generate the initial state
        self.timeseries_states = self.statefnc(self.timeseries, self.timeseries_curser)
        print(f"Environment reset to dataset index {self.datasetidx}.")

        # Update states list
        self.states_list = self.get_states_list()

        return self.timeseries_states

    def get_states_list(self):
        """
        Process the entire time series into a list of states.

        Returns:
            list: List of state sequences.
        """
        state_list = []
        for cursor in range(self.timeseries_curser_init, self.timeseries['value'].size):
            if cursor < self.n_steps:
                state = self.statefnc(self.timeseries, cursor)
                if state is not None:
                    state_list.append(state)
            else:
                state = self.statefnc(self.timeseries, cursor, state_list[-1], action=None)
                if state is not None:
                    # state is a numpy array with shape (2, n_steps, 2)
                    # Each entry corresponds to action 0 and action 1
                    # Append both possible states
                    state_list.extend(state)
        print(f"Generated {len(state_list)} states from dataset index {self.datasetidx}.")
        return state_list

    def step(self, action):
        """
        Take an action in the environment.

        Args:
            action (int): The action taken by the agent (0 or 1).

        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Get the reward
        reward = self.rewardfnc(self.timeseries, self.timeseries_curser, action)

        # Move to the next time step
        self.timeseries_curser += 1

        # Check if we've reached the end of the time series
        if self.timeseries_curser >= self.timeseries['value'].size:
            done = True
            next_state = None
            print("Reached the end of the time series.")
        else:
            done = False
            # Generate the next state based on the action
            next_state = self.statefnc(self.timeseries, self.timeseries_curser, self.timeseries_states, action)

            if next_state is not None:
                if len(next_state.shape) > len(self.timeseries_states.shape):
                    # Choose the state corresponding to the action
                    self.timeseries_states = next_state[action]
                else:
                    self.timeseries_states = next_state

        return next_state, reward, done, {}
