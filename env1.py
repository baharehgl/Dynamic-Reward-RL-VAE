import os
import random
import pandas as pd
import numpy as np
import sklearn.preprocessing

# Define constants.
NOT_ANOMALY = 0
ANOMALY = 1

REWARD_CORRECT = 1
REWARD_INCORRECT = -1

action_space = [NOT_ANOMALY, ANOMALY]


def defaultStateFuc(timeseries, timeseries_curser, previous_state=None, action=None):
    """
    Default state function: returns the value at the current time index.
    In your RL setup, you can replace this with a sliding-window state function.
    """
    return timeseries['value'][timeseries_curser]


def defaultRewardFuc(timeseries, timeseries_curser, action):
    """
    Default reward: returns REWARD_CORRECT if the action matches the anomaly label,
    and REWARD_INCORRECT otherwise.
    """
    if action == timeseries['anomaly'][timeseries_curser]:
        return REWARD_CORRECT
    else:
        return REWARD_INCORRECT


class EnvTimeSeriesfromRepo():
    def __init__(self, repodir='environment/time_series_repo/'):
        # Get all CSV file paths in the repository directory.
        self.repodir = repodir
        self.repodirext = []
        for subdir, dirs, files in os.walk(self.repodir):
            for file in files:
                if file.endswith('.csv'):
                    self.repodirext.append(os.path.join(subdir, file))

        # Check that CSV files were found.
        if len(self.repodirext) == 0:
            raise ValueError("No CSV files found in directory: {}".format(self.repodir))

        self.action_space_n = len(action_space)

        # Initialize variables.
        self.timeseries = None
        self.timeseries_curser = -1
        self.timeseries_curser_init = 0
        self.timeseries_states = None
        self.statefnc = defaultStateFuc
        self.rewardfnc = defaultRewardFuc

        self.timeseries_repo = []
        self.states_list = []

        # Process each CSV file.
        for i in range(len(self.repodirext)):
            fname = os.path.basename(self.repodirext[i])
            try:
                if "phase2_train" in fname:
                    # For KPI training file: assume it has two (or more) columns (e.g. an id, a value, etc.)
                    ts = pd.read_csv(self.repodirext[i], encoding='latin1')
                    # If there's no 'value' column, assume the second column is the value.
                    if 'value' not in ts.columns:
                        cols = list(ts.columns)
                        if len(cols) >= 2:
                            ts.rename(columns={cols[1]: 'value'}, inplace=True)
                        else:
                            raise ValueError("File {} does not contain enough columns.".format(self.repodirext[i]))
                    # Convert the 'value' column to numeric (coerce errors to NaN) and drop rows with NaN.
                    ts['value'] = pd.to_numeric(ts['value'], errors='coerce')
                    ts = ts.dropna(subset=['value'])
                    ts['value'] = ts['value'].astype(np.float32)
                    # For KPI training, we might not have anomaly info; set it to 0.
                    ts['anomaly'] = 0
                    ts['label'] = -1
                else:
                    # For other files assume Yahoo format: column index 1 is "value", and index 2 is "anomaly".
                    ts = pd.read_csv(self.repodirext[i],
                                     usecols=[1, 2],
                                     header=0,
                                     names=['value', 'anomaly'],
                                     encoding='latin1')
                    # Convert 'value' column to numeric and drop invalid rows.
                    ts['value'] = pd.to_numeric(ts['value'], errors='coerce')
                    ts = ts.dropna(subset=['value'])
                    ts['value'] = ts['value'].astype(np.float32)
                    ts['label'] = -1
            except Exception as e:
                print("Error reading file:", self.repodirext[i])
                raise e

            # Scale the 'value' column to [0,1] using MinMaxScaler.
            scaler = sklearn.preprocessing.MinMaxScaler()
            if ts[['value']].shape[0] == 0:
                print("Warning: file {} has no valid data; skipping.".format(self.repodirext[i]))
                continue
            ts['value'] = scaler.fit_transform(ts[['value']])
            self.timeseries_repo.append(ts)

        # Update dataset size.
        if len(self.timeseries_repo) == 0:
            raise ValueError("No valid time series data found in directory: {}".format(self.repodir))
        self.datasetsize = len(self.timeseries_repo)
        self.datasetfix = 0
        self.datasetidx = random.randint(0, self.datasetsize - 1)
        self.datasetrng = self.datasetsize

    def reset(self):
        """
        Reset the environment: choose a new time series, reset the cursor, and compute the initial state.
        """
        if self.datasetfix == 0:
            self.datasetidx = (self.datasetidx + 1) % self.datasetrng

        print("Loading file: ", self.repodirext[self.datasetidx])
        self.timeseries = self.timeseries_repo[self.datasetidx]
        self.timeseries_curser = self.timeseries_curser_init
        self.timeseries_states = self.statefnc(self.timeseries, self.timeseries_curser)
        self.states_list = self.get_states_list()
        return self.timeseries_states

    def reset_to(self, id):
        """
        Reset the environment to a specific file by its index.
        """
        if id < 0 or id >= self.datasetrng:
            raise ValueError("Invalid dataset index: {}".format(id))
        self.datasetidx = id
        self.timeseries = self.timeseries_repo[self.datasetidx]
        self.timeseries_curser = self.timeseries_curser_init
        self.timeseries_states = self.statefnc(self.timeseries, self.timeseries_curser)
        self.states_list = self.get_states_list()
        return self.timeseries_states

    def reset_getall(self):
        """
        Load the entire dataset (including a timestamp column) and scale the values.
        """
        if self.datasetfix == 0:
            self.datasetidx = (self.datasetidx + 1) % self.datasetrng

        self.timeseries = pd.read_csv(self.repodirext[self.datasetidx],
                                      usecols=[0, 1, 2],
                                      header=0,
                                      names=['timestamp', 'value', 'anomaly'],
                                      encoding='latin1')
        self.timeseries = self.timeseries.astype(np.float32)
        self.timeseries_curser = self.timeseries_curser_init

        scaler = sklearn.preprocessing.MinMaxScaler()
        self.timeseries['value'] = scaler.fit_transform(self.timeseries[['value']])
        return self.timeseries

    def step(self, action):
        """
        Take a step in the environment.
        Returns a tuple: (state, reward, done, info)
        """
        # 1. Get the reward based on the current state and the given action.
        reward = self.rewardfnc(self.timeseries, self.timeseries_curser, action)
        # 2. Advance the time series cursor.
        self.timeseries_curser += 1

        if self.timeseries_curser >= self.timeseries['value'].size:
            done = 1
            # At terminal state, return the same state twice.
            state = np.array([self.timeseries_states, self.timeseries_states])
        else:
            done = 0
            # Compute the next state.
            state = self.statefnc(self.timeseries, self.timeseries_curser, self.timeseries_states, action)

        # Update the stored state.
        if isinstance(state, np.ndarray) and state.ndim > np.array(self.timeseries_states).ndim:
            self.timeseries_states = state[action]
        else:
            self.timeseries_states = state

        return state, reward, done, []

    def get_states_list(self):
        """
        Build and return a list of states for the current time series.
        """
        self.timeseries = self.timeseries_repo[self.datasetidx]
        self.timeseries_curser = self.timeseries_curser_init
        state_list = []
        for cursor in range(self.timeseries_curser_init, self.timeseries['value'].size):
            if len(state_list) == 0:
                state = self.statefnc(self.timeseries, cursor)
            else:
                state = self.statefnc(self.timeseries, cursor, state_list[-1])
                # If the state function returns multiple states (e.g., for binary branching), take the first one.
                if isinstance(state, np.ndarray) and state.ndim > 1:
                    state = state[0]
            state_list.append(state)
        return state_list
