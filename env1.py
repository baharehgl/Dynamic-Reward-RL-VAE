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
    Default state function: returns the 'value' at the current time index.
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
        """
        This environment now ONLY looks for .csv files in repodir.
        It does NOT read any .hdf files, so PyTables is unnecessary.
        """
        self.repodir = repodir
        self.repodirext = []

        # Walk through the directory, gather ONLY .csv files, skip __MACOSX and hidden files.
        for subdir, dirs, files in os.walk(self.repodir):
            if '__MACOSX' in subdir:
                continue
            for file in files:
                # skip hidden files that start with ._
                if file.endswith('.csv') and not file.startswith('._'):
                    self.repodirext.append(os.path.join(subdir, file))

        if len(self.repodirext) == 0:
            raise ValueError(f"No CSV files found in directory: {self.repodir}")

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

        # Process each CSV file and store it in timeseries_repo
        for path in self.repodirext:
            try:
                # Attempt to read columns [1, 2] as 'value' and 'anomaly'
                # Adjust if your CSV has a different column layout.
                ts = pd.read_csv(
                    path,
                    usecols=[1, 2],
                    header=0,
                    names=['value', 'anomaly'],
                    encoding='latin1'
                )

                # Convert 'value' to numeric, drop rows with NaN
                ts['value'] = pd.to_numeric(ts['value'], errors='coerce')
                ts.dropna(subset=['value'], inplace=True)
                ts['value'] = ts['value'].astype(np.float32)

                # If there's no 'anomaly' column or it's empty, set it to 0
                if 'anomaly' not in ts.columns:
                    ts['anomaly'] = 0
                else:
                    # Convert 'anomaly' to numeric in case it's not
                    ts['anomaly'] = pd.to_numeric(ts['anomaly'], errors='coerce').fillna(0).astype(np.int32)

                # Optionally add a label column for RL usage
                ts['label'] = -1

                # Scale the 'value' column to [0,1] using MinMaxScaler.
                if ts.shape[0] == 0:
                    print(f"Warning: file {path} has no valid data; skipping.")
                    continue
                scaler = sklearn.preprocessing.MinMaxScaler()
                ts['value'] = scaler.fit_transform(ts[['value']])

                self.timeseries_repo.append(ts)

            except Exception as e:
                print(f"Error reading file: {path}\n{e}")
                # skip this file
                continue

        if len(self.timeseries_repo) == 0:
            raise ValueError(f"No valid time series data found in directory: {self.repodir}")

        self.datasetsize = len(self.timeseries_repo)
        self.datasetfix = 0
        # Start on a random file
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
        Load the entire dataset with columns [0,1,2], typically [timestamp, value, anomaly].
        If your CSV doesn't have these columns, adjust accordingly.
        """
        if self.datasetfix == 0:
            self.datasetidx = (self.datasetidx + 1) % self.datasetrng

        path = self.repodirext[self.datasetidx]
        ts = pd.read_csv(
            path,
            usecols=[0, 1, 2],
            header=0,
            names=['timestamp', 'value', 'anomaly'],
            encoding='latin1'
        )
        ts = ts.astype(np.float32)
        self.timeseries_curser = self.timeseries_curser_init

        scaler = sklearn.preprocessing.MinMaxScaler()
        ts['value'] = scaler.fit_transform(ts[['value']])
        self.timeseries = ts
        return self.timeseries

    def step(self, action):
        """
        Take a step in the environment.
        Returns a tuple: (state, reward, done, info)
        """
        reward = self.rewardfnc(self.timeseries, self.timeseries_curser, action)
        self.timeseries_curser += 1

        if self.timeseries_curser >= self.timeseries['value'].size:
            done = 1
            state = np.array([self.timeseries_states, self.timeseries_states])
        else:
            done = 0
            state = self.statefnc(self.timeseries, self.timeseries_curser, self.timeseries_states, action)

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
                if isinstance(state, np.ndarray) and state.ndim > 1:
                    state = state[0]
            state_list.append(state)
        return state_list
