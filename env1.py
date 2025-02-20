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
        """
        This environment now supports two file types:
          - CSV  (searches for *.csv)
          - HDF  (searches for *.hdf)
        For HDF, we assume it has columns like ["timestamp", "value", "anomaly"] or at least "value".
        If a file is missing "value", we skip it. If missing "anomaly", we set anomaly=0 by default.
        """
        self.repodir = repodir
        self.repodirext = []

        # Walk through directory, gather CSV and HDF files, but skip __MACOSX and hidden files.
        for subdir, dirs, files in os.walk(self.repodir):
            if '__MACOSX' in subdir:
                continue
            for file in files:
                if file.startswith('._'):
                    continue
                if file.endswith('.csv') or file.endswith('.hdf'):
                    self.repodirext.append(os.path.join(subdir, file))

        if len(self.repodirext) == 0:
            raise ValueError("No CSV or HDF files found in directory: {}".format(self.repodir))

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

        # Read and preprocess each file in repodirext.
        for path in self.repodirext:
            fname = os.path.basename(path)
            # Attempt to read CSV or HDF:
            df = None
            try:
                if path.endswith('.csv'):
                    df = self._load_csv(path, fname)
                else:
                    df = self._load_hdf(path, fname)
            except Exception as e:
                print(f"Error reading file: {path}\n{e}")
                continue

            if df is None or df.shape[0] == 0:
                print(f"Warning: file {path} had no valid data; skipping.")
                continue

            # Scale the 'value' column to [0,1] using MinMaxScaler.
            scaler = sklearn.preprocessing.MinMaxScaler()
            df['value'] = scaler.fit_transform(df[['value']])

            self.timeseries_repo.append(df)

        if len(self.timeseries_repo) == 0:
            raise ValueError("No valid time series data found in directory: {}".format(self.repodir))

        # The environment picks a random file to start with.
        self.datasetsize = len(self.timeseries_repo)
        self.datasetfix = 0
        self.datasetidx = random.randint(0, self.datasetsize - 1)
        self.datasetrng = self.datasetsize

    def _load_csv(self, path, fname):
        """
        Load a CSV file. If it's the KPI training file, we handle it differently from the 'Yahoo' format.
        """
        df = pd.read_csv(path, encoding='latin1')
        # If "phase2_train" in filename, assume it's KPI training data with no anomaly column:
        if "phase2_train" in fname:
            # If there's no 'value' column, rename the second column to 'value' if it exists.
            if 'value' not in df.columns:
                cols = list(df.columns)
                if len(cols) >= 2:
                    df.rename(columns={cols[1]: 'value'}, inplace=True)
                else:
                    raise ValueError(f"File {path} does not contain enough columns for KPI training.")
            # Convert 'value' to numeric, drop rows with NaN.
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df.dropna(subset=['value'], inplace=True)
            df['value'] = df['value'].astype(np.float32)

            # We do not have an 'anomaly' column for training. Let's set it to 0.
            if 'anomaly' not in df.columns:
                df['anomaly'] = 0
            df['label'] = -1
        else:
            # For 'Yahoo' format: if it doesn't match, we attempt reading columns [1,2].
            # But we already read the entire file, so let's see if 'value' or 'anomaly' is present.
            if 'value' not in df.columns or 'anomaly' not in df.columns:
                # If not present, let's try reading columns 1 and 2:
                df = pd.read_csv(path, usecols=[1, 2], header=0, names=['value', 'anomaly'], encoding='latin1')
            # Convert columns to numeric, drop invalid.
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df.dropna(subset=['value'], inplace=True)
            df['value'] = df['value'].astype(np.float32)
            # If no 'anomaly' column, set it to 0.
            if 'anomaly' not in df.columns:
                df['anomaly'] = 0
            df['label'] = -1

        return df

    def _load_hdf(self, path, fname):
        """
        Load an HDF file. We assume it has columns 'value' and possibly 'anomaly'.
        If 'anomaly' is missing, set it to 0.
        """
        # If the HDF has only one key, read it directly; if multiple, specify the key you need.
        # For KPI 'phase2_ground_truth.hdf', you might need a specific key. If unknown, try listing them.
        # e.g. keys = pd.HDFStore(path).keys()

        # Here we assume there's only one top-level dataset or the user wants the default.
        df = pd.read_hdf(path)  # read default or single dataset
        # Make sure there's a 'value' column
        if 'value' not in df.columns:
            raise ValueError(f"HDF file {path} does not contain a 'value' column.")
        # Convert 'value' to numeric, drop invalid.
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df.dropna(subset=['value'], inplace=True)
        df['value'] = df['value'].astype(np.float32)

        # If there's no 'anomaly' column, set it to 0.
        if 'anomaly' not in df.columns:
            df['anomaly'] = 0
        # If there's no 'label' column, add it.
        if 'label' not in df.columns:
            df['label'] = -1

        return df

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

        # If you specifically need [timestamp, value, anomaly] from the CSV or HDF, adapt here.
        # For now, let's assume it's a CSV with three columns [0,1,2].
        path = self.repodirext[self.datasetidx]
        if path.endswith('.csv'):
            df = pd.read_csv(path, usecols=[0, 1, 2], header=0, names=['timestamp', 'value', 'anomaly'],
                             encoding='latin1')
        else:
            # If HDF, read it differently or read the entire thing.
            df = pd.read_hdf(path)
            # If it doesn't have these columns, you need to adapt accordingly.
            if 'timestamp' not in df.columns:
                df['timestamp'] = range(len(df))  # a dummy index
            if 'value' not in df.columns:
                raise ValueError(f"HDF file {path} does not have 'value' column.")
            if 'anomaly' not in df.columns:
                df['anomaly'] = 0
            # rename columns if needed
        df = df.astype(np.float32)
        self.timeseries = df
        self.timeseries_curser = self.timeseries_curser_init

        scaler = sklearn.preprocessing.MinMaxScaler()
        df['value'] = scaler.fit_transform(df[['value']])
        return df

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
