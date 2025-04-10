import pandas as pd
import numpy as np
import random
import os
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
    def __init__(self, sensor_dir='SMD/ServerMachineDataset/test/', label_dir='SMD/ServerMachineDataset/test_label/'):
        """
        sensor_dir : Directory that contains sensor CSV or TXT files.
        label_dir  : Directory that contains corresponding label files (one column of labels).
        Matching is done based on the file name without extension.
        """
        # List sensor files.
        self.sensor_files = []
        for subdir, dirs, files in os.walk(sensor_dir):
            for file in files:
                if file.endswith('.csv') or file.endswith('.txt'):
                    self.sensor_files.append(os.path.join(subdir, file))

        if len(self.sensor_files) == 0:
            raise ValueError("No sensor files found in directory: {}".format(sensor_dir))

        # List label files.
        self.label_files = {}
        for subdir, dirs, files in os.walk(label_dir):
            for file in files:
                if file.endswith('.csv') or file.endswith('.txt'):
                    # Use the file name without extension as key.
                    base_name = os.path.splitext(file)[0]
                    self.label_files[base_name] = os.path.join(subdir, file)

        if len(self.label_files) == 0:
            raise ValueError("No label files found in directory: {}".format(label_dir))

        self.action_space_n = len(action_space)
        self.statefnc = defaultStateFuc
        self.rewardfnc = defaultRewardFuc

        self.datasetsize = len(self.sensor_files)
        self.datasetfix = 0
        self.datasetidx = random.randint(0, self.datasetsize - 1)
        self.datasetrng = self.datasetsize

        self.timeseries = None  # This will hold the merged sensor & label DataFrame.
        self.timeseries_curser = -1
        self.timeseries_curser_init = 0
        self.timeseries_states = None
        self.states_list = []

        # Pre-load and merge each sensor file with its matching label file.
        self.timeseries_repo = []
        for sensor_path in self.sensor_files:
            # Read sensor data. We assume sensor value is contained in the first column.
            df_sensor = pd.read_csv(sensor_path, sep=",", header=None)
            df_sensor = df_sensor[[0]]
            df_sensor.columns = ['value']
            # Scale the 'value' column to [0,1].
            scaler = sklearn.preprocessing.MinMaxScaler()
            df_sensor['value'] = scaler.fit_transform(df_sensor[['value']])

            # Get the base name for matching.
            sensor_base = os.path.splitext(os.path.basename(sensor_path))[0]
            if sensor_base in self.label_files:
                label_path = self.label_files[sensor_base]
            else:
                raise ValueError("No matching label file found for sensor file: {}".format(sensor_path))

            # Read the label file (assumed one column without header).
            df_label = pd.read_csv(label_path, sep=",", header=None)
            df_label.columns = ['anomaly']
            # Trim both to the same length.
            min_length = min(df_sensor.shape[0], df_label.shape[0])
            df_sensor = df_sensor.iloc[:min_length].reset_index(drop=True)
            df_label = df_label.iloc[:min_length].reset_index(drop=True)

            # Merge sensor values and labels.
            df_merged = pd.concat([df_sensor, df_label], axis=1)
            # Add a "label" column for training/pseudo-labeling (initially set to -1).
            df_merged['label'] = -1
            df_merged = df_merged.astype(np.float32)

            self.timeseries_repo.append(df_merged)

    def reset(self):
        """
        Reset the environment by selecting a new timeseries, resetting the cursor,
        and computing the initial state.
        """
        if self.datasetfix == 0:
            self.datasetidx = (self.datasetidx + 1) % self.datasetrng

        print("Loading sensor file:", self.sensor_files[self.datasetidx])
        base_name = os.path.splitext(os.path.basename(self.sensor_files[self.datasetidx]))[0]
        print("Using label file:", self.label_files.get(base_name, "Not found"))
        self.timeseries = self.timeseries_repo[self.datasetidx]
        self.timeseries_curser = self.timeseries_curser_init
        self.timeseries_states = self.statefnc(self.timeseries, self.timeseries_curser)
        self.states_list = self.get_states_list()
        return self.timeseries_states

    def reset_to(self, id):
        if id < 0 or id >= self.datasetrng:
            raise ValueError("Invalid dataset index: {}".format(id))
        self.datasetidx = id
        self.timeseries = self.timeseries_repo[self.datasetidx]
        self.timeseries_curser = self.timeseries_curser_init
        self.timeseries_states = self.statefnc(self.timeseries, self.timeseries_curser)
        self.states_list = self.get_states_list()
        return self.timeseries_states

    def get_states_list(self):
        self.timeseries = self.timeseries_repo[self.datasetidx]
        self.timeseries_curser = self.timeseries_curser_init
        state_list = []
        for cursor in range(self.timeseries_curser_init, self.timeseries.shape[0]):
            if len(state_list) == 0:
                state = self.statefnc(self.timeseries, cursor)
            else:
                state = self.statefnc(self.timeseries, cursor, state_list[-1])
                if isinstance(state, np.ndarray) and state.ndim > 1:
                    state = state[0]
            state_list.append(state)
        return state_list
