# env_wadi.py
import pandas as pd
import numpy as np

# Default state function: returns the sensor value at the current time index.
def defaultStateFuc(timeseries, timeseries_curser, previous_state=None, action=None):
    return timeseries['value'][timeseries_curser]

# Default reward: returns +1 if action equals the ground truth anomaly value, else -1.
def defaultRewardFuc(timeseries, timeseries_curser, action):
    if action == timeseries['anomaly'][timeseries_curser]:
        return 1
    else:
        return -1

class EnvTimeSeriesWaDi:
    """
    Environment wrapper for WaDi dataset. Reads one sensor CSV and one label CSV,
    aligns lengths, exposes timeseries DataFrame with columns ['value','anomaly','label'].
    Usage:
        env = EnvTimeSeriesWaDi(sensor_csv, label_csv, n_steps)
    """
    def __init__(self, sensor_csv, label_csv, n_steps):
        # Load sensor data
        df_sensor = pd.read_csv(sensor_csv)
        # Load label data, skip header rows
        df_label = pd.read_csv(label_csv, header=None, low_memory=False)
        raw_labels = df_label.iloc[2:, -1].astype(int).reset_index(drop=True)
        # map WaDi labels (1 → normal, -1 → attack) to (0,1)
        labels = raw_labels.replace({1: 0, -1: 1}).values

        # Align lengths
        len_sensor = len(df_sensor)
        len_labels = len(labels)
        min_len = min(len_sensor, len_labels)

        # Trim sensor and labels to same length
        values = df_sensor['TOTAL_CONS_REQUIRED_FLOW'].astype(float).values[:min_len]
        labels = labels[:min_len]

        # Build merged DataFrame: 'value', 'anomaly', 'label'(-1 init)
        ts = pd.DataFrame({
            'value':   values,
            'anomaly': labels,
            'label':   np.full(min_len, -1, dtype=np.int32)
        })

        # store attributes
        self.timeseries_repo        = [ts.astype(np.float32)]
        self.timeseries             = None
        self.timeseries_curser_init = n_steps
        self.timeseries_curser      = -1
        self.statefnc               = defaultStateFuc
        self.rewardfnc              = defaultRewardFuc
        self.action_space_n         = 2
        self.datasetsize            = 1
        self.datasetrng             = 1
        self.datasetidx             = 0
        self.states_list            = []

    def reset(self):
        """Reset cursor and return initial state."""
        self.timeseries = self.timeseries_repo[0]
        self.timeseries_curser = self.timeseries_curser_init
        self.timeseries_states = self.statefnc(self.timeseries, self.timeseries_curser)
        self.states_list = self.get_states_list()
        return self.timeseries_states

    def get_states_list(self):
        """Build all states from cursor init to end."""
        states = []
        for cursor in range(self.timeseries_curser_init, len(self.timeseries)):
            if cursor == self.timeseries_curser_init:
                state = self.statefnc(self.timeseries, cursor)
            else:
                state = self.statefnc(self.timeseries, cursor, states[-1])
                if isinstance(state, np.ndarray) and state.ndim > 1:
                    state = state[0]
            states.append(state)
        return states

    def step(self, action):
        """Take action, return (next_state, reward, done, info)."""
        reward = self.rewardfnc(self.timeseries, self.timeseries_curser, action)
        self.timeseries_curser += 1
        if self.timeseries_curser >= len(self.timeseries):
            done = 1
            state = np.array([self.timeseries_states, self.timeseries_states])
        else:
            done = 0
            state = self.statefnc(self.timeseries, self.timeseries_curser, self.timeseries_states, action)
        # update stored state
        if isinstance(state, np.ndarray) and state.ndim > np.array(self.timeseries_states).ndim:
            self.timeseries_states = state[action]
        else:
            self.timeseries_states = state
        return state, reward, done, {}
