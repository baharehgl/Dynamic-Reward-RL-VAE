# env_wadi.py
import pandas as pd
import numpy as np

# Default state function: returns the sensor value at the current time index.
def defaultStateFuc(timeseries, timeseries_curser, previous_state=None, action=None):
    return timeseries['value'][timeseries_curser]

# Default reward: returns +1 if action equals the ground truth anomaly value, else -1.
def defaultRewardFuc(timeseries, timeseries_curser, action):
    return 1 if action == timeseries['anomaly'][timeseries_curser] else -1

class EnvTimeSeriesWaDi:
    """
    Environment wrapper for WaDi dataset.
    """
    def __init__(self, sensor_csv, label_csv, n_steps):
        # Load sensor and label data
        df_sensor = pd.read_csv(sensor_csv)
        df_label  = pd.read_csv(label_csv, header=None, low_memory=False)
        raw_labels = df_label.iloc[2:, -1].astype(int).reset_index(drop=True)
        labels = raw_labels.replace({1: 0, -1: 1}).values

        # Align lengths
        min_len = min(len(df_sensor), len(labels))
        values = df_sensor['TOTAL_CONS_REQUIRED_FLOW'].astype(float).values[:min_len]
        labels = labels[:min_len]

        # Build DataFrame
        ts = pd.DataFrame({
            'value':   values,
            'anomaly': labels,
            'label':   np.full(min_len, -1, dtype=np.int32)
        }).astype(np.float32)

        self.timeseries_repo        = [ts]
        self.timeseries_curser_init = n_steps
        self.statefnc               = defaultStateFuc
        self.rewardfnc              = defaultRewardFuc
        self.action_space_n         = 2

    def reset(self):
        self.timeseries = self.timeseries_repo[0]
        self.timeseries_curser = self.timeseries_curser_init
        self.timeseries_states = self.statefnc(self.timeseries, self.timeseries_curser)
        self.states_list = self.get_states_list()
        return self.timeseries_states

    def get_states_list(self):
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
        reward = self.rewardfnc(self.timeseries, self.timeseries_curser, action)
        self.timeseries_curser += 1
        done = int(self.timeseries_curser >= len(self.timeseries))
        if done:
            state = np.array([self.timeseries_states, self.timeseries_states])
        else:
            state = self.statefnc(self.timeseries, self.timeseries_curser, self.timeseries_states, action)
        if isinstance(state, np.ndarray) and state.ndim > np.array(self.timeseries_states).ndim:
            self.timeseries_states = state[action]
        else:
            self.timeseries_states = state
        return state, reward, done, {}
