import pandas as pd
import numpy as np

# Default state: just the sensor “value” at cursor.
def defaultStateFuc(timeseries, timeseries_curser, previous_state=None, action=None):
    return timeseries['value'][timeseries_curser]

# Default reward: +1 for correct anomaly guess, -1 otherwise.
def defaultRewardFuc(timeseries, timeseries_curser, action):
    return 1 if action == timeseries['anomaly'][timeseries_curser] else -1

class EnvTimeSeriesWaDi:
    """
    WaDi environment wrapper.
    - sensor_csv: path to WADI_14days_new.csv
    - label_csv:  path to WADI_attackdataLABLE.csv
    - n_steps:    sliding window length
    """
    def __init__(self, sensor_csv, label_csv, n_steps):
        # 1) Load sensor readings
        df_sensor = pd.read_csv(sensor_csv)
        # 2) Load labels (skip 2 header rows), map 1→0 (normal), -1→1 (attack)
        df_label = pd.read_csv(label_csv, header=None, low_memory=False)
        raw    = df_label.iloc[2:, -1].astype(int).reset_index(drop=True)
        labels = raw.replace({1:0, -1:1}).values

        # 3) Truncate to same length
        L = min(len(df_sensor), len(labels))
        vals   = df_sensor['TOTAL_CONS_REQUIRED_FLOW'].astype(float).values[:L]
        labels = labels[:L]

        # 4) Build unified DataFrame
        ts = pd.DataFrame({
            'value':   vals,
            'anomaly': labels,
            'label':   np.full(L, -1, dtype=np.int32)   # to be filled by RL/AL
        }).astype(np.float32)

        self.timeseries_repo        = [ts]
        self.timeseries_curser_init = n_steps
        self.statefnc               = defaultStateFuc
        self.rewardfnc              = defaultRewardFuc
        self.action_space_n         = 2

    def reset(self):
        # Pick our single series, reset cursor, build states_list
        self.timeseries       = self.timeseries_repo[0]
        self.timeseries_curser= self.timeseries_curser_init
        self.timeseries_states= self.statefnc(self.timeseries, self.timeseries_curser)
        self.states_list      = self.get_states_list()
        return self.timeseries_states

    def get_states_list(self):
        # Build the full list of initial states (for warm‑up)
        states = []
        for c in range(self.timeseries_curser_init, len(self.timeseries)):
            if c == self.timeseries_curser_init:
                s = self.statefnc(self.timeseries, c)
            else:
                s = self.statefnc(self.timeseries, c, states[-1])
                # If our statefnc returns a pair [s0, s1], take s0 for warm‑up
                if isinstance(s, np.ndarray) and s.ndim > 1:
                    s = s[0]
            states.append(s)
        return states

    def step(self, action):
        # Compute reward
        r = self.rewardfnc(self.timeseries, self.timeseries_curser, action)
        # Advance
        self.timeseries_curser += 1
        done = int(self.timeseries_curser >= len(self.timeseries))
        if done:
            # return dummy double‐state so shapes match
            state = np.array([self.timeseries_states, self.timeseries_states])
        else:
            state = self.statefnc(self.timeseries, self.timeseries_curser, self.timeseries_states, action)
        # Store the filtered state for next iteration
        if isinstance(state, np.ndarray) and state.ndim > np.array(self.timeseries_states).ndim:
            # pick branch = action for internal tracking
            self.timeseries_states = state[action]
        else:
            self.timeseries_states = state
        return state, r, done, {}
