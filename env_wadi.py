import pandas as pd
import numpy as np

def defaultStateFuc(timeseries, timeseries_curser, previous_state=None, action=None):
    return timeseries['value'][timeseries_curser]

def defaultRewardFuc(timeseries, timeseries_curser, action):
    return 1 if action == timeseries['anomaly'][timeseries_curser] else -1

class EnvTimeSeriesWaDi:
    def __init__(self, sensor_csv, label_csv, n_steps):
        df_sensor = pd.read_csv(sensor_csv)
        df_label  = pd.read_csv(label_csv, header=1, low_memory=False)
        raw       = df_label["Attack LABLE (1:No Attack, -1:Attack)"].astype(int).values
        labels    = np.where(raw==1, 0, 1)
        L         = min(len(df_sensor), len(labels))
        vals      = df_sensor['TOTAL_CONS_REQUIRED_FLOW'].astype(float).values[:L]
        labels    = labels[:L]
        ts = pd.DataFrame({
            'value':   vals,
            'anomaly': labels,
            'label':   np.full(L, -1, dtype=np.int32)
        }).astype(np.float32)
        self.timeseries_repo        = [ts]
        self.timeseries_curser_init = n_steps
        self.statefnc               = defaultStateFuc
        self.rewardfnc              = defaultRewardFuc
        self.action_space_n         = 2

    def reset(self):
        self.timeseries        = self.timeseries_repo[0]
        self.timeseries_curser = self.timeseries_curser_init
        self.timeseries_states = self.statefnc(self.timeseries, self.timeseries_curser)
        self.states_list       = self.get_states_list()
        return self.timeseries_states

    def get_states_list(self):
        states = []
        for c in range(self.timeseries_curser_init, len(self.timeseries)):
            if c == self.timeseries_curser_init:
                s = self.statefnc(self.timeseries, c)
            else:
                s = self.statefnc(self.timeseries, c, states[-1])
                if isinstance(s, np.ndarray) and s.ndim>1:
                    s = s[0]
            states.append(s)
        return states

    def step(self, action):
        r = self.rewardfnc(self.timeseries, self.timeseries_curser, action)
        self.timeseries_curser += 1
        done = int(self.timeseries_curser >= len(self.timeseries))
        if done:
            state = np.array([self.timeseries_states, self.timeseries_states])
        else:
            state = self.statefnc(self.timeseries,
                                  self.timeseries_curser,
                                  self.timeseries_states,
                                  action)
        if isinstance(state, np.ndarray) and state.ndim>np.array(self.timeseries_states).ndim:
            self.timeseries_states = state[action]
        else:
            self.timeseries_states = state
        return state, r, done, {}
