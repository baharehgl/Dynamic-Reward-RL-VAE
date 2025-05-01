# env_wadi.py  – full-sensor version compatible with make_state()
# -----------------------------------------------------------------
import pandas as pd
import numpy as np

# ---------------------- default callbacks (unchanged) ------------------------
def defaultStateFuc(timeseries, timeseries_curser, previous_state=None, action=None):
    return timeseries.iloc[timeseries_curser]          # full row

def defaultRewardFuc(timeseries, timeseries_curser, action):
    return 1 if action == timeseries['anomaly'][timeseries_curser] else -1


class EnvTimeSeriesWaDi:
    """
    Minimal environment for the WADI time-series.
    Keeps **all numeric sensor columns** so the outer RL code can slice
    `ts[feature_cols]` without a mismatch.
    """

    def __init__(self, sensor_csv, label_csv, n_steps):
        # ---------- 1) read sensor CSV and keep *every* numeric column -------
        df_sensor = pd.read_csv(sensor_csv, decimal='.')
        df_sensor.columns = df_sensor.columns.str.strip()          # trim blanks
        df_sensor = df_sensor.apply(pd.to_numeric, errors='coerce')# coerce
        df_sensor = df_sensor.dropna(axis=1, how='all')            # drop empty
        df_sensor = df_sensor.dropna(axis=0, how='all').reset_index(drop=True)
        if 'Row' in df_sensor.columns:                             # optional
            df_sensor = df_sensor.drop(columns=['Row'])

        # ---------- 2) attach ground-truth anomaly labels --------------------
        df_label = pd.read_csv(label_csv, header=1, low_memory=False)
        raw_lbl  = df_label["Attack LABLE (1:No Attack, -1:Attack)"] \
                     .astype(int).values
        anomalies = np.where(raw_lbl == 1, 0, 1)                   # 0 = normal

        L = min(len(df_sensor), len(anomalies))
        ts = df_sensor.iloc[:L].copy()
        ts['anomaly'] = anomalies[:L]         # immutable ground truth
        ts['label']   = -1                    # will be filled by AL / LP

        # ---------- 3) store & initialise ------------------------------------
        self.timeseries_repo        = [ts]
        self.timeseries_curser_init = n_steps
        self.statefnc               = defaultStateFuc
        self.rewardfnc              = defaultRewardFuc
        self.action_space_n         = 2

    # ------------------------------- RL wrappers -----------------------------
    def reset(self):
        self.timeseries        = self.timeseries_repo[0]
        self.timeseries_curser = self.timeseries_curser_init
        self.timeseries_states = self.statefnc(self.timeseries,
                                               self.timeseries_curser)
        self.states_list       = self._precompute_states()
        return self.timeseries_states

    def _precompute_states(self):
        states = []
        for c in range(self.timeseries_curser_init, len(self.timeseries)):
            if c == self.timeseries_curser_init:
                s = self.statefnc(self.timeseries, c)
            else:
                s = self.statefnc(self.timeseries, c, states[-1])
                if isinstance(s, np.ndarray) and s.ndim > 1:
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
            state = self.statefnc(self.timeseries, self.timeseries_curser)

        self.timeseries_states = state if not isinstance(
            state, np.ndarray) or state.ndim == np.array(self.timeseries_states).ndim \
            else state[action]
        return state, r, done, {}
