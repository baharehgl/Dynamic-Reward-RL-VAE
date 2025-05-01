# env_wadi.py  ────────────────────────────────────────────────────────────────
"""
Light-weight environment wrapper for the WADI 14-day dataset that keeps the
*entire* numeric sensor table, so the outer RL code can slice ts[feature_cols]
without column-mismatch errors.

The state-function signature is (timeseries, cursor) – **two positional
arguments only** – exactly what `make_state` in myasp-wadi.py expects.
"""

import pandas as pd
import numpy as np


# ─── default callbacks (can be overwritten by the main script) ────────────────
def default_state_fnc(timeseries, cursor):
    """Return the raw sensor row at position <cursor>."""
    return timeseries.iloc[cursor]


def default_reward_fnc(timeseries, cursor, action):
    """+1 if action matches ground-truth anomaly flag, else -1."""
    return 1 if action == timeseries["anomaly"][cursor] else -1


# ─── environment class ───────────────────────────────────────────────────────
class EnvTimeSeriesWaDi:
    """
    Minimal, framework-agnostic environment.

    Attributes filled/used by the outer RL script:
      • statefnc   – callable(ts, cursor)         (default above)
      • rewardfnc  – callable(ts, cursor, action) (default above)
    """

    # ---------------------------------------------------------------- init --
    def __init__(self, sensor_csv, label_csv, n_steps):
        self._load_sensor_csv(sensor_csv)
        self._attach_labels(label_csv)

        self.timeseries_repo        = [self._ts]
        self.timeseries_curser_init = n_steps
        self.statefnc               = default_state_fnc
        self.rewardfnc              = default_reward_fnc
        self.action_space_n         = 2

    # ------------------------------------------------ private helpers -------
    def _load_sensor_csv(self, path):
        """Read CSV and keep every column that contains at least one number."""
        df = pd.read_csv(path, decimal=".")
        df.columns = df.columns.str.strip()                       # trim blanks
        df = df.apply(pd.to_numeric, errors="coerce")             # coerce
        df = df.dropna(axis=1, how="all")                         # drop empty
        df = df.dropna(axis=0, how="all").reset_index(drop=True)  # drop blank
        if "Row" in df.columns:                                   # optional
            df = df.drop(columns=["Row"])
        self._sensor_df = df

    def _attach_labels(self, label_csv):
        lbl_df  = pd.read_csv(label_csv, header=1, low_memory=False)
        raw_lbl = lbl_df["Attack LABLE (1:No Attack, -1:Attack)"].astype(int)
        anomalies = np.where(raw_lbl.values == 1, 0, 1)           # 0 = normal

        L = min(len(self._sensor_df), len(anomalies))
        ts = self._sensor_df.iloc[:L].copy()
        ts["anomaly"] = anomalies[:L]   # immutable ground truth
        ts["label"]   = -1              # to be filled by AL / LP
        self._ts = ts

    def _precompute_states(self):
        """Cache the state for every time-step (optional, used by LP logic)."""
        states = []
        for c in range(self.timeseries_curser_init, len(self.timeseries)):
            s = self.statefnc(self.timeseries, c)
            states.append(s if not (isinstance(s, np.ndarray) and s.ndim > 1)
                          else s[0])
        return states

    # ---------------------------------------------------------------- API --
    def reset(self):
        self.timeseries        = self.timeseries_repo[0]
        self.timeseries_curser = self.timeseries_curser_init
        self.timeseries_states = self.statefnc(self.timeseries,
                                               self.timeseries_curser)
        self.states_list       = self._precompute_states()
        return self.timeseries_states

    def step(self, action):
        r = self.rewardfnc(self.timeseries, self.timeseries_curser, action)
        self.timeseries_curser += 1
        done = int(self.timeseries_curser >= len(self.timeseries))

        if done:   # duplicate last state so outer code can index [action]
            state = np.array([self.timeseries_states,
                              self.timeseries_states])
        else:
            state = self.statefnc(self.timeseries, self.timeseries_curser)

        # update cached last state (match dimensionality logic in main code)
        if isinstance(state, np.ndarray) and \
           state.ndim > np.array(self.timeseries_states).ndim:
            self.timeseries_states = state[action]
        else:
            self.timeseries_states = state

        return state, r, done, {}


# ──────────────────────────────────── test run ───────────────────────────────
if __name__ == "__main__":
    import os
    BASE = os.path.dirname(os.path.abspath(__file__))
    env = EnvTimeSeriesWaDi(
        os.path.join(BASE, "WaDi", "WADI_14days_new.csv"),
        os.path.join(BASE, "WaDi", "WADI_attackdataLABLE.csv"),
        n_steps=25,
    )
    s0 = env.reset()
    print("first state shape:", np.shape(s0))
    s1, r, done, _ = env.step(0)
    print("step -> state shape:", np.shape(s1), "reward:", r, "done:", done)
