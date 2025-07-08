import os
import random
import numpy as np
import pandas as pd
import sklearn.preprocessing

# Constants
NOT_ANOMALY = 0
ANOMALY     = 1
REWARD_CORRECT   = 1
REWARD_INCORRECT = -1
action_space = [NOT_ANOMALY, ANOMALY]

def defaultStateFuc(timeseries, cursor, previous_state=None, action=None):
    return timeseries['value'].iloc[cursor]

def defaultRewardFuc(timeseries, cursor, action):
    if action == timeseries['anomaly'].iloc[cursor]:
        return REWARD_CORRECT
    else:
        return REWARD_INCORRECT

class EnvKPI():
    """
    A time‐series RL env that:
     - Reads one CSV of all KPIs for training (timestamp,value,label,KPI ID)
     - Reads one CSV of ground‐truth anomalies (timestamp,value,anomaly)
     - Splits the training CSV by KPI ID, merges in the anomaly column,
       and builds one self.timeseries_repo entry per KPI.
    """
    def __init__(self, train_csv, test_csv, statefnc=defaultStateFuc, rewardfnc=defaultRewardFuc):
        self.statefnc  = statefnc
        self.rewardfnc = rewardfnc

        # 1) load
        df_train = pd.read_csv(train_csv)
        df_test  = pd.read_csv(test_csv)

        # 2) for each KPI, slice and merge
        self.timeseries_repo = []
        for kpi in df_train['KPI ID'].unique():
            df_kpi = df_train[df_train['KPI ID']==kpi].copy()
            df_kpi.sort_values('timestamp', inplace=True)
            # merge the anomaly column from test (on timestamp)
            df_merged = pd.merge(df_kpi,
                                 df_test[['timestamp','anomaly']],
                                 on='timestamp', how='left')
            # fill any missing anomalies as 0
            df_merged['anomaly'].fillna(0, inplace=True)
            # scale value to [0,1]
            scaler = sklearn.preprocessing.MinMaxScaler()
            df_merged['value'] = scaler.fit_transform(df_merged[['value']])
            # keep only the three columns we need
            ts = df_merged[['value','label','anomaly']].astype(np.float32)
            self.timeseries_repo.append(ts)

        if not self.timeseries_repo:
            raise ValueError(f"No KPIs found in train file {train_csv}")

        # internal state
        self.datasetidx        = 0
        self.timeseries        = None
        self.timeseries_states = None
        self.timeseries_cursor = 0

        self.action_space_n = len(action_space)
        self.datasetsize    = len(self.timeseries_repo)

    def reset(self, to_idx=None):
        """ Pick KPI series #to_idx (or next one) and return initial state. """
        if to_idx is not None:
            self.datasetidx = to_idx % self.datasetsize
        else:
            self.datasetidx = (self.datasetidx + 1) % self.datasetsize

        self.timeseries        = self.timeseries_repo[self.datasetidx]
        self.timeseries_cursor = 0
        self.timeseries_states = self.statefnc(self.timeseries, 0)
        return self.timeseries_states

    def step(self, action):
        """ Apply `action` at the current cursor, return (state, r, done, info). """
        r      = self.rewardfnc(self.timeseries, self.timeseries_cursor, action)
        self.timeseries_cursor += 1

        done = int(self.timeseries_cursor >= len(self.timeseries))
        if done:
            next_state = self.timeseries_states  # dummy
        else:
            next_state = self.statefnc(
                self.timeseries,
                self.timeseries_cursor,
                self.timeseries_states,
                action
            )
        # update stored state
        self.timeseries_states = (
            next_state[action]
            if (isinstance(next_state, np.ndarray) and next_state.ndim>np.ndim(self.timeseries_states))
            else next_state
        )
        return next_state, r, done, {}

    def get_states_list(self):
        """ Replay all states to build the full `states_list` (for warm-up / AL). """
        states = []
        for t in range(len(self.timeseries)):
            st = self.statefnc(self.timeseries, t, states[-1] if states else None)
            # if array of two, grab the first branch
            if isinstance(st, np.ndarray) and st.ndim>1:
                st = st[0]
            states.append(st)
        return states
