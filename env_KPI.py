# env_KPI.py
import os
import random
import numpy as np
import pandas as pd
import sklearn.preprocessing

# Constants\NOT_ANOMALY = 0
ANOMALY     = 1
REWARD_CORRECT   = 1
REWARD_INCORRECT = -1
action_space = [NOT_ANOMALY, ANOMALY]

# Default state and reward

def defaultStateFuc(timeseries, cursor, previous_state=None, action=None):
    return timeseries['value'].iloc[cursor]

def defaultRewardFuc(timeseries, cursor, action):
    if action == timeseries['anomaly'].iloc[cursor]:
        return REWARD_CORRECT
    else:
        return REWARD_INCORRECT

class EnvKPI():
    """
    Environment reading KPI_data/train/phase2_train.csv and KPI_data/test/phase2_ground_truth.csv
    Splits by KPI ID and merges in anomaly column, then exposes reset(), step(), and get_states_list().
    """
    def __init__(self, train_csv, test_csv,
                 statefnc=defaultStateFuc,
                 rewardfnc=defaultRewardFuc):
        self.statefnc  = statefnc
        self.rewardfnc = rewardfnc

        # Load
        df_train = pd.read_csv(train_csv)
        df_test  = pd.read_csv(test_csv)

        self.timeseries_repo = []
        # Split by KPI and merge
        for kpi in df_train['KPI ID'].unique():
            df_kpi = df_train[df_train['KPI ID']==kpi].copy()
            df_kpi.sort_values('timestamp', inplace=True)
            merged = pd.merge(
                df_kpi,
                df_test[['timestamp','anomaly']],
                on='timestamp', how='left'
            )
            merged['anomaly'].fillna(0, inplace=True)

            scaler = sklearn.preprocessing.MinMaxScaler()
            merged['value'] = scaler.fit_transform(merged[['value']])

            ts = merged[['value','label','anomaly']].astype(np.float32)
            self.timeseries_repo.append(ts)

        if not self.timeseries_repo:
            raise ValueError(f"No KPI series found in {train_csv}")

        self.datasetidx        = 0
        self.timeseries        = None
        self.timeseries_states = None
        self.timeseries_cursor = 0

        self.action_space_n = len(action_space)
        self.datasetsize    = len(self.timeseries_repo)

    def reset(self, to_idx=None):
        # cycle or set index
        if to_idx is None:
            self.datasetidx = (self.datasetidx + 1) % self.datasetsize
        else:
            self.datasetidx = to_idx % self.datasetsize

        self.timeseries        = self.timeseries_repo[self.datasetidx]
        self.timeseries_cursor = 0
        self.timeseries_states = self.statefnc(self.timeseries, 0)
        return self.timeseries_states

    def step(self, action):
        r = self.rewardfnc(self.timeseries, self.timeseries_cursor, action)
        self.timeseries_cursor += 1

        done = int(self.timeseries_cursor >= len(self.timeseries))
        if done:
            next_state = self.timeseries_states
        else:
            next_state = self.statefnc(
                self.timeseries,
                self.timeseries_cursor,
                self.timeseries_states,
                action
            )

        # update stored state
        if (isinstance(next_state, np.ndarray)
            and next_state.ndim > np.ndim(self.timeseries_states)):
            self.timeseries_states = next_state[action]
        else:
            self.timeseries_states = next_state

        return next_state, r, done, {}

    def get_states_list(self):
        """
        Return list of (n_steps x 2) states for RNNBinaryStateFuc.
        Skips t<n_steps and unwraps any 2-branch arrays.
        """
        state_list = []
        for t in range(len(self.timeseries)):
            prev = state_list[-1] if state_list else None
            st = self.statefnc(self.timeseries, t, prev)
            if st is None:
                continue
            # unwrap if shape (2,n_steps,2)
            if isinstance(st, np.ndarray) and st.ndim==3 and st.shape[0]==2:
                st = st[0]
            if not (isinstance(st, np.ndarray) and st.ndim==2 and st.shape[1]==2):
                raise ValueError(f"Unexpected state shape at t={t}: {st.shape}")
            state_list.append(st)
        return state_list
