import pandas as pd
import numpy as np
import random
import os
import sklearn.preprocessing

NOT_ANOMALY = 0
ANOMALY = 1
action_space = [NOT_ANOMALY, ANOMALY]


class EnvTimeSeriesfromRepo():
    """
    A revised environment that:
      1) Loads CSV files from 'repodir' (each must have columns [value, anomaly]).
      2) Scales 'value' to [0,1] with MinMaxScaler.
      3) Builds an RNN-friendly state list: each state is a (n_steps,1) window of 'value'.
      4) Steps through the time series, returning (state, reward, done, info).
    """

    def __init__(self,
                 repodir='environment/',
                 n_steps=25,
                 datasetfix=0):
        """
        Args:
            repodir (str): path containing .csv files.
            n_steps (int): sliding window size for RNN states.
            datasetfix (int): if 0, cycle through all CSVs; otherwise fix an index.
        """
        self.repodir = repodir
        self.repodirext = []

        for subdir, dirs, files in os.walk(self.repodir):
            for file in files:
                if file.endswith('.csv'):
                    self.repodirext.append(os.path.join(subdir, file))

        if len(self.repodirext) == 0:
            raise ValueError(f"No .csv files found in {repodir} — can't create environment.")

        self.n_steps = n_steps
        self.action_space_n = len(action_space)

        # Will hold each CSV as a DataFrame of columns: ['value','anomaly','label']
        self.timeseries_repo = []

        # Load all CSVs into memory
        for path_ in self.repodirext:
            # Expect columns [1,2] => 'value','anomaly'
            ts = pd.read_csv(path_, usecols=[1, 2], header=0, names=['value','anomaly'])
            # Mark all labels as -1 initially (unlabeled)
            ts['label'] = -1

            # Convert to float32
            ts = ts.astype(np.float32)

            # Scale 'value' to [0,1]
            scaler = sklearn.preprocessing.MinMaxScaler()
            ts['value'] = scaler.fit_transform(ts[['value']])  # shape => (len,1)

            self.timeseries_repo.append(ts.reset_index(drop=True))

        self.datasetsize = len(self.repodirext)
        self.datasetfix = datasetfix
        self.datasetidx = random.randint(0, self.datasetsize - 1) if self.datasetsize>0 else 0
        self.datasetrng = self.datasetsize

        # We'll track current timeseries as a DataFrame, and a cursor for stepping
        self.timeseries = None
        self.timeseries_curser_init = self.n_steps
        self.timeseries_curser = self.n_steps

        # For building & storing the entire sequence of states
        self.states_list = []

    def reset(self):
        """
        Move to the next dataset (unless datasetfix!=0) and reset the time-series cursor.
        Build self.states_list as a list of shape-(n_steps,1) states.
        Returns: The initial state (n_steps,1).
        """
        # 1) pick next dataset, or keep the same if datasetfix != 0
        if self.datasetfix == 0:
            self.datasetidx = (self.datasetidx + 1) % self.datasetrng

        print(f"Loading dataset: {self.repodirext[self.datasetidx]}")
        self.timeseries = self.timeseries_repo[self.datasetidx].copy()

        # 2) set the cursor
        self.timeseries_curser = self.timeseries_curser_init

        # 3) build states_list
        self.states_list = []
        n_total = len(self.timeseries)
        for idx in range(n_total):
            st = self._build_state(idx)
            self.states_list.append(st)

        # If we don't even have n_steps rows, states_list can be short
        # Return an initial state
        if self.timeseries_curser < len(self.states_list):
            return self.states_list[self.timeseries_curser]
        else:
            # If the series is < n_steps, we can't index timeseries_curser
            # Return zeros for safety
            return np.zeros((self.n_steps,1), dtype=np.float32)

    def _build_state(self, idx):
        """
        Build a single RNN state by taking the last self.n_steps
        values of 'value' up to 'idx' (exclusive).
        Returns shape (n_steps,1).
        """
        if idx < self.n_steps:
            # If not enough history, front-pad with zeros
            pad_len = self.n_steps - idx
            front_pad = np.zeros(pad_len, dtype=np.float32)
            val_part  = self.timeseries['value'][:idx].values
            st_1d     = np.concatenate([front_pad, val_part], axis=0)
        else:
            st_1d     = self.timeseries['value'][idx - self.n_steps : idx].values
        # reshape => (n_steps,1)
        return st_1d.reshape((self.n_steps, 1))

    def step(self, action):
        """
        Moves one step forward in time (self.timeseries_curser += 1).
        Returns: (next_state, reward, done, info)
          next_state: shape (n_steps,1) or possibly [state, state] if your RL code expects 2D
          reward: [r_if_action0, r_if_action1],
                  because your RL code wants a 2-element list for the 2 actions
          done: 1 if we're past the end, else 0
          info: {}
        """
        # If we're already at or past end => done
        n_total = len(self.timeseries)
        if self.timeseries_curser >= n_total:
            done = 1
            # Provide a final state (just zeros)
            final_st = np.zeros((self.n_steps,1), dtype=np.float32)
            # Provide a 2-element reward array => [0,0] by default
            reward_array = [0.0, 0.0]
            return [final_st, final_st], reward_array, done, {}

        # 1) Calculate reward based on ground truth "anomaly" vs action
        #    We return a 2-element array for the code that does reward[action].
        #    Example: if anomaly=0 => correct if action=0 => +1 => we do [1, -1].
        true_label = int(self.timeseries.at[self.timeseries_curser, 'anomaly'])
        if true_label == action:
            # correct
            r_value = 1.0
        else:
            # incorrect
            r_value = -1.0

        # We create a reward array = [something_for_NOT_ANOMALY, something_for_ANOMALY]
        # Then we fill the correct index with r_value, the other with 0 or negative, etc.
        # But in your code, you often do e.g. [TN_Value, FP_Value] or [FN_Value, TP_Value].
        # We'll keep it simple: [0,0], then put r_value in the chosen action index
        # so your code can do: reward[action].
        reward_array = [0.0, 0.0]
        reward_array[action] = r_value

        # 2) Advance cursor
        self.timeseries_curser += 1
        done = 1 if (self.timeseries_curser >= n_total) else 0

        # 3) Next state if not done
        if not done:
            st = self.states_list[self.timeseries_curser]
        else:
            # if done, we can return a final or zero state
            st = np.zeros((self.n_steps,1), dtype=np.float32)

        # The code in your RL typically expects step() to return 2 states in a list:
        # e.g. next_state[action], etc. So we do a quick trick: [st, st].
        # That way your code can do: next_state[action].
        # If your code is simpler, you can just return st alone.
        return [st, st], reward_array, done, {}

    # Optionally, a helper if you want to re-build states after changing 'label'
    def get_states_list(self):
        return self.states_list
