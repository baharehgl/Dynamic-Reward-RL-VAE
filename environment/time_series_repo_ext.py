import pandas as pd
import numpy as np
import random

NOT_ANOMALY = 0
ANOMALY = 1
action_space = [NOT_ANOMALY, ANOMALY]


class EnvTimeSeriesfromRepo:
    """
    A custom environment that does NOT read CSV from disk.
    Instead, you set data via set_train_data(...) or set_test_data(...).

    Each dataset must have columns:
      'value'   -> numeric time-series data
      'anomaly' -> ground-truth label (0=normal, 1=anomaly)

    We convert that to a DataFrame with optional 'label'=-1 (for unlabeled).

    The environment then:
      - Creates sliding-window states of size n_steps, shape (n_steps,1).
      - On 'reset()', we start at timeseries_curser = n_steps.
      - On 'step(action)', we compute a reward array = [r0, r1] and move forward.
    """

    def __init__(self, n_steps=25):
        self.n_steps = n_steps

        # The environment can hold separate train/test data, but at any moment
        # we operate on a single "timeseries" DataFrame (with 'value','anomaly','label').
        self.train_df = None
        self.test_df = None

        self.timeseries = None  # Will be the currently active dataset (train or test)
        self.timeseries_curser = 0
        self.states_list = []  # list of shape-(n_steps,1) states
        self.action_space_n = len(action_space)

        # By default, let's assume we start with train data
        self.use_train_data = True

    def set_train_data(self, x_train):
        """
        x_train can be:
          - A numpy array of shape (N,) or (N,1). We assume anomaly=0 for all rows.
          - A DataFrame with columns ['value','anomaly']. If anomaly missing, we set anomaly=0.
        """
        if isinstance(x_train, np.ndarray):
            # shape check
            if len(x_train.shape) == 1:
                # expand to (N,1)
                x_train = x_train.reshape(-1, 1)
            # build a DataFrame
            df = pd.DataFrame(x_train, columns=['value'])
            df['anomaly'] = 0  # assume no anomalies in train
        elif isinstance(x_train, pd.DataFrame):
            df = x_train.copy()
            # ensure 'value' column exists
            if 'value' not in df.columns:
                raise ValueError("Train DataFrame must have a 'value' column.")
            if 'anomaly' not in df.columns:
                df['anomaly'] = 0
        else:
            raise ValueError("x_train must be either a NumPy array or a pandas DataFrame.")

        df['label'] = -1  # unlabeled by default
        self.train_df = df.reset_index(drop=True)

    def set_test_data(self, df_test):
        """
        df_test must be a DataFrame with 'value' and 'anomaly' columns.
        We keep them as is. If there's no 'label' column, set to -1.
        """
        if not isinstance(df_test, pd.DataFrame):
            raise ValueError("df_test must be a pandas DataFrame.")
        if 'value' not in df_test.columns:
            raise ValueError("Test DataFrame must have a 'value' column.")
        if 'anomaly' not in df_test.columns:
            raise ValueError("Test DataFrame must have an 'anomaly' column.")

        df = df_test.copy()
        if 'label' not in df.columns:
            df['label'] = -1
        self.test_df = df.reset_index(drop=True)

    def reset(self, use_train=True):
        """
        Reset the environment. If use_train=True, load from self.train_df,
        else load from self.test_df.

        We then build self.states_list (sliding windows),
        set timeseries_curser = n_steps, and return the initial state.
        """
        self.use_train_data = use_train

        if self.use_train_data:
            if self.train_df is None:
                raise ValueError("train_df is None. Call set_train_data(...) first.")
            self.timeseries = self.train_df.copy()
        else:
            if self.test_df is None:
                raise ValueError("test_df is None. Call set_test_data(...) first.")
            self.timeseries = self.test_df.copy()

        # Build states
        self.states_list = self._build_all_states(self.timeseries)
        # Start the cursor at n_steps
        self.timeseries_curser = self.n_steps

        if self.timeseries_curser < len(self.states_list):
            return self.states_list[self.timeseries_curser]
        else:
            # If dataset smaller than n_steps, return zero
            return np.zeros((self.n_steps, 1), dtype=np.float32)

    def _build_all_states(self, df):
        """
        Build a list of shape-(n_steps,1) states for each time index in df.
        For index i < n_steps, we zero-pad the front.
        """
        states = []
        n_total = len(df)
        for i in range(n_total):
            if i < self.n_steps:
                pad_len = self.n_steps - i
                front_pad = np.zeros(pad_len, dtype=np.float32)
                val_part = df['value'].values[:i]
                st_1d = np.concatenate([front_pad, val_part], axis=0)
            else:
                st_1d = df['value'].values[i - self.n_steps: i]
            st_2d = st_1d.reshape(self.n_steps, 1)
            states.append(st_2d)
        return states

    def step(self, action):
        """
        Return (next_state, reward_array, done, info).

        We return a 2-element list for next_state => [st, st],
        so your RL code can do next_state[action].

        reward_array = [r_if_ACTION0, r_if_ACTION1].
        We'll fill it based on comparing 'anomaly' vs the chosen action.

        done=1 if we pass beyond data length, else 0.
        info={}
        """
        n_total = len(self.timeseries)
        if self.timeseries_curser >= n_total:
            # Already done
            done = 1
            # final state => zeros
            final_st = np.zeros((self.n_steps, 1), dtype=np.float32)
            reward_array = [0.0, 0.0]
            return [final_st, final_st], reward_array, done, {}

        # ground truth label for current index
        true_label = int(self.timeseries.at[self.timeseries_curser, 'anomaly'])
        # compare to action => +1 if correct, -1 if wrong (simple version).
        r_value = 1.0 if (true_label == action) else -1.0

        # reward array => place r_value in the chosen action index
        reward_array = [0.0, 0.0]
        reward_array[action] = r_value

        # Move forward
        self.timeseries_curser += 1
        done = 1 if (self.timeseries_curser >= n_total) else 0

        if not done:
            st = self.states_list[self.timeseries_curser]
        else:
            st = np.zeros((self.n_steps, 1), dtype=np.float32)

        # Return next_state as [st, st], so your RL code can do next_state[action].
        return [st, st], reward_array, done, {}

    def get_states_list(self):
        """
        Returns the entire list of shape-(n_steps,1) states if needed.
        Typically not used if you do step-by-step.
        """
        return self.states_list

    @property
    def datasetidx(self):
        """
        Some legacy references in your RL code might try to read env.datasetidx.
        We'll just return 0 if we're in train mode, 1 if in test mode.
        """
        return 0 if self.use_train_data else 1

    @property
    def datasetrng(self):
        """
        Similarly, we can say we have '2' datasets (train+test).
        Or just return 1 if you only want to treat them as one.
        """
        return 2

    @property
    def datasetfix(self):
        """
        If your code references this, we can store a variable or return a default.
        """
        return 0

    @datasetfix.setter
    def datasetfix(self, val):
        pass  # ignore or store if needed

    @property
    def timeseries_curser_init(self):
        """
        For code that references this, we can define n_steps as the init cursor.
        """
        return self.n_steps

    @timeseries_curser_init.setter
    def timeseries_curser_init(self, val):
        # ignore or store if needed
        pass

    @property
    def datasetsize(self):
        """
        For code that references env.datasetsize, just return 2 (train+test).
        Or return 1 if you only want a single dataset concept.
        """
        if self.train_df is not None and self.test_df is not None:
            return 2
        elif self.train_df is not None or self.test_df is not None:
            return 1
        else:
            return 0
