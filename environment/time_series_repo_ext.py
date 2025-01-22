import pandas as pd
import numpy as np
import random

NOT_ANOMALY = 0
ANOMALY = 1
action_space = [NOT_ANOMALY, ANOMALY]


class EnvTimeSeriesfromRepo:
    """
    A custom environment that uses data set by set_train_data(...) or set_test_data(...).
    We do NOT load CSV files from disk, so you won't get "No .csv files found...".

    This version also supports datasetfix, datasetidx, timeseries_curser_init properties,
    because your RL code might do env.datasetfix=0, env.datasetidx=..., env.timeseries_curser_init=..., etc.
    """

    def __init__(self, n_steps=25):
        """
        Args:
            n_steps (int): sliding window length for RNN states.
        """
        self._train_df = None
        self._test_df = None

        # By default, we start with the "train" data
        self.use_train_data = True

        # We'll store the active timeseries DataFrame as we "reset()"
        self.timeseries = None
        # We'll store all sliding-window states in a list
        self.states_list = []

        # Internal defaults
        self._n_steps = n_steps
        self._datasetfix = 0
        self._datasetidx = 0
        self._timeseries_curser_init = n_steps

        self.timeseries_curser = 0  # actual dynamic cursor while stepping
        self.action_space_n = len(action_space)

    # ----------------------------------------------------------------
    # Properties to prevent "can't set attribute" errors
    # ----------------------------------------------------------------

    @property
    def datasetfix(self):
        return self._datasetfix

    @datasetfix.setter
    def datasetfix(self, val):
        self._datasetfix = val

    @property
    def datasetidx(self):
        return self._datasetidx

    @datasetidx.setter
    def datasetidx(self, val):
        self._datasetidx = val

    @property
    def timeseries_curser_init(self):
        return self._timeseries_curser_init

    @timeseries_curser_init.setter
    def timeseries_curser_init(self, val):
        self._timeseries_curser_init = val

    @property
    def datasetsize(self):
        """
        Some RL scripts reference env.datasetsize to loop over multiple datasets.
        We'll say '2' if we have train+test, '1' if only train or only test, or '0' if none.
        """
        has_train = (self._train_df is not None)
        has_test = (self._test_df is not None)
        if has_train and has_test:
            return 2
        elif has_train or has_test:
            return 1
        else:
            return 0

    @property
    def datasetrng(self):
        """
        Also used in some RL scripts to limit dataset index range.
        We'll say 2 if we have train+test, else 1 or 0.
        """
        return self.datasetsize

    # ----------------------------------------------------------------
    # Setting data
    # ----------------------------------------------------------------

    def set_train_data(self, x_train):
        """
        x_train can be:
          - a NumPy array (N,) or (N,1) -> we assume anomaly=0 for training
          - a pandas DataFrame with columns 'value', optionally 'anomaly' (we set to 0 if missing)
        """
        df = self._convert_to_df(x_train, is_train=True)
        self._train_df = df

    def set_test_data(self, x_test):
        """
        x_test must have 'value' and 'anomaly' columns if DataFrame,
        or if it's a NumPy array, we force 'anomaly' = 0?
        But typically, for test, you have ground-truth anomalies.
        """
        df = self._convert_to_df(x_test, is_train=False)
        self._test_df = df

    def _convert_to_df(self, data, is_train=True):
        """
        Internal helper to standardize input to a DataFrame with 'value' and 'anomaly'.
        If is_train=True and no 'anomaly' col is found, set anomaly=0.
        If is_train=False but user didn't supply an anomaly col, we set anomaly=0 anyway (lack of GT).
        """
        import pandas as pd
        import numpy as np

        if isinstance(data, np.ndarray):
            # shape => (N,) or (N,1)
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            df = pd.DataFrame(data, columns=['value'])
            # set anomaly=0 if is_train or we have no GT
            df['anomaly'] = 0
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            if 'value' not in df.columns:
                raise ValueError("DataFrame must have a 'value' column.")
            if 'anomaly' not in df.columns:
                # if is_train, we assume all normal => anomaly=0
                # if test but missing anomaly, we do the same or raise?
                print("Warning: 'anomaly' column not found, setting all anomalies=0.")
                df['anomaly'] = 0
        else:
            raise ValueError("Data must be either a NumPy array or a DataFrame.")

        # Always add a 'label' = -1 column if missing
        if 'label' not in df.columns:
            df['label'] = -1

        df = df.reset_index(drop=True)
        return df

    # ----------------------------------------------------------------
    # Reset & Step methods
    # ----------------------------------------------------------------

    def reset(self):
        """
        Called at the start of an episode.
        If datasetfix=0, we might switch between train & test depending on datasetidx,
        or just always pick train.
        Typically your RL code does: env.reset() for training, or env.reset() for validation.

        We'll do a simple approach:
        - if datasetidx=0, use train data
        - if datasetidx=1, use test data
        (since we have at most 2 datasets, train/test).

        Then build states_list (sliding windows).
        timeseries_curser = timeseries_curser_init (default = n_steps).

        Return the initial state if available, else zeros.
        """
        if self.datasetsize == 0:
            raise ValueError("No train/test data set. Use set_train_data(...) or set_test_data(...) first.")

        # if we want to cycle among multiple datasets, do it here,
        # but let's keep it simple. We'll pick train if datasetidx=0, test if datasetidx=1.
        # unless user does something else in their code.
        if self._datasetidx == 0:
            if self._train_df is None:
                raise ValueError("Train data not set, but datasetidx=0.")
            self.timeseries = self._train_df.copy()
        else:
            if self._test_df is None:
                raise ValueError("Test data not set, but datasetidx=1.")
            self.timeseries = self._test_df.copy()

        # Build sliding-window states
        self.states_list = self._build_all_states(self.timeseries)
        self.timeseries_curser = self._timeseries_curser_init

        # Return initial state
        if self.timeseries_curser < len(self.states_list):
            return self.states_list[self.timeseries_curser]
        else:
            # If not enough data, return zeros
            return np.zeros((self._n_steps, 1), dtype=np.float32)

    def _build_all_states(self, df):
        """
        For each index i in df, build a window of size n_steps.
        If i < n_steps, front-pad with zeros.
        Return a list of shape-(n_steps,1) arrays.
        """
        n_total = len(df)
        states = []
        for i in range(n_total):
            if i < self._n_steps:
                pad_len = self._n_steps - i
                front_pad = np.zeros(pad_len, dtype=np.float32)
                val_part = df['value'].values[:i]
                st_1d = np.concatenate([front_pad, val_part], axis=0)
            else:
                st_1d = df['value'].values[i - self._n_steps: i]
            st_2d = st_1d.reshape(self._n_steps, 1)
            states.append(st_2d)
        return states

    def step(self, action):
        """
        Return (next_state_list, reward_array, done, info),
        where next_state_list is [st, st] so your RL can do next_state[action].

        We'll do a simple reward: +1 if action == anomaly, else -1.
        Then store that in reward_array[action].
        """
        n_total = len(self.timeseries)
        if self.timeseries_curser >= n_total:
            # already done
            done = 1
            final_st = np.zeros((self._n_steps, 1), dtype=np.float32)
            reward_array = [0.0, 0.0]
            return [final_st, final_st], reward_array, done, {}

        # ground truth anomaly
        true_label = int(self.timeseries.at[self.timeseries_curser, 'anomaly'])
        if true_label == action:
            r_value = 1.0
        else:
            r_value = -1.0

        reward_array = [0.0, 0.0]
        reward_array[action] = r_value

        # increment cursor
        self.timeseries_curser += 1
        done = 1 if (self.timeseries_curser >= n_total) else 0

        if not done:
            st = self.states_list[self.timeseries_curser]
        else:
            st = np.zeros((self._n_steps, 1), dtype=np.float32)

        return [st, st], reward_array, done, {}

    def get_states_list(self):
        """
        Returns the entire list of shape-(n_steps,1) states.
        Typically not needed if you do step by step.
        """
        return self.states_list
