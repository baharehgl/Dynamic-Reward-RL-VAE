import pandas as pd
import numpy as np

NOT_ANOMALY = 0
ANOMALY = 1
action_space = [NOT_ANOMALY, ANOMALY]

class EnvTimeSeriesfromRepo:
    """
    An environment that uses data from set_train_data(...) or set_test_data(...),
    and builds one sliding-window state for every 'stride' rows instead of every row.
    This reduces states from e.g. 6 million to ~600k if stride=10.
    """

    def __init__(self, n_steps=25, stride=10):
        """
        Args:
            n_steps (int): size of the sliding window for each state.
            stride (int): skip factor; e.g., 10 => keep 1 row in every 10.
        """
        self._n_steps = n_steps
        self._stride  = stride

        self._train_df = None
        self._test_df  = None

        self._datasetfix = 0
        self._datasetidx = 0
        self._timeseries_curser_init = n_steps

        self.timeseries = None
        self.states_list = []
        self.timeseries_curser = 0

        self.action_space_n = len(action_space)

    # ----------------------------------------------------------------
    # Properties (to avoid "can't set attribute" errors)
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
        has_train = (self._train_df is not None)
        has_test  = (self._test_df  is not None)
        if has_train and has_test:
            return 2
        elif has_train or has_test:
            return 1
        else:
            return 0

    @property
    def datasetrng(self):
        return self.datasetsize

    # ----------------------------------------------------------------
    # Setting data
    # ----------------------------------------------------------------
    def set_train_data(self, x_train):
        df = self._convert_to_df(x_train, is_train=True)
        self._train_df = df

    def set_test_data(self, x_test):
        df = self._convert_to_df(x_test, is_train=False)
        self._test_df = df

    def _convert_to_df(self, data, is_train=True):
        if isinstance(data, np.ndarray):
            if len(data.shape) == 1:
                data = data.reshape(-1,1)
            df = pd.DataFrame(data, columns=['value'])
            df['anomaly'] = 0
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            if 'value' not in df.columns:
                raise ValueError("DataFrame must have a 'value' column.")
            if 'anomaly' not in df.columns:
                print("Warning: 'anomaly' column not found, set anomaly=0.")
                df['anomaly'] = 0
        else:
            raise ValueError("Data must be NumPy or DataFrame with 'value' col.")

        if 'label' not in df.columns:
            df['label'] = -1
        return df.reset_index(drop=True)

    # ----------------------------------------------------------------
    # Reset & Step
    # ----------------------------------------------------------------
    def reset(self):
        if self.datasetsize == 0:
            raise ValueError("No train/test data set. Call set_train_data(...) / set_test_data(...) first.")

        if self._datasetidx == 0:
            if self._train_df is None:
                raise ValueError("Train data not set, but datasetidx=0.")
            self.timeseries = self._train_df.copy()
        else:
            if self._test_df is None:
                raise ValueError("Test data not set, but datasetidx=1.")
            self.timeseries = self._test_df.copy()

        self.states_list = self._build_all_states(self.timeseries)
        self.timeseries_curser = self._timeseries_curser_init

        if self.timeseries_curser < len(self.states_list):
            return self.states_list[self.timeseries_curser]
        else:
            # Not enough data
            return np.zeros((self._n_steps,1), dtype=np.float32)

    def _build_all_states(self, df):
        """
        Build a list of shape-(n_steps,1) states by skipping with self._stride.
        E.g., if stride=10, we take row 0,10,20,... as 'end' of a sliding window.
        Avoiding 6 M states
        """
        n_total = len(df)
        states = []
        for i in range(0, n_total, self._stride):
            if i < self._n_steps:
                pad_len = self._n_steps - i
                front_pad = np.zeros(pad_len, dtype=np.float32)
                val_part  = df['value'].values[:i]
                st_1d = np.concatenate([front_pad, val_part], axis=0)
            else:
                st_1d = df['value'].values[i - self._n_steps : i]
            st_2d = st_1d.reshape(self._n_steps, 1)
            states.append(st_2d)
        return states

    def step(self, action):
        n_total = len(self.states_list)
        if self.timeseries_curser >= n_total:
            done = 1
            final_st = np.zeros((self._n_steps,1), dtype=np.float32)
            reward_array = [0.0, 0.0]
            return [final_st, final_st], reward_array, done, {}

        # Convert self.timeseries_curser -> the corresponding row index in df
        # For example, if stride=10, index in states_list = i => actual row in df is i*self._stride.
        df_row = self.timeseries_curser * self._stride
        if df_row >= len(self.timeseries):
            # If for some reason this surpasses the actual data length
            done = 1
            final_st = np.zeros((self._n_steps,1), dtype=np.float32)
            reward_array = [0.0, 0.0]
            return [final_st, final_st], reward_array, done, {}

        true_label = int(self.timeseries.at[df_row, 'anomaly'])
        r_value = 1.0 if (true_label == action) else -1.0

        reward_array = [0.0, 0.0]
        reward_array[action] = r_value

        self.timeseries_curser += 1
        done = 1 if (self.timeseries_curser >= n_total) else 0

        if not done:
            st = self.states_list[self.timeseries_curser]
        else:
            st = np.zeros((self._n_steps,1), dtype=np.float32)

        return [st, st], reward_array, done, {}

    def get_states_list(self):
        return self.states_list
