import pandas as pd
import numpy as np

NOT_ANOMALY = 0
ANOMALY = 1
action_space = [NOT_ANOMALY, ANOMALY]


class EnvTimeSeriesfromRepo:
    """
    A custom environment that does NOT read CSV from disk, but instead uses
    data set by set_train_data(...) or set_test_data(...).

    It supports properties like datasetfix, datasetidx, timeseries_curser_init
    so your RL code can do things like:
      env.datasetfix = 0
      env.datasetidx = 0
      env.timeseries_curser_init = 25
    without error.

    The environment returns sliding-window states of shape (n_steps,1),
    and does a simple reward: +1 if your chosen action == anomaly, else -1.
    """

    def __init__(self, n_steps=25):
        """
        Args:
            n_steps (int): size of the sliding window for RNN-based state.
        """
        self._n_steps = n_steps

        # We'll store two DataFrames: train & test
        self._train_df = None
        self._test_df = None

        # By default, environment picks train if datasetidx=0, test if datasetidx=1
        self._datasetfix = 0
        self._datasetidx = 0
        self._timeseries_curser_init = n_steps

        self.timeseries = None
        self.states_list = []
        self.timeseries_curser = 0

        # The RL code may also reference these
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
        """
        For RL code that references env.datasetsize,
        we define '2' if we have train & test, '1' if only one, '0' if none.
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
        Another property for RL code referencing env.datasetrng.
        We'll say it's equal to the number of datasets we have (train+test).
        """
        return self.datasetsize

    # ----------------------------------------------------------------
    # Setting data
    # ----------------------------------------------------------------
    def set_train_data(self, x_train):
        """
        x_train: either a NumPy array (N,) or (N,1),
                 or a DataFrame with columns ['value'] & optionally 'anomaly'].
        We'll assume 'anomaly'=0 if missing for train.
        """
        df = self._convert_to_df(x_train, is_train=True)
        self._train_df = df

    def set_test_data(self, x_test):
        """
        x_test: same logic, but usually you have 'anomaly' for test.
        If missing, we set anomaly=0 and warn.
        """
        df = self._convert_to_df(x_test, is_train=False)
        self._test_df = df

    def _convert_to_df(self, data, is_train=True):
        import pandas as pd
        import numpy as np

        if isinstance(data, np.ndarray):
            # shape => (N,) or (N,1)
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            df = pd.DataFrame(data, columns=['value'])
            # For train or test with no known anomalies, set anomaly=0
            df['anomaly'] = 0
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            if 'value' not in df.columns:
                raise ValueError("DataFrame must have a 'value' column.")
            if 'anomaly' not in df.columns:
                # no anomaly column => set to 0
                print("Warning: 'anomaly' column not found, set all anomaly=0.")
                df['anomaly'] = 0
        else:
            raise ValueError("Data must be a NumPy array or a DataFrame with 'value' col.")

        if 'label' not in df.columns:
            df['label'] = -1

        return df.reset_index(drop=True)

    # ----------------------------------------------------------------
    # Reset & Step
    # ----------------------------------------------------------------
    def reset(self):
        """
        If datasetidx=0 => use _train_df
        If datasetidx=1 => use _test_df
        Then build sliding-window states.
        Set timeseries_curser = timeseries_curser_init (default = n_steps).
        Return the initial state => shape (n_steps,1) or zero if not enough rows.
        """
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
            import numpy as np
            return np.zeros((self._n_steps, 1), dtype=np.float32)

    def _build_all_states(self, df):
        """
        For each row i in df, build a (n_steps,1) sliding window.
        If i < n_steps, front-pad with zeros.
        """
        import numpy as np

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
        Return ( [next_st, next_st], [r0,r1], done, {} )
        We do +1 if action == anomaly, else -1, stored in reward[action].
        """
        import numpy as np

        n_total = len(self.timeseries)
        if self.timeseries_curser >= n_total:
            # We're past the end => done
            done = 1
            final_st = np.zeros((self._n_steps, 1), dtype=np.float32)
            reward_array = [0.0, 0.0]
            return [final_st, final_st], reward_array, done, {}

        true_label = int(self.timeseries.at[self.timeseries_curser, 'anomaly'])
        r_value = 1.0 if (true_label == action) else -1.0

        reward_array = [0.0, 0.0]
        reward_array[action] = r_value

        self.timeseries_curser += 1
        done = 1 if (self.timeseries_curser >= n_total) else 0

        if not done:
            st = self.states_list[self.timeseries_curser]
        else:
            st = np.zeros((self._n_steps, 1), dtype=np.float32)

        return [st, st], reward_array, done, {}

    def get_states_list(self):
        """
        If your code wants the full list of states, it can retrieve them here.
        Typically not needed if you do step-by-step.
        """
        return self.states_list
