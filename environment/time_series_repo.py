# environment/time_series_repo.py

import os
import pandas as pd
import numpy as np


class EnvTimeSeriesfromRepo:
    """
    Custom Environment for Time Series Data from Repository.
    """

    def __init__(self, repodir, n_steps=50):
        """
        Initializes the environment by loading time series data from CSV files.

        Args:
            repodir (str): Directory containing CSV files.
            n_steps (int): Number of steps in each state.
        """
        self.repodir = repodir
        self.n_steps = n_steps
        self.datasets = self.load_datasets()
        self.datasetsize = len(self.datasets)
        self.dataset_idx = 0
        self.timeseries = None
        self.states_list = []

    def load_datasets(self):
        """
        Loads all CSV files from the repository directory.

        Returns:
            list of pd.DataFrame: List containing all loaded datasets.
        """
        if not os.path.exists(self.repodir):
            raise FileNotFoundError(f"Repository directory {self.repodir} does not exist.")

        all_files = [os.path.join(self.repodir, fname) for fname in os.listdir(self.repodir) if fname.endswith('.csv')]
        if not all_files:
            raise FileNotFoundError(f"No CSV files found in directory {self.repodir}.")

        datasets = []
        for file in all_files:
            try:
                df = pd.read_csv(file)
                if 'value' not in df.columns or 'anomaly' not in df.columns:
                    print(f"Skipping {file}: Missing 'value' or 'anomaly' columns.")
                    continue
                df = df[['value', 'anomaly']].dropna()
                if df.shape[0] < self.n_steps:
                    print(f"Skipping {file}: Less than {self.n_steps} rows after cleaning.")
                    continue
                datasets.append(df)
                print(f"Loaded dataset {file} with {df.shape[0]} rows.")
            except Exception as e:
                print(f"Error loading {file}: {e}")

        if not datasets:
            raise ValueError(
                f"No valid datasets found in {self.repodir}. Ensure CSV files have 'value' and 'anomaly' columns with at least {self.n_steps} rows.")

        print(f"Total datasets loaded: {len(datasets)}")
        return datasets

    def reset(self):
        """
        Resets the environment to the initial state.

        Returns:
            np.ndarray: The first state.
        """
        self.dataset_idx = 0  # Reset to the first dataset
        self.timeseries = self.load_timeseries(self.dataset_idx)
        print(f"Environment reset to dataset index {self.dataset_idx}. Timeseries length: {len(self.timeseries)}")
        self.states_list = self.get_states_list()
        return self.states_list[0] if self.states_list else None

    def load_timeseries(self, idx):
        """
        Loads the time series data for the given dataset index.

        Args:
            idx (int): Index of the dataset to load.

        Returns:
            pd.DataFrame: The loaded time series data.
        """
        if idx < 0 or idx >= self.datasetsize:
            raise IndexError(f"Dataset index {idx} out of range. Total datasets: {self.datasetsize}")
        print(f"Loading dataset index {idx}")
        return self.datasets[idx].reset_index(drop=True)

    def get_states_list(self):
        """
        Generates a list of states from the timeseries data.

        Returns:
            list of np.ndarray: List containing all generated states.
        """
        state_list = []
        for cursor in range(len(self.timeseries)):
            if cursor < self.n_steps:
                # Not enough data to form a state
                continue
            if not state_list:
                # Initialize the first state without a previous state
                state = self.statefnc(self.timeseries, cursor, previous_state=None, action=None)
                print(f"Initializing first state at cursor {cursor}.")
            else:
                # Use the last state in the list
                state = self.statefnc(self.timeseries, cursor, previous_state=state_list[-1], action=None)
            state_list.append(state)
            print(f"State {len(state_list)} generated at cursor {cursor}.")
        print(f"Total states generated: {len(state_list)}")
        return state_list

    def statefnc(self, timeseries, cursor, previous_state=None, action=None):
        """
        Function to generate the next state based on the current action and previous state.

        Args:
            timeseries (pd.DataFrame): The time series data.
            cursor (int): Current position in the time series.
            previous_state (np.ndarray or None): The previous state.
            action (int or None): The action taken.

        Returns:
            np.ndarray: The new state.
        """
        if previous_state is None:
            # Initialize the state with the first n_steps data points
            state = timeseries[['value', 'anomaly']].values[cursor - self.n_steps:cursor]
            print(f"Created initial state with shape {state.shape}.")
        else:
            # Update the state based on the action
            # For simplicity, assume state shifts and appends new data point
            state = np.roll(previous_state, -1, axis=0)
            state[-1] = timeseries[['value', 'anomaly']].values[cursor]
            print(f"Updated state at cursor {cursor} with shape {state.shape}.")
        return state

    def step(self, action):
        """
        Applies the given action to the environment and returns the next state and reward.

        Args:
            action (int): The action to take.

        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Implement the environment's response to the action.
        # This is a placeholder and should be replaced with actual logic.
        # For example purposes, we'll assume a random next state and reward.

        # Find the current state index based on states_list
        # This requires tracking the current position in states_list
        # For simplicity, let's assume that 'state_list' progresses linearly

        # Placeholder implementation:
        # Determine the index of the current state
        current_state_index = self.states_list.index(self.current_state) if hasattr(self, 'current_state') else 0

        if current_state_index + 1 >= len(self.states_list):
            # Reached the end of the states_list
            done = True
            next_state = None
            reward = 0
            print("Reached the end of the states_list.")
        else:
            # Move to the next state
            next_state = self.states_list[current_state_index + 1]
            self.current_state = next_state
            done = False

            # Placeholder reward logic
            # Replace this with your actual reward function
            reward = 1 if action == 1 else 0  # Example: reward for taking action '1'

            print(f"Took action {action}. Reward: {reward}. Next state index: {current_state_index + 1}")

        return next_state, reward, done, {}

