import os
import pandas as pd

def preview_smd_data(base_dir):
    """
    Recursively scans the base_dir (where the SMD dataset is located),
    finds CSV files, and prints out a quick preview:
      - Full path of the CSV
      - Column names (header)
      - First few rows (head)
    """
    print(f"Scanning directory: {base_dir}")
    files_found = False
    # Walk through all subdirectories
    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            if filename.lower().endswith('.csv'):
                files_found = True
                file_path = os.path.join(root, filename)
                print("=" * 80)
                print(f"CSV File: {file_path}")
                try:
                    df = pd.read_csv(file_path)
                    print("Columns:", df.columns.tolist())
                    print("Shape:", df.shape)
                    print("Head:\n", df.head(5))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    if not files_found:
        print("No CSV files were found in the specified directory.")

# Example usage:
if __name__ == "__main__":
    # Update this path to point to your SMD dataset folder
    smd_base_dir = r"C:\Users\robotics\Documents\GitHub\Adaptive-Reward-Scaling-Reinforcement-Learning\SMD\ServerMachineDataset"
    preview_smd_data(smd_base_dir)
