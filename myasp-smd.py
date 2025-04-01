import os
import pandas as pd


def preview_smd_txt(base_dir):
    """
    Recursively scans the base_dir (where the SMD dataset is located),
    finds .txt files, and prints out a quick preview:
      - Full path of the .txt file
      - Column names (if any)
      - Shape of the DataFrame
      - First 5 rows
    """
    print(f"Scanning directory: {base_dir}")
    files_found = False

    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            # Look for .txt files
            if filename.lower().endswith('.txt'):
                files_found = True
                file_path = os.path.join(root, filename)
                print("=" * 80)
                print(f"TXT File: {file_path}")

                try:
                    # By default, we assume comma-separated.
                    # If your data is space/tab-separated, change sep to '\s+' or '\t'.
                    # If there's no header row, use header=None.
                    df = pd.read_csv(file_path, sep=',', header=None)

                    print("Shape:", df.shape)
                    # If you don't have column headers, you can optionally assign them here:
                    # df.columns = [f"col_{i}" for i in range(df.shape[1])]

                    # Print the first few rows
                    print("Head:\n", df.head(5))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    if not files_found:
        print("No .txt files were found in the specified directory.")


if __name__ == "__main__":
    # Update this path to point to your local SMD/ServerMachineDataset folder
    smd_base_dir = r"C:\Users\robotics\Documents\GitHub\Adaptive-Reward-Scaling-Reinforcement-Learning\SMD\ServerMachineDataset"
    preview_smd_txt(smd_base_dir)
