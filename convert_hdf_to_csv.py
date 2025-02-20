import pandas as pd

# Path to your HDF file
hdf_path = "phase2_ground_truth.hdf"

# 1) Optional: Check the available keys in the HDF store.
store = pd.HDFStore(hdf_path)
print("Available HDF keys:", store.keys())
store.close()

# 2) Read the dataset by specifying the correct key.
# If your HDF file has only one dataset, you might not need to specify a key.
df = pd.read_hdf(hdf_path)  # or pd.read_hdf(hdf_path, key='some_key')

# 3) Convert to CSV
df.to_csv("phase2_ground_truth.csv", index=False)

print("Conversion complete! CSV saved as phase2_ground_truth.csv")