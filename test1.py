import pandas as pd

path = "SMD/ServerMachineDataset/train/machine-1-1.txt"
df = pd.read_csv(path, sep=",", header=None)  # if there's no header
print(df.shape)       # see rows x columns
print(df.head(10))    # see first 10 rows