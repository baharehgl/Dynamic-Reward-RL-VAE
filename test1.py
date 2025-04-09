import pandas as pd

label_path = "SMD/ServerMachineDataset/test_label/machine-1-1.txt"
label_df = pd.read_csv(label_path, sep=",", header=None)
print(label_df.shape)
print(label_df.head(10))