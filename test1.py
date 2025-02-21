import pandas as pd

df = pd.read_csv("KPI_data/train/phase2_train.csv")
print(df.head(20))
print(df['anomaly'].unique())