import pandas as pd

label_path = "WaDi/WADI.A2_19 Nov 2019/WADI_14days_new.csv"
label_df = pd.read_csv(label_path, sep=",", header=None)


df_14days = pd.read_csv(label_path, sep=",", header=None)
print("First 20 rows of WADI_14days_new.csv:")
print(df_14days.head(20))

print("\n" + "="*80 + "\n")

# Read the second CSV (WADI_attackdataLABLE.csv) and show first 20 rows

label_path2 = "WaDi/WADI.A2_19 Nov 2019/WADI_attackdataLABLE.csv"
df_attack_label = pd.read_csv(label_path2, sep=",", header=None)
print("First 20 rows of WADI_attackdataLABLE.csv:")
print(df_attack_label.head(20))