import pandas as pd

# Read the first CSV (WADI_14days_new.csv) and show first 20 rows
df_14days = pd.read_csv("WADI_14days_new.csv")
print("First 20 rows of WADI_14days_new.csv:")
print(df_14days.head(20))

print("\n" + "="*80 + "\n")

# Read the second CSV (WADI_attackdataLABLE.csv) and show first 20 rows
df_attack_label = pd.read_csv("WADI_attackdataLABLE.csv")
print("First 20 rows of WADI_attackdataLABLE.csv:")
print(df_attack_label.head(20))