import pandas as pd


# Load data
df = pd.read_csv("data/creditcard_intermediate.csv")

# Sort transactions chronologically
df = df.sort_values("Time").reset_index(drop=True)

# Define split points
train_end = int(len(df) * 0.70)
val_end = int(len(df) * 0.85)

# Time-based split
train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

# Check sizes
print("Train shape:", train_df.shape)
print("Validation shape:", val_df.shape)
print("Test shape:", test_df.shape)

# Check fraud counts
print("\nFraud counts:")
print("Train:", train_df["Class"].value_counts())
print("Validation:", val_df["Class"].value_counts())
print("Test:", test_df["Class"].value_counts())

# Save splits
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/validation.csv", index=False)
test_df.to_csv("data/test.csv", index=False)