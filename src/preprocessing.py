from data_loader import load_raw_data
from pathlib import Path



def preprocess_data(df):

    orig_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    clean = df[orig_cols].drop_duplicates().sort_values('Time').reset_index(drop=True)

    print("Before:", df.shape[0], "| After:", clean.shape[0])
    print("Fraud before:", (df.Class == 1).sum(), "| Fraud after:", (clean.Class == 1).sum())
    print("Fraud % after:", round(clean.Class.mean() * 100, 4))   
    print("\nClass distribution after removing duplicates:")
    print(clean["Class"].value_counts())

        # Save preprocessed data
    output_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "creditcard_intermediate.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean.to_csv(output_path, index=False)

    print(f"Preprocessed data saved to: {output_path}")
    print(f"Rows: {len(clean)}")

    return df


if __name__ == "__main__":
    raw_df = load_raw_data()

    clean_df = preprocess_data(raw_df)

    print("\nFinal preprocessed dataset shape:", clean_df.shape)

    print("\nClass distribution:")
    print(clean_df["Class"].value_counts())