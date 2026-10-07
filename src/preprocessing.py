from data_loader import load_raw_data


def preprocess_data(df):
    # Remove exact duplicate rows
    df = df.drop_duplicates()

    print("Dataset shape after removing duplicates:", df.shape)

    print("\nClass distribution after removing duplicates:")
    print(df["Class"].value_counts())

    return df


if __name__ == "__main__":
    raw_df = load_raw_data()

    clean_df = preprocess_data(raw_df)

    print("\nFinal preprocessed dataset shape:", clean_df.shape)

    print("\nClass distribution:")
    print(clean_df["Class"].value_counts())