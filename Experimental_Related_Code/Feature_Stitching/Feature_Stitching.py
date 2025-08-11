import pandas as pd

def merge_and_save(file1_path, file2_path, output_path):
    """
    Merge two CSV files by concatenating their features and keep the label column from the first file.
    Save the merged data to output_path.
    """
    data1 = pd.read_csv(file1_path)
    data2 = pd.read_csv(file2_path)

    if 'Label' not in data1.columns:
        raise ValueError("Label column not found in the first file!")

    label_data = data1[['Label']]
    data1 = data1.drop(columns=['Label'])
    data2 = data2.drop(columns=['Label'])

    merged_features = pd.concat([data1, data2], axis=1)
    merged_features.columns = [str(i) for i in range(merged_features.shape[1])]
    final_data = pd.concat([merged_features, label_data], axis=1)

    final_data.to_csv(output_path, index=False)
    print(f"Data successfully merged and saved to: {output_path}")

if __name__ == "__main__":
    # Train dataset paths
    train_file1 = "../Feature_Extraction/Pos_and_Neg_train_3572+74042.csv"
    train_file2 = "../Feature_Extraction/ProstT5_Pos_and_Neg_train_3572+74042.csv"
    train_output = "ESM2_ProstT5_train_without_resampling.csv"

    # Test dataset paths
    test_file1 = "../Feature_Extraction/Pos_and_Neg_test_397+8227.csv"
    test_file2 = "../Feature_Extraction/ProstT5_Pos_and_Neg_test_397+8227.csv"
    test_output = "ESM2_ProstT5_test.csv"

    # Merge and save train dataset
    merge_and_save(train_file1, train_file2, train_output)

    # Merge and save test dataset
    merge_and_save(test_file1, test_file2, test_output)
