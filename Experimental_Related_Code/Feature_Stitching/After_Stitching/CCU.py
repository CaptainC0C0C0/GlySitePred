import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from imblearn.under_sampling import ClusterCentroids
from collections import Counter

# Function to read CSV file and perform ClusterCentroids undersampling
def UnderClusterCentroids_CSV(input_file, output_file, label_column, cluster_strategy, n_init, voting):
    """
    Read data from a CSV file, perform ClusterCentroids undersampling, and save the result.

    Parameters:
    - input_file: Path to the input CSV file
    - output_file: Path to save the undersampled CSV file
    - label_column: Name of the label column
    - cluster_strategy: Dictionary specifying the target sample count for each label
    - n_init: Number of initializations for MiniBatchKMeans
    - voting: Voting strategy for ClusterCentroids
    """
    # Read data
    df = pd.read_csv(input_file)

    # Extract feature and label columns
    X = df.drop(label_column, axis=1)  # Features
    y = df[label_column]  # Labels

    # Display label counts before undersampling
    print("Label counts before undersampling:")
    print(y.value_counts())

    # Perform undersampling
    X_resampled, y_resampled = Cluster_Centroids(X, y, cluster_strategy, n_init, voting)

    # Combine resampled features and labels into a single DataFrame
    resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
    resampled_data = pd.concat([resampled_data, pd.Series(y_resampled, name=label_column)], axis=1)

    # Display label counts after undersampling
    print("\nLabel counts after undersampling:")
    print(Counter(y_resampled))

    # Save the undersampled dataset
    resampled_data.to_csv(output_file, index=False)
    print(f"Undersampling completed, saved as '{output_file}'.")

# ClusterCentroids function implementation
def Cluster_Centroids(X, y, sampling_strategy, n_init, voting):
    """
    Perform undersampling using ClusterCentroids.

    Parameters:
    - X: Feature data
    - y: Label data
    - sampling_strategy: Dictionary specifying the target sample count for each label
    - n_init: Number of initializations for MiniBatchKMeans
    - voting: Voting strategy for ClusterCentroids
    """
    y = y.ravel()
    under = ClusterCentroids(
        sampling_strategy=sampling_strategy,
        random_state=1,
        voting=voting,
        estimator=MiniBatchKMeans(n_init=n_init, random_state=1, batch_size=2048)
    )
    x_resampled, y_resampled = under.fit_resample(X, y)
    return x_resampled, y_resampled

# Example usage
if __name__ == "__main__":
    # Input file path
    file1_path = 'ESM2_ProstT5_train_ncl.csv'
    output_csv = 'ESM2_ProstT5_train_NCRCC.csv'

    # Specify label column name
    label_col = 'Label'

    # Specify the target sample count for each class
    cluster_sampling_strategy = {0: 3572}  # Adjust the sample count for each class as needed

    # ClusterCentroids parameters
    n_init_param = 20
    voting_param = 'hard'  # 'hard' or 'soft'

    # Perform undersampling
    UnderClusterCentroids_CSV(file1_path, output_csv, label_col, cluster_sampling_strategy, n_init_param, voting_param)