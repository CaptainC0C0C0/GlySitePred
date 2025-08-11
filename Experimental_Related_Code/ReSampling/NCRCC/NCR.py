import pandas as pd
from imblearn.under_sampling import NeighbourhoodCleaningRule

# Read the CSV file
df = pd.read_csv('../../Feature_Extraction/Pos_and_Neg_train_3572+74042.csv')

# Extract feature columns and label column
X = df.drop('Label', axis=1)  # Features
y = df['Label']  # Labels

# Display the count of each label before undersampling
print("Label counts before undersampling:")
print(y.value_counts())

# Create NeighbourhoodCleaningRule object
ncr = NeighbourhoodCleaningRule(sampling_strategy='auto',  # Default proportional undersampling
                                n_neighbors=5)  # Number of neighbors set to 5

# Apply NeighbourhoodCleaningRule to undersample the entire dataset
X_resampled, y_resampled = ncr.fit_resample(X, y)

# Combine resampled features and labels into a single DataFrame
resampled_data = pd.DataFrame(X_resampled, columns=X.columns)

# Use pd.concat to merge the label column, avoiding data fragmentation
resampled_data = pd.concat([resampled_data, pd.Series(y_resampled, name='Label')], axis=1)

# Display the count of each label after undersampling
print("\nLabel counts after undersampling:")
print(y_resampled.value_counts())

# Save the resampled dataset
resampled_data.to_csv('esm2_Neg_features_train_ncl.csv', index=False)

print("Undersampling completed, saved as 'esm2_Neg_features_train_ncl.csv'.")