import xgboost as xgb
import pandas as pd
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold
import numpy as np

# Load CSV data
def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Label']).values  # Features
    y = df['Label'].values  # Labels
    return X, y

# Train and evaluate model with cross-validation
def train_and_evaluate_cross_val(file_path, num_epochs=300, batch_size=32, lr=0.001, n_splits=10):
    # Load data
    X, y = load_data(file_path)

    # Initialize K-Fold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Lists to store evaluation metrics
    accuracies = []
    mccs = []
    sensitivities = []
    specificities = []
    aucs = []

    # Perform K-Fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}/{n_splits}:")

        # Get training and validation sets
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Initialize XGBoost model
        model = xgb.XGBClassifier(
            objective='binary:logistic',  # Binary classification
            eval_metric='logloss',  # Evaluation metric
            learning_rate=lr,
            max_depth=5,  # Maximum tree depth
            n_estimators=num_epochs,  # Number of trees
            use_label_encoder=False,  # Disable label encoder
            verbosity=1,  # Control output verbosity
            subsample=0.8,  # Random sampling
            colsample_bytree=0.9,  # Features used per tree
            # tree_method='hist',  # Use efficient histogram algorithm for CPU or GPU
            # device='cuda',  # Use GPU for training
            # booster='dart'  # Use DART (Dropout Additive Regression Trees)
        )

        # Train the model
        model.fit(X_train, y_train)

        # Predict results
        y_pred = model.predict(X_val)
        y_probs = model.predict_proba(X_val)[:, 1]  # Get probabilities for positive class

        # Calculate evaluation metrics
        acc = accuracy_score(y_val, y_pred)  # Accuracy (ACC)
        mcc = matthews_corrcoef(y_val, y_pred)  # Matthews Correlation Coefficient (MCC)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()  # Confusion matrix
        sn = tp / (tp + fn)  # Sensitivity (Recall)
        sp = tn / (tn + fp)  # Specificity
        auc = roc_auc_score(y_val, y_probs)  # AUC

        # Store metrics for current fold
        accuracies.append(acc)
        mccs.append(mcc)
        sensitivities.append(sn)
        specificities.append(sp)
        aucs.append(auc)

        # Print evaluation results for current fold
        print(f'Accuracy (ACC): {acc * 100:.2f}%')
        print(f'Matthews Correlation Coefficient (MCC): {mcc:.4f}')
        print(f'Sensitivity (SN): {sn:.4f}')
        print(f'Specificity (SP): {sp:.4f}')
        print(f'AUC: {auc:.4f}')

    # Calculate average evaluation results
    print("\nCross-validation Results (average):")
    print(f'Average Accuracy (ACC): {np.mean(accuracies) * 100:.2f}%')
    print(f'Average Matthews Correlation Coefficient (MCC): {np.mean(mccs):.4f}')
    print(f'Average Sensitivity (SN): {np.mean(sensitivities):.4f}')
    print(f'Average Specificity (SP): {np.mean(specificities):.4f}')
    print(f'Average AUC: {np.mean(aucs):.4f}')

if __name__ == '__main__':
    # Set dataset path
    file_path = '../ReSampling/NCRCC/esm2_features_train_NCRCC.csv'  # Dataset CSV file path (NCRCC)

    # Train and evaluate using 10-fold cross-validation
    train_and_evaluate_cross_val(file_path, num_epochs=2000, batch_size=32, lr=0.0001)