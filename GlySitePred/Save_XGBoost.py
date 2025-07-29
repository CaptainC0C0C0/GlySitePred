import xgboost as xgb
import pandas as pd
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# Load CSV data
def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Label']).values  # Features
    y = df['Label'].values
    return X, y

# Model training and evaluation
def train_and_evaluate(train_file, val_file, num_epochs=100, batch_size=32, lr=0.001, model_save_path='xgboost_model.json'):
    # Load training and validation sets
    X_train, y_train = load_data(train_file)
    X_val, y_val = load_data(val_file)

    # Initialize XGBoost model
    model = xgb.XGBClassifier(
        objective='binary:logistic',  # Binary classification
        eval_metric='logloss',  # Evaluation metric
        learning_rate=lr,
        max_depth=4,  # Maximum tree depth
        n_estimators=num_epochs,  # Number of trees
        use_label_encoder=False,  # Disable label encoder
        verbosity=1,  # Control output verbosity
        subsample=0.8,  # Random sampling of data
        colsample_bytree=0.9,  # Features used per tree
        #tree_method='hist',  # Use efficient histogram algorithm for CPU or GPU
        device='cuda',  # Use GPU for training
        #booster='dart'  # Use DART (Dropout Additive Regression Trees)
    )

    # Train the model
    model.fit(X_train, y_train)

    # Save the XGBoost model to a file
    model.save_model(model_save_path)
    print(f'Model saved to {model_save_path}')

    # Predict results
    y_pred = model.predict(X_val)
    y_probs = model.predict_proba(X_val)[:, 1]  # Get probabilities for the positive class

    # Calculate evaluation metrics
    acc = accuracy_score(y_val, y_pred)  # Accuracy (ACC)
    mcc = matthews_corrcoef(y_val, y_pred)  # Matthews Correlation Coefficient (MCC)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()  # Confusion matrix
    sn = tp / (tp + fn)  # Sensitivity (Recall)
    sp = tn / (tn + fp)  # Specificity
    auc = roc_auc_score(y_val, y_probs)  # AUC

    # Print evaluation results
    print("\nEvaluation Results:")
    print(f'Accuracy (ACC): {acc * 100:.2f}%')
    print(f'Matthews Correlation Coefficient (MCC): {mcc:.4f}')
    print(f'Sensitivity (SN): {sn:.4f}')
    print(f'Specificity (SP): {sp:.4f}')
    print(f'AUC: {auc:.4f}')

if __name__ == '__main__':
    # Set paths for training and validation sets
    train_file = 'Resampling/Under/esm2+ProstT5_features_train_ncl_cluster.csv'  # Training set CSV file path
    val_file = 'Resampling/esm2+ProstT5_features_val_400_2.csv'  # Validation set CSV file path
    model_save_path = 'xgboost_model.json'  # Path to save the model

    # Train and evaluate the model
    train_and_evaluate(train_file, val_file, num_epochs=2000, batch_size=32, lr=0.0001, model_save_path=model_save_path)