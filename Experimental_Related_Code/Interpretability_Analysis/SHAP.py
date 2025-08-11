import shap
import xgboost as xgb
import pandas as pd

# Load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Label']).values  # Features
    y = df['Label'].values  # Labels
    return X, y

# Train model and compute SHAP values
def train_and_explain(train_file, val_file):
    # Load data
    X_train, y_train = load_data(train_file)
    X_val, y_val = load_data(val_file)

    # Initialize XGBoost model
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        learning_rate=0.001,
        max_depth=4,
        n_estimators=2000,
        use_label_encoder=False,
        verbosity=1
    )

    # Train model
    model.fit(X_train, y_train)

    # Explain model using SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    # Visualize SHAP values
    shap.summary_plot(shap_values, X_val)

if __name__ == '__main__':
    train_file = '../Feature_Stitching/After_Stitching/ESM2_ProstT5_train_NCRCC.csv'  # Training dataset CSV path
    val_file = '../Feature_Stitching/ESM2_ProstT5_test.csv'      # Validation dataset CSV path

    # Train and explain the model
    train_and_explain(train_file, val_file)
