import lime
import lime.lime_tabular
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Label']).values  # Features
    y = df['Label'].values  # Labels
    return X, y

# Train XGBoost model
def train_xgboost_model(train_file):
    X_train, y_train = load_data(train_file)

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        learning_rate=0.001,
        max_depth=4,
        n_estimators=2000,
        use_label_encoder=False,
        verbosity=1
    )

    model.fit(X_train, y_train)
    return model, X_train, y_train

# Use LIME for local explanation
def lime_explanation(model, X_train, instance_index):
    # Initialize LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=[f'Feature {i}' for i in range(X_train.shape[1])],
        class_names=['Class 0', 'Class 1'],  # Class names
        mode='classification'  # Set mode as classification
    )

    # Select an instance to explain
    instance = X_train[instance_index]

    # Generate LIME explanation for the instance's prediction
    explanation = explainer.explain_instance(instance, model.predict_proba, num_features=5)

    # Plot LIME results using matplotlib
    features, importances = zip(*explanation.as_list())
    plt.figure(figsize=(10, 6))
    plt.barh(features, importances, color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title(f'LIME Explanation for Instance {instance_index}')
    plt.show()


if __name__ == '__main__':
    train_file = '../Feature_Stitching/After_Stitching/ESM2_ProstT5_train_NCRCC.csv'  # Path to training CSV file

    # Train XGBoost model
    model, X_train, y_train = train_xgboost_model(train_file)

    # Choose a sample instance for LIME explanation
    instance_index = 10  # You can select any index to see explanation for that sample
    lime_explanation(model, X_train, instance_index)
