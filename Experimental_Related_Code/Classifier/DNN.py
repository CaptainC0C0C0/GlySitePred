import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# Define DNN model
class DNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First fully connected layer
        self.fc2 = nn.Linear(128, 64)          # Second fully connected layer
        self.fc3 = nn.Linear(64, output_size)  # Output layer, size = output_size

        self.relu = nn.ReLU()                  # ReLU activation function
        self.softmax = nn.Softmax(dim=1)       # Softmax activation for the output layer

    def forward(self, x):
        x = self.relu(self.fc1(x))  # First FC layer + ReLU
        x = self.relu(self.fc2(x))  # Second FC layer + ReLU
        x = self.fc3(x)             # Output layer
        x = self.softmax(x)         # Softmax activation
        return x


# Load CSV data
def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(columns=['Label']).values  # Features
    y = df['Label'].values                 # Labels
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


# Train model with K-Fold cross-validation
def train_model_kfold(data, labels, k=10, num_epochs=300, batch_size=32, lr=0.001):
    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    auc_scores = []
    acc_scores = []
    mcc_scores = []
    sn_scores = []
    sp_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
        print(f'Fold {fold + 1}/{k}')

        # Split dataset
        X_train, X_val = data[train_idx], data[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Create DataLoaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        input_size = X_train.size(1)
        output_size = 2  # Binary classification
        model = DNN(input_size, output_size).to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Train model
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}')

        # Evaluate model
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                prob = torch.softmax(outputs, dim=1)[:, 1]  # Probability of positive class

                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                all_probs.extend(prob.cpu().numpy())

        # Accuracy (ACC)
        acc = correct / total

        # Matthews Correlation Coefficient (MCC)
        mcc = matthews_corrcoef(all_labels, all_preds)

        # Sensitivity (SN) & Specificity (SP)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        sn = tp / (tp + fn)
        sp = tn / (tn + fp)

        # AUC
        auc = roc_auc_score(all_labels, all_probs)

        # Store fold results
        auc_scores.append(auc)
        acc_scores.append(acc)
        mcc_scores.append(mcc)
        sn_scores.append(sn)
        sp_scores.append(sp)

        print(f'Validation Accuracy (ACC): {acc * 100:.2f}%')
        print(f'Matthews Correlation Coefficient (MCC): {mcc:.4f}')
        print(f'Sensitivity (SN): {sn:.4f}')
        print(f'Specificity (SP): {sp:.4f}')
        print(f'AUC: {auc:.4f}')

    # Average results
    print(f'\nAverage AUC: {np.mean(auc_scores):.4f}')
    print(f'Average Accuracy: {np.mean(acc_scores) * 100:.2f}%')
    print(f'Average MCC: {np.mean(mcc_scores):.4f}')
    print(f'Average Sensitivity (SN): {np.mean(sn_scores):.4f}')
    print(f'Average Specificity (SP): {np.mean(sp_scores):.4f}')


if __name__ == '__main__':
    # Load training data from CSV
    data_file = '../ReSampling/NCRCC/esm2_features_train_NCRCC.csv'
    X, y = load_data(data_file)

    # Run 10-fold cross-validation
    train_model_kfold(X, y, k=10, num_epochs=300, batch_size=32, lr=0.001)
