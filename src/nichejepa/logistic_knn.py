import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score

def prepare_dataframes(df, feature_prefix='feature_', label_column='cluster_label', num_features=192):
    if label_column not in df.columns:
        print(f"{label_column} doesn't exist")
        return None, None

    feature_columns = [f'{feature_prefix}{i}' for i in range(num_features)]
    X = df[feature_columns].values

    le = LabelEncoder()
    y = df[label_column].dropna()
    X = df.loc[y.index, feature_columns].values
    y = le.fit_transform(y)

    return X, y

def create_dataloaders(X, y, batch_size=32):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_model(model, train_loader):
    X_train, y_train = [], []
    for X_batch, y_batch in train_loader:
        X_train.extend(X_batch.numpy())
        y_train.extend(y_batch.numpy())
    model.fit(X_train, y_train)

def evaluate_model(model, data_loader):
    X, y_true = [], []
    for X_batch, y_batch in data_loader:
        X.extend(X_batch.numpy())
        y_true.extend(y_batch.numpy())
    y_pred = model.predict(X)
    return f1_score(y_true, y_pred, average='weighted'), accuracy_score(y_true, y_pred)

def run_logistic_regression(train_df, test_df, label_column, num_features=None):
    X_train, y_train = prepare_dataframes(train_df, label_column=label_column, num_features=num_features)
    X_test, y_test = prepare_dataframes(test_df, label_column=label_column, num_features=num_features)
    if X_train is None or X_test is None:
        return None
    train_loader = create_dataloaders(X_train, y_train)
    test_loader = create_dataloaders(X_test, y_test)

    model = LogisticRegression()
    train_model(model, train_loader)
    
    train_f1, train_acc = evaluate_model(model, train_loader)
    test_f1, test_acc = evaluate_model(model, test_loader)
    
    print(f"Train F1 Score: {train_f1}, Train Accuracy: {train_acc}")
    print(f"Test F1 Score: {test_f1}, Test Accuracy: {test_acc}")

    return test_acc

def run_knn_classification(train_df, test_df, label_column, num_features=None, n_neighbors=4):
    X_train, y_train = prepare_dataframes(train_df, label_column=label_column, num_features=num_features)
    X_test, y_test = prepare_dataframes(test_df, label_column=label_column, num_features=num_features)
    if X_train is None or X_test is None:
        return None
    train_loader = create_dataloaders(X_train, y_train)
    test_loader = create_dataloaders(X_test, y_test)

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    train_model(model, train_loader)
    
    train_f1, train_acc = evaluate_model(model, train_loader)
    test_f1, test_acc = evaluate_model(model, test_loader)
    
    print(f"Train F1 Score: {train_f1}, Train Accuracy: {train_acc}")
    print(f"Test F1 Score: {test_f1}, Test Accuracy: {test_acc}")

    return test_acc

def logistic_and_knn(df, num_features=None):
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']

    # Logistic Regression
    test_acc_niche_logistic = run_logistic_regression(train_df, test_df, 'niche_type', num_features=num_features)
    test_acc_cell_logistic = run_logistic_regression(train_df, test_df, 'cell_type', num_features=num_features)

    # KNN Classification
    test_acc_niche_knn = run_knn_classification(train_df, test_df, 'niche_type', num_features=num_features)
    test_acc_cell_knn = run_knn_classification(train_df, test_df, 'cell_type', num_features=num_features)

    print(f"Logistic Regression Niche Label Test Accuracy: {test_acc_niche_logistic}")
    print(f"Logistic Regression Cell Type Test Accuracy: {test_acc_cell_logistic}")
    print(f"KNN Classifier Niche Label Test Accuracy: {test_acc_niche_knn}")
    print(f"KNN Classifier Cell Type Test Accuracy: {test_acc_cell_knn}")

    return {
        "logistic_regression": {
            "niche_type": test_acc_niche_logistic,
            "cell_type": test_acc_cell_logistic
        },
        "knn_classifier": {
            "niche_type": test_acc_niche_knn,
            "cell_type": test_acc_cell_knn
        }
    }

