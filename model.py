import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Save scaler parameters (corrected for RobustScaler)
def save_scaler_params(scaler_X, scaler_y, is_log_transformed):
    # Handle RobustScaler for X
    scaler_X_data = {
        "feature_names": list(scaler_X.feature_names_in_),
        "centers": scaler_X.center_.tolist(),  # Use center_ instead of mean_
        "scales": scaler_X.scale_.tolist()
    }
    with open("scaler_X.json", "w") as f:
        json.dump(scaler_X_data, f)

    # Handle StandardScaler for y
    scaler_y_data = {
        "mean": float(scaler_y.mean_[0]),
        "scale": float(scaler_y.scale_[0]),
        "is_log_transformed": is_log_transformed
    }
    with open("scaler_y.json", "w") as f:
        json.dump(scaler_y_data, f)

# (Previous functions like get_execution_time, extract_features_from_file, etc., remain unchanged)

# Prepare data with robust scaling
def prepare_data_for_model(train_features, test_features):
    train_df, test_df = clean_and_transform_features(train_features, test_features)
    
    y_train = train_df['execution_time_log'].values.reshape(-1, 1)
    y_test = test_df['execution_time_log'].values.reshape(-1, 1)
    X_train = train_df.drop(['execution_time', 'execution_time_log'], axis=1)
    X_test = test_df.drop(['execution_time', 'execution_time_log'], axis=1)
    
    scaler_X = RobustScaler(quantile_range=(25.0, 75.0))
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Input feature dimension: {X_train_scaled.shape[1]}")
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_X, scaler_y, X_train_scaled.shape[1], True

# (Other functions like EnhancedLSTMModel, train_model, evaluate_model remain unchanged)

# Main function
def main(main_dir="synthetic_data"):
    print(f"Processing {main_dir}")
    train_features, test_features, test_file_names = process_main_directory(main_dir)
    
    X_train, y_train, X_test, y_test, scaler_X, scaler_y, input_size, is_log_transformed = prepare_data_for_model(train_features, test_features)
    save_scaler_params(scaler_X, scaler_y, is_log_transformed)
    
    model = EnhancedLSTMModel(input_size=input_size, hidden_sizes=[256, 128, 64], num_heads=4, dropout_rate=0.4)
    criterion = nn.HuberLoss(delta=1.0)
    
    print("Training with cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    best_model = None
    best_val_loss = float('inf')
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\nFold {fold+1}/5")
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        fold_model = EnhancedLSTMModel(input_size=input_size, hidden_sizes=[256, 128, 64], num_heads=4, dropout_rate=0.4)
        _, val_losses, trained_model = train_model(fold_model, X_tr, y_tr, X_val, y_val, criterion)
        
        fold_val_loss = min(val_losses)
        cv_scores.append(fold_val_loss)
        if fold_val_loss < best_val_loss:
            best_val_loss = fold_val_loss
            best_model = trained_model
    
    print(f"\nCross-validation MAPE scores: {cv_scores}, Mean: {np.mean(cv_scores):.4f}")
    
    print("\nFinal training on full dataset...")
    train_losses, val_losses, best_model = train_model(best_model, X_train, y_train, X_test, y_test, criterion)
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual, avg_actual, avg_predicted = evaluate_model(best_model, X_test, y_test, scaler_y, test_file_names, is_log_transformed)
    
    torch.jit.save(torch.jit.trace(best_model.cpu(), torch.randn(1, 1, input_size)), "lstm_model.pt")
    print("Model saved as 'lstm_model.pt'")
    
    print(f"\nSummary: Avg Actual: {avg_actual:.2f} ms, Avg Predicted: {avg_predicted:.2f} ms")
    return best_model, y_scaler, y_test_actual, y_pred_actual, avg_actual, avg_predicted

if __name__ == "__main__":
    main_dir = "synthetic_data"
    model, y_scaler, y_test_actual, y_pred_actual, avg_actual, avg_predicted = main(main_dir)
