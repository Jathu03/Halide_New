import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import matplotlib.pyplot as plt

# [Previous functions: get_execution_time, extract_features_from_file, process_directory, 
# process_main_directory, clean_and_transform_features remain unchanged]

def prepare_data_for_model(train_features, test_features):
    train_df, test_df = clean_and_transform_features(train_features, test_features)
    
    if 'execution_time_log' in train_df.columns:
        y_train = train_df['execution_time_log'].values.reshape(-1, 1)
        y_test = test_df['execution_time_log'].values.reshape(-1, 1)
        train_df = train_df.drop(['execution_time', 'execution_time_log'], axis=1)
        test_df = test_df.drop(['execution_time', 'execution_time_log'], axis=1)
        is_log_transformed = True
    else:
        y_train = train_df['execution_time'].values.reshape(-1, 1)
        y_test = test_df['execution_time'].values.reshape(-1, 1)
        train_df = train_df.drop('execution_time', axis=1)
        test_df = test_df.drop('execution_time', axis=1)
        is_log_transformed = False
    
    print("\nDebugging target values in prepare_data_for_model:")
    print(f"First 5 y_train raw: {y_train[:5].flatten()}")
    print(f"First 5 y_test raw: {y_test[:5].flatten()}")
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(train_df)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(test_df)
    y_test_scaled = scaler_y.transform(y_test)
    
    print(f"First 5 y_train scaled: {y_train_scaled[:5].flatten()}")
    print(f"First 5 y_test scaled: {y_test_scaled[:5].flatten()}")
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Input feature dimension: {X_train_scaled.shape[1]}")
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            scaler_X, scaler_y, train_df.columns.tolist(), X_train_scaled.shape[1], is_log_transformed)

# [Previous functions: EnhancedLSTMModel, create_data_loaders, train_model, evaluate_model remain unchanged]

def save_scaler_params(scaler_X, scaler_y, feature_names, is_log_transformed):
    # Save scaler_X parameters
    scaler_x_data = {
        "feature_names": feature_names,
        "means": scaler_X.mean_.tolist(),
        "scales": scaler_X.scale_.tolist()
    }
    with open("scaler_X.json", "w") as f:
        json.dump(scaler_x_data, f, indent=4)
    print("Saved scaler_X parameters to 'scaler_X.json'")
    
    # Save scaler_y parameters
    scaler_y_data = {
        "mean": float(scaler_y.mean_[0]),
        "scale": float(scaler_y.scale_[0]),
        "is_log_transformed": is_log_transformed
    }
    with open("scaler_y.json", "w") as f:
        json.dump(scaler_y_data, f, indent=4)
    print("Saved scaler_y parameters to 'scaler_y.json'")

def main(main_dir):
    print(f"Processing main directory: {main_dir}")
    train_features, test_features, test_file_names = process_main_directory(main_dir)
    
    print(f"Total training samples: {len(train_features)} (randomly selected)")
    print(f"Total test samples: {len(test_features)} (50 randomly selected)")
    
    if len(train_features) == 0 or len(test_features) == 0:
        print("Error: No valid training or test data found")
        return None
    
    original_execution_times = {}
    for feature, fname in zip(test_features, test_file_names):
        original_execution_times[fname] = feature['execution_time']
    
    X_train, y_train, X_test, y_test, scaler_X, scaler_y, feature_names, input_size, is_log_transformed = prepare_data_for_model(train_features, test_features)
    
    # Save scaler parameters
    save_scaler_params(scaler_X, scaler_y, feature_names, is_log_transformed)
    
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=8)
    
    model = EnhancedLSTMModel(
        input_size=input_size,
        hidden_sizes=[256, 128, 64, 32],
        output_size=1,
        dropout_rate=0.2
    )
    
    criterion = nn.HuberLoss(delta=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
    
    print("Building and training Enhanced LSTM model...")
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        test_loader, 
        criterion, 
        optimizer, 
        num_epochs=200,
        patience=30
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_enhanced_model.png')
    plt.close()
    print("Training plot saved as 'loss_enhanced_model.png'")
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(
        model, X_test, y_test, scaler_y, test_file_names, 
        is_log_transformed, original_execution_times
    )
    
    print("\nSaving the trained model as 'lstm_model.pt'...")
    model.eval()
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")
    
    try:
        sample_input = torch.randn(1, 1, input_size).to(device)
        traced_model = torch.jit.trace(model, sample_input)
        traced_model.save("lstm_model.pt")
        print("Model successfully saved as 'lstm_model.pt'")
    except Exception as e:
        print(f"Error saving the model: {str(e)}")
    
    return model, scaler_y, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "synthetic_data"
    random.seed(42)
    result = main(main_dir)
    if result is not None:
        model, y_scaler, y_test_actual, y_pred_actual = result
