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
import joblib

# Set device to CPU explicitly
device = torch.device('cpu')
print(f"Using device: {device}")

# [Your existing functions: get_execution_time, extract_features_from_file, process_directory, 
# process_main_directory, clean_and_transform_features remain unchanged]

def prepare_data_for_model(train_features, test_features):
    train_df, test_df = clean_and_transform_features(train_features, test_features)
    
    if 'execution_time_log' in train_df.columns:
        y_train = train_df['execution_time_log'].values.reshape(-1, 1)
        y_test = test_df['execution_time_log'].values.reshape(-1, 1)
        train_df = train_df.drop(['execution_time', 'execution_time_log'], axis=1)
        test_df = test_df.drop(['execution_time', 'execution_time_log'], axis=1)
    else:
        y_train = train_df['execution_time'].values.reshape(-1, 1)
        y_test = test_df['execution_time'].values.reshape(-1, 1)
        train_df = train_df.drop('execution_time', axis=1)
        test_df = test_df.drop('execution_time', axis=1)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(train_df)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(test_df)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Save scalers (unchanged)
    feature_names = list(train_df.columns)
    scaler_data = {'feature_names': feature_names, 'means': scaler_X.mean_.tolist(), 'scales': scaler_X.scale_.tolist()}
    with open('scaler_X.json', 'w') as f:
        json.dump(scaler_data, f)
    print("Saved feature scaler parameters to 'scaler_X.json'")
    
    y_scaler_data = {'mean': scaler_y.mean_[0], 'scale': scaler_y.scale_[0], 'is_log_transformed': 'execution_time_log' in train_df.columns}
    with open('scaler_y.json', 'w') as f:
        json.dump(y_scaler_data, f)
    print("Saved target scaler parameters to 'scaler_y.json'")
    
    # Create tensors with explicit CPU device
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1).to(device)
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1).to(device)
    y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)
    
    print(f"Input feature dimension: {X_train_scaled.shape[1]}")
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            scaler_y, X_train_scaled.shape[1], 'execution_time_log' in train_df.columns)

class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=1, dropout_rate=0.3):
        super(EnhancedLSTMModel, self).__init__()
        self.hidden_sizes = hidden_sizes
        
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        self.lstm_layers.append(nn.LSTM(input_size, hidden_sizes[0], batch_first=True))
        self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(nn.LSTM(hidden_sizes[i-1], hidden_sizes[i], batch_first=True))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        self.attention = nn.Linear(hidden_sizes[-1], 1)
        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        self.fc_layers.append(nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2))
        self.bn_layers.append(nn.BatchNorm1d(hidden_sizes[-1] // 2))
        self.fc_layers.append(nn.Linear(hidden_sizes[-1] // 2, hidden_sizes[-1] // 4))
        self.bn_layers.append(nn.BatchNorm1d(hidden_sizes[-1] // 4))
        
        self.output_layer = nn.Linear(hidden_sizes[-1] // 4, output_size)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        self.has_residual = (hidden_sizes[-1] // 4 == hidden_sizes[-1] // 2)
        if not self.has_residual:
            self.residual_adapter = nn.Linear(hidden_sizes[-1] // 2, hidden_sizes[-1] // 4)
    
    # [Forward method unchanged, no device movement needed since inputs are on CPU]

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader

# [train_model, evaluate_model, create_schedule_representation unchanged]

def main(main_dir):
    print(f"Processing main directory: {main_dir}")
    try:
        train_features, test_features, test_file_names = process_main_directory(main_dir)
        
        print(f"Total training samples: {len(train_features)}")
        print(f"Total test samples: {len(test_features)}")
        
        if len(train_features) == 0 or len(test_features) == 0:
            print("Error: No valid training or test data found")
            return None, None, None, None
        
        X_train, y_train, X_test, y_test, y_scaler, input_size, is_log_transformed = prepare_data_for_model(train_features, test_features)
        
        train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16)
        
        model = EnhancedLSTMModel(input_size=input_size, hidden_sizes=[128, 64, 32], output_size=1, dropout_rate=0.3).to(device)
        
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
        
        print("Building and training Enhanced LSTM model...")
        train_losses, val_losses = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, patience=20)
        
        print("\nEvaluating model:")
        y_test_actual, y_pred_actual = evaluate_model(model, X_test, y_test, y_scaler, test_file_names, is_log_transformed)
        
        print("\nSaving the trained model as 'lstm_model.pt'...")
        model.eval()
        sample_input = torch.randn(1, 1, input_size).to(device)
        
        traced_model = torch.jit.trace(model, sample_input, strict=False)
        traced_model.save("lstm_model.pt")
        print("Model successfully saved as 'lstm_model.pt'")
        
        joblib.dump(y_scaler, "y_scaler.pkl")
        print("Scaler saved as 'y_scaler.pkl'")
        
        return model, y_scaler, y_test_actual, y_pred_actual
    
    except Exception as e:
        print(f"Error in main: {str(e)}")
        return None, None, None, None

if __name__ == "__main__":
    main_dir = "synthetic_data"
    random.seed(42)
    result = main(main_dir)
    if result is not None:
        model, y_scaler, y_test_actual, y_pred_actual = result
    else:
        print("Main function failed to return valid results")
