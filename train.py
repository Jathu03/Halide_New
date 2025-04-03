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

# Set device at the start
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_execution_time(file_path):
    # [Same as original implementation]
    return execution_time

def extract_features_from_file(file_path):
    # [Same as original implementation]
    return features

def process_directory(directory_path):
    # [Same as original implementation]
    return all_features, file_names

def process_main_directory(main_dir):
    # [Same as original implementation]
    return train_features, test_features, list(test_file_names)

def clean_and_transform_features(train_features, test_features):
    # [Same as original implementation]
    return train_df, test_df

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
    
    # Save scaler parameters
    with open('scaler_X.json', 'w') as f:
        json.dump({
            'feature_names': list(train_df.columns),
            'means': scaler_X.mean_.tolist(),
            'scales': scaler_X.scale_.tolist()
        }, f)
    with open('scaler_y.json', 'w') as f:
        json.dump({
            'mean': scaler_y.mean_[0],
            'scale': scaler_y.scale_[0],
            'is_log_transformed': 'execution_time_log' in train_df.columns
        }, f)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1).to(device)
    y_train_tensor = torch.FloatTensor(y_train_scaled).to(device)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1).to(device)
    y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            scaler_y, X_train_scaled.shape[1], 'execution_time_log' in train_df.columns)

class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=1, dropout_rate=0.3):
        super(EnhancedLSTMModel, self).__init__()
        # [Same as original implementation]
    
    def attention_net(self, lstm_output):
        # [Same as original implementation]
        return context
        
    def forward(self, x):
        # [Same as original implementation]
        return output

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=150, patience=20):
    model.to(device)
    # [Rest of the implementation same, ensuring all tensors are moved to device]
    return train_losses, val_losses

def evaluate_model(model, X_test, y_test, y_scaler, file_names_test, is_log_transformed=False):
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        y_pred_scaled = model(X_test)
    
    y_pred_scaled = y_pred_scaled.cpu().numpy()
    y_test = y_test.cpu().numpy()
    # [Rest of the implementation same]
    return y_test_actual, y_pred_actual

def main(main_dir):
    train_features, test_features, test_file_names = process_main_directory(main_dir)
    
    if len(train_features) == 0 or len(test_features) == 0:
        print("Error: No valid training or test data found")
        return None, None, None, None
    
    X_train, y_train, X_test, y_test, y_scaler, input_size, is_log_transformed = prepare_data_for_model(train_features, test_features)
    
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16)
    
    model = EnhancedLSTMModel(
        input_size=input_size,
        hidden_sizes=[128, 64, 32],
        output_size=1,
        dropout_rate=0.3
    ).to(device)
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    print("Building and training Enhanced LSTM model...")
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        test_loader, 
        criterion, 
        optimizer, 
        num_epochs=10,
        patience=20
    )
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(model, X_test, y_test, y_scaler, test_file_names, is_log_transformed)
    
    print("\nSaving the trained model as 'lstm_model.pt'...")
    model.eval()
    sample_input = torch.randn(1, 1, input_size).to(device)
    
    try:
        traced_model = torch.jit.trace(model, sample_input)
        traced_model.save("lstm_model.pt")
        print("Model successfully saved as 'lstm_model.pt'")
        
        joblib.dump(y_scaler, "y_scaler.pkl")
        print("Scaler saved as 'y_scaler.pkl'")
        
    except Exception as e:
        print(f"Error saving the model: {str(e)}")
        return None, None, None, None
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "synthetic_data"
    random.seed(42)
    result = main(main_dir)
    if result is not None:
        model, y_scaler, y_test_actual, y_pred_actual = result
    else:
        print("Main function failed to return valid results")
