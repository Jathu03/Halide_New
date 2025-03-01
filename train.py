import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Step 1: Load JSON Files from Directory
def load_json_files(directory):
    data = []
    program_names = []
    
    for program_folder in os.listdir(directory):
        program_path = os.path.join(directory, program_folder)
        if os.path.isdir(program_path):
            program_data = []
            for json_file in os.listdir(program_path):
                if json_file.endswith('.json'):
                    file_path = os.path.join(program_path, json_file)
                    try:
                        with open(file_path, 'r') as f:
                            json_data = json.load(f)
                            program_data.append(json_data)
                    except json.JSONDecodeError as e:
                        print(f"Skipping {file_path}: Invalid JSON ({e})")
                    except Exception as e:
                        print(f"Skipping {file_path}: Error ({e})")
            if program_data:
                data.append(program_data)
                program_names.append(program_folder)
            else:
                print(f"Warning: No valid JSON files loaded from {program_folder}")
    
    if not data:
        raise ValueError("No valid data loaded from Output_Programs folder")
    return data, program_names

# Step 2: Feature Extraction
def extract_features(json_data):
    features = []
    
    # Extract from Edges
    edges = json_data.get('programming_details', {}).get('Edges', [])
    num_edges = len(edges)
    footprint_sizes = [len(edge['Details']['Footprint']) for edge in edges if 'Details' in edge and 'Footprint' in edge['Details']]
    jacobian_sizes = [len(edge['Details']['Load Jacobians']) for edge in edges if 'Details' in edge and 'Load Jacobians' in edge['Details']]
    
    features.extend([
        num_edges,
        np.mean(footprint_sizes) if footprint_sizes else 0,
        np.mean(jacobian_sizes) if jacobian_sizes else 0
    ])
    
    # Extract from Nodes
    nodes = json_data.get('programming_details', {}).get('Nodes', [])
    num_nodes = len(nodes)
    memory_patterns = [sum(map(int, node['Details']['Memory access patterns'][0].split()[1:])) 
                       for node in nodes if node.get('Details', {}).get('Memory access patterns')]
    op_histogram = [sum(int(op.split()[-1]) for op in node['Details']['Op histogram']) 
                    for node in nodes if node.get('Details', {}).get('Op histogram')]
    
    features.extend([
        num_nodes,
        np.mean(memory_patterns) if memory_patterns else 0,
        np.mean(op_histogram) if op_histogram else 0
    ])
    
    # Extract Scheduling Features
    sched_features = []
    for item in json_data.get('schedule_feature', []):
        if isinstance(item, dict) and 'Details' in item and 'scheduling_feature' in item['Details']:
            sched_features.append(item['Details']['scheduling_feature'])
    if sched_features:
        avg_sched_features = {k: np.mean([sf[k] for sf in sched_features]) 
                              for k in sched_features[0].keys()}
        features.extend(list(avg_sched_features.values()))
    
    # Target: Execution Time
    execution_time = None
    for item in json_data.get('schedule_feature', []):
        if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
            execution_time = item.get('value', 0)
            break
    if execution_time is None:
        execution_time = 0
    
    return np.array(features, dtype=float), execution_time

# Step 3: Prepare Data for LSTM
def prepare_lstm_data(data):
    X, y = [], []
    feature_dim = None
    
    for program_data in data:
        program_X, program_y = [], []
        for schedule in program_data:
            features, exec_time = extract_features(schedule)
            if feature_dim is None:
                feature_dim = len(features)
            elif len(features) != feature_dim:
                features = np.pad(features, (0, feature_dim - len(features)), 'constant')[:feature_dim]
            program_X.append(features)
            program_y.append(exec_time)
        
        if program_X and program_y:
            X.append(program_X)
            y.append(program_y)
    
    if not X or not y:
        raise ValueError("No valid sequences prepared from the data")
    
    X = np.array(X)  # Shape: (num_programs, num_schedules, feature_dim)
    y = np.array(y)  # Shape: (num_programs, num_schedules)
    
    # Normalize features
    scaler_X = MinMaxScaler()
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_normalized = scaler_X.fit_transform(X_reshaped).reshape(X.shape)
    
    # Normalize target
    scaler_y = MinMaxScaler()
    y_reshaped = y.reshape(-1, 1)
    y_normalized = scaler_y.fit_transform(y_reshaped).reshape(y.shape)
    
    return X_normalized, y_normalized, scaler_X, scaler_y

# Step 4: Custom Dataset for PyTorch
class HalideDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Step 5: Define LSTM Model in PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)  # Removed return_sequences
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        out, _ = self.lstm1(x)  # out: (batch_size, seq_len, hidden_size1)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)  # out: (batch_size, seq_len, hidden_size2)
        out = self.dropout2(out[:, -1, :])  # Take the last timestep: (batch_size, hidden_size2)
        out = self.fc1(out)  # (batch_size, 32)
        out = self.relu(out)
        out = self.fc2(out)  # (batch_size, 1)
        return out

# Step 6: Training and Evaluation
def train_model(model, train_loader, val_loader, epochs=50, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch[:, -1].unsqueeze(1))  # Predict last timestep
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch[:, -1].unsqueeze(1)).item()
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.6f}, Val Loss: {val_loss/len(val_loader):.6f}")

def evaluate_model(model, test_loader, scaler_y, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.append(outputs.cpu().numpy())
            actuals.append(y_batch[:, -1].numpy())  # Last timestep
    
    y_pred = np.concatenate(predictions)
    y_test = np.concatenate(actuals)
    y_pred_rescaled = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_test_rescaled = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    mse = np.mean((y_test_rescaled - y_pred_rescaled) ** 2)
    print(f"Test MSE: {mse:.6f}")
    return y_test_rescaled, y_pred_rescaled

# Main Execution
def predict_halide_speedup(data_dir):
    # Load and prepare data
    data, program_names = load_json_files(data_dir)
    print(f"Loaded data from {len(program_names)} programs")
    
    X, y, scaler_X, scaler_y = prepare_lstm_data(data)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 60% train, 20% val, 20% test
    
    # Create datasets and loaders
    train_dataset = HalideDataset(X_train, y_train)
    val_dataset = HalideDataset(X_val, y_val)
    test_dataset = HalideDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    input_size = X.shape[2]  # feature_dim
    model = LSTMModel(input_size)
    
    # Train model
    train_model(model, train_loader, val_loader, epochs=50)
    
    # Evaluate model
    y_test_rescaled, y_pred_rescaled = evaluate_model(model, test_loader, scaler_y)
    
    # Print example predictions
    for i in range(min(5, len(y_test_rescaled))):
        print(f"Program {i+1}: True Time: {y_test_rescaled[i]:.2f} ms, Predicted Time: {y_pred_rescaled[i]:.2f} ms")
    
    # Save model
    torch.save(model.state_dict(), 'lstm_execution_time_predictor.pt')

if __name__ == "__main__":
    predict_halide_speedup(data_dir="Output_Programs")
