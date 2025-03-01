import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import json

# Helper function to parse fractions
def parse_number(s):
    try:
        # If it's a simple number, convert to float
        return float(s)
    except ValueError:
        # If it's a fraction (e.g., '1/8'), evaluate it
        if '/' in s:
            num, denom = s.split('/')
            return float(num) / float(denom)
        raise ValueError(f"Cannot convert {s} to float")

# 1. Feature Extraction Function (Modified)
def extract_features(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    features = []
    
    # Extract Edge Features (numerical values from Load Jacobians)
    edge_features = []
    for edge in data['programming_details']['Edges']:
        jacobian = edge['Details']['Load Jacobians']
        jacobian_vals = []
        for row in jacobian:
            # Handle both numbers and fractions, exclude '_', '0', '1'
            nums = [parse_number(x) for x in row.split() if x not in ['_', '0', '1']]
            jacobian_vals.extend(nums)
        edge_features.extend(jacobian_vals)
    
    # Extract Node Features (numerical values from Op histogram)
    node_features = []
    for node in data['programming_details']['Nodes']:
        histogram = node['Details']['Op histogram']
        hist_vals = []
        for line in histogram:
            if ':' in line:
                val = float(line.split(':')[1].strip())
                hist_vals.append(val)
        node_features.extend(hist_vals)
    
    # Extract Scheduling Features
    scheduling_features = []
    for schedule in data['programming_details']['Schedules']:
        if isinstance(schedule, dict) and 'Details' in schedule:
            sched_data = schedule['Details']['scheduling_feature']
            sched_vals = [float(v) for v in sched_data.values()]
            scheduling_features.extend(sched_vals)
    
    # Combine all features
    features.extend(edge_features)
    features.extend(node_features)
    features.extend(scheduling_features)
    
    # Get execution time (y_data)
    execution_time = next(item['value'] for item in data['programming_details']['Schedules'] 
                         if item.get('name') == 'total_execution_time_ms')
    
    return np.array(features), execution_time

# 2. Load and Process All Files
def load_data(folder_path):
    X_data = []
    y_data = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            features, exec_time = extract_features(file_path)
            X_data.append(features)
            y_data.append(exec_time)
    
    max_length = max(len(x) for x in X_data)
    X_data_padded = np.array([np.pad(x, (0, max_length - len(x)), 'constant') 
                            for x in X_data])
    
    return X_data_padded, np.array(y_data)

# 3. Prepare Sequences for LSTM
def create_sequences(X, y, sequence_length=5):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:(i + sequence_length)])
        y_seq.append(y[i + sequence_length])
    return np.array(X_seq), np.array(y_seq)

# 4. Custom Dataset Class for PyTorch
class ScheduleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 5. LSTM Model Definition in PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=100, hidden_size2=50, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, num_layers=1)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True, num_layers=1)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size2, 1)
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = self.fc(out[:, -1, :])  # Take the last time step output
        return out

# 6. Main Training Function
def train_lstm_model(folder_path):
    # Load and preprocess data
    X_data, y_data = load_data(folder_path)
    
    # Normalize features
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_data)
    
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1))
    
    # Create sequences
    sequence_length = 5
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)
    
    # Split into train and test sets
    train_size = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:train_size], X_seq[train_size:]
    y_train, y_test = y_seq[:train_size], y_seq[train_size:]
    
    # Create PyTorch datasets and dataloaders
    train_dataset = ScheduleDataset(X_train, y_train)
    test_dataset = ScheduleDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMModel(input_size=X_scaled.shape[1]).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation (using test set as validation here)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
        
        val_loss /= len(test_loader.dataset)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    # Final evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            test_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
    
    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss}")
    
    # Save model and scalers
    torch.save(model.state_dict(), 'lstm_schedule_predictor.pt')
    np.save('scaler_X.npy', scaler_X)
    np.save('scaler_y.npy', scaler_y)
    
    return model, scaler_X, scaler_y

# 7. Prediction Function
def predict_execution_time(model, scaler_X, scaler_y, new_file_path, sequence_length=5):
    # Load and preprocess new data
    features, _ = extract_features(new_file_path)
    max_length = scaler_X.data_max_.shape[0]
    features_padded = np.pad(features, (0, max_length - len(features)), 'constant')
    features_scaled = scaler_X.transform(features_padded.reshape(1, -1))
    
    # Create sequence
    sequence = np.repeat(features_scaled, sequence_length, axis=0)[np.newaxis, :]
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
    
    # Make prediction
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        sequence_tensor = sequence_tensor.to(device)
        pred_scaled = model(sequence_tensor)
        pred_scaled = pred_scaled.cpu().numpy()
    
    prediction = scaler_y.inverse_transform(pred_scaled)
    return prediction[0][0]

# Main execution
if __name__ == "__main__":
    folder_path = 'Output_Programs/program_50001'  # Replace with actual path
    model, scaler_X, scaler_y = train_lstm_model(folder_path)
    
    # Example prediction
    new_file = 'path/to/new/schedule.json'
    predicted_time = predict_execution_time(model, scaler_X, scaler_y, new_file)
    print(f"Predicted Execution Time: {predicted_time} ms")
