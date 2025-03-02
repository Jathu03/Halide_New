import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Function to extract features from a single JSON file
def extract_features_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract execution time (target variable)
    execution_time = None
    for item in data:
        if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
            execution_time = item.get('value')
            break
    
    if execution_time is None:
        print(f"Warning: No execution time found in {file_path}")
        return None
    
    # Extract programming details (node and edge features)
    nodes_features = []
    edges_features = []
    
    # Look for programming_details
    programming_details = None
    for key, value in data.items():
        if key == "programming_details":
            programming_details = value
            break
    
    if programming_details:
        # Process nodes
        if 'Nodes' in programming_details:
            for node in programming_details['Nodes']:
                node_feature = {}
                node_feature['Name'] = node.get('Name', '')
                
                # Extract op histogram if available
                if 'Details' in node and 'Op histogram' in node['Details']:
                    op_hist = node['Details']['Op histogram']
                    for op_line in op_hist:
                        parts = op_line.strip().split(':')
                        if len(parts) == 2:
                            op_name = parts[0].strip()
                            op_count = int(parts[1].strip())
                            node_feature[f'op_{op_name.lower()}'] = op_count
                
                nodes_features.append(node_feature)
        
        # Process edges
        if 'Edges' in programming_details:
            for edge in programming_details['Edges']:
                edge_feature = {}
                edge_feature['From'] = edge.get('From', '')
                edge_feature['To'] = edge.get('To', '')
                edge_feature['Name'] = edge.get('Name', '')
                
                edges_features.append(edge_feature)
    
    # Extract scheduling details
    scheduling_features = []
    
    # Look for scheduling_data
    scheduling_data = None
    for key, value in data.items():
        if key == "scheduling_data":
            scheduling_data = value
            break
    
    if scheduling_data:
        for sched in scheduling_data:
            sched_feature = {}
            sched_feature['Name'] = sched.get('Name', '')
            
            if 'Details' in sched and 'scheduling_feature' in sched['Details']:
                sf = sched['Details']['scheduling_feature']
                for key, value in sf.items():
                    sched_feature[key] = value
            
            scheduling_features.append(sched_feature)
    
    # Create a feature dictionary
    features = {
        'execution_time': execution_time,
        'nodes_count': len(nodes_features),
        'edges_count': len(edges_features),
        'scheduling_count': len(scheduling_features)
    }
    
    # Add aggregated node features
    op_counts = {}
    for node in nodes_features:
        for key, value in node.items():
            if key.startswith('op_'):
                op_counts[key] = op_counts.get(key, 0) + value
    
    features.update(op_counts)
    
    # Add important scheduling features (aggregate or take first few entries)
    if scheduling_features:
        # Common scheduling metrics to extract (based on your sample data)
        important_metrics = [
            'bytes_at_production', 'bytes_at_realization', 'bytes_at_root', 'bytes_at_task',
            'inner_parallelism', 'outer_parallelism', 'num_productions', 'num_realizations',
            'num_scalars', 'num_vectors', 'points_computed_total', 'working_set'
        ]
        
        # Take the first scheduling feature for these metrics (can be modified to use aggregation)
        if scheduling_features and scheduling_features[0]:
            for metric in important_metrics:
                if metric in scheduling_features[0]:
                    features[f'sched_{metric}'] = scheduling_features[0][metric]
        
        # Aggregate some metrics across all scheduling features
        total_bytes_at_production = sum(sf.get('bytes_at_production', 0) for sf in scheduling_features if isinstance(sf, dict))
        total_vectors = sum(sf.get('num_vectors', 0) for sf in scheduling_features if isinstance(sf, dict))
        total_parallelism = sum(sf.get('inner_parallelism', 0) * sf.get('outer_parallelism', 1) for sf in scheduling_features if isinstance(sf, dict))
        
        features['total_bytes_at_production'] = total_bytes_at_production
        features['total_vectors'] = total_vectors
        features['total_parallelism'] = total_parallelism
    
    return features

# Function to process all JSON files in a directory
def process_all_files(directory_path):
    all_features = []
    file_names = []
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            features = extract_features_from_file(file_path)
            
            if features is not None:
                all_features.append(features)
                file_names.append(filename)
    
    return all_features, file_names

# Function to prepare data for LSTM
def prepare_data_for_lstm(all_features, test_indices=None):
    # Convert to dataframe
    df = pd.DataFrame(all_features)
    
    # Check if we have enough data
    if len(df) <= 5:  # arbitrary small number
        raise ValueError("Not enough data samples to train the model")
    
    # Separate features and target
    X = df.drop('execution_time', axis=1)
    y = df['execution_time']
    
    # Handle missing values (replace with mean or 0)
    X = X.fillna(0)
    
    # Normalize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # If test_indices is provided, split data accordingly
    if test_indices is not None:
        test_mask = np.zeros(len(df), dtype=bool)
        test_mask[test_indices] = True
        train_mask = ~test_mask
        
        X_train = X[train_mask]
        y_train = y[train_mask].values.reshape(-1, 1)
        X_test = X[test_mask]
        y_test = y[test_mask].values.reshape(-1, 1)
        
        # Fit scalers on training data only
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        
        # Transform test data using the same scalers
        X_test_scaled = scaler_X.transform(X_test)
        y_test_scaled = scaler_y.transform(y_test)
    else:
        # Random split if test_indices not provided
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Reshape y for scaling
        y_train = y_train.values.reshape(-1, 1)
        y_test = y_test.values.reshape(-1, 1)
        
        # Scale the data
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_test_scaled = scaler_y.transform(y_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    # Reshape for LSTM [batch, seq_len, features]
    X_train_tensor = X_train_tensor.unsqueeze(1)  # Add sequence dimension of 1
    X_test_tensor = X_test_tensor.unsqueeze(1)   # Add sequence dimension of 1
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_y, X_train_scaled.shape[1]

# Define the LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=64, hidden_size2=32, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_size)
        
    def forward(self, x):
        # Input shape: [batch_size, seq_len, input_size]
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2[:, -1, :])  # Take the last time step output
        fc1_out = self.relu(self.fc1(lstm_out2))
        output = self.fc2(fc1_out)
        return output

# Function to create data loaders
def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=4):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Function to train the model
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=100, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model.to(device)
    
    # To track the best validation loss
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and
