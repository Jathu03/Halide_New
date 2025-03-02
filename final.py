import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def get_execution_time(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw_content = f.read()
            content = raw_content.decode('utf-8', errors='replace').replace('\0', '')
            data = json.loads(content)
        schedules = data["scheduling_data"]
        for item in schedules:
            if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
                execution_time = item.get('value')
                if execution_time is not None:
                    return float(execution_time)
        print(f"Warning: 'total_execution_time_ms' not found in 'Schedules' of {file_path}")
        return schedules[-1]["value"]
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def extract_optimized_features(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        execution_time = get_execution_time(file_path)
        if execution_time is None:
            return None
        
        nodes = data["programming_details"]["Nodes"]
        edges = data["programming_details"]["Edges"]
        scheduling = data["scheduling_data"]
        
        node_dict = {node["Name"]: node["Details"] for node in nodes}
        sched_dict = {item["Name"]: item["Details"]["scheduling_feature"] for item in scheduling if "Name" in item and "Details" in item}
        
        features = {'execution_time': execution_time}
        
        # Operation counts
        op_counts = {}
        for node_name, details in node_dict.items():
            op_hist = details.get("Op histogram", [])
            for entry in op_hist:
                parts = entry.split(':')
                if len(parts) == 2:
                    op_name = parts[0].strip().lower()
                    op_count = int(parts[1].strip().split()[0])
                    op_counts[f'op_{op_name}'] = op_counts.get(f'op_{op_name}', 0) + op_count
        features.update(op_counts)
        
        # Structural features
        features['nodes_count'] = len(nodes)
        features['edges_count'] = len(edges)
        features['avg_inputs'] = np.mean([len([e for e in edges if e["To"] == n["Name"]]) for n in nodes]) if nodes else 0
        features['reductions_count'] = sum(1 for e in edges if ".update(" in e["To"])
        
        # Scheduling features
        loop_features = {
            'parallel_count': 0, 'tile_count': 0, 'total_tile_factor': 0,
            'vectorize_count': 0, 'total_vector_size': 0, 'unroll_count': 0, 'total_unroll_factor': 0
        }
        for node_name in node_dict:
            sched = sched_dict.get(node_name, {})
            for _ in range(2):  # Assuming up to 2 loops
                if sched.get("inner_parallelism", 1.0) > 1.0:
                    loop_features['parallel_count'] += 1
                tile_factor = sched.get("unrolled_loop_extent", 1.0)
                if tile_factor > 1.0:
                    loop_features['tile_count'] += 1
                    loop_features['total_tile_factor'] += tile_factor
                vector_size = sched.get("vector_size", 1.0)
                if vector_size > 1.0:
                    loop_features['vectorize_count'] += 1
                    loop_features['total_vector_size'] += vector_size
                if tile_factor > 1.0:
                    loop_features['unroll_count'] += 1
                    loop_features['total_unroll_factor'] += tile_factor
        features.update(loop_features)
        
        features['total_bytes_at_production'] = sum(sched.get('bytes_at_production', 0) for sched in sched_dict.values())
        features['total_vectors'] = sum(sched.get('num_vectors', 0) for sched in sched_dict.values())
        features['total_parallelism'] = sum(sched.get('inner_parallelism', 0) * sched.get('outer_parallelism', 1) for sched in sched_dict.values())
        
        return features
    
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

def process_main_directory(main_dir):
    train_features = []
    test_features = []
    test_file_names = []
    
    subdirs = sorted([d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))])
    
    if not subdirs:
        raise ValueError(f"No subdirectories found in {main_dir}")
    
    for subdir in subdirs:
        subdir_path = os.path.join(main_dir, subdir)
        all_features = []
        all_file_names = []
        
        for filename in sorted(os.listdir(subdir_path)):
            if filename.endswith('.json'):
                file_path = os.path.join(subdir_path, filename)
                features = extract_optimized_features(file_path)
                if features is not None:
                    all_features.append(features)
                    all_file_names.append(filename)
        
        if len(all_features) != 32:
            print(f"Warning: Expected 32 files in {subdir_path}, found {len(all_features)}")
            continue
        
        train_features.extend(all_features[:30])
        test_features.extend(all_features[30:])
        test_file_names.extend([os.path.join(subdir, fname) for fname in all_file_names[30:]])
        print(f"Processed {len(all_features)} files from {subdir}: 30 for training, 2 for testing")
    
    return train_features, test_features, test_file_names

def prepare_data_for_lstm(train_features, test_features):
    all_features_df = pd.DataFrame(train_features + test_features)
    if len(all_features_df) <= 5:
        raise ValueError("Not enough data samples to train the model")
    
    X = all_features_df.drop('execution_time', axis=1)
    y = all_features_df['execution_time']
    X = X.fillna(0)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    train_size = len(train_features)
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size].values.reshape(-1, 1)
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:].values.reshape(-1, 1)
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Remove unsqueeze since we're not using LSTM with sequence length
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_y, X_train_scaled.shape[1]

class ResidualDNN(nn.Module):
    def __init__(self, input_size, hidden_size1=256, hidden_size2=128, hidden_size3=64, output_size=1):
        super(ResidualDNN, self).__init__()
        # Input layer
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        # Hidden layer 1 with residual connection
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc2_residual = nn.Linear(hidden_size1, hidden_size2)  # Residual projection
        
        # Hidden layer 2 with residual connection
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.bn3 = nn.BatchNorm1d(hidden_size3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        self.fc3_residual = nn.Linear(hidden_size2, hidden_size3)  # Residual projection
        
        # Output layer
        self.fc4 = nn.Linear(hidden_size3, output_size)
    
    def forward(self, x):
        # Input layer
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Hidden layer 1 with residual
        residual = self.fc2_residual(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + residual  # Residual connection
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Hidden layer 2 with residual
        residual = self.fc3_residual(out)
        out = self.fc3(out)
        out = self.bn3(out)
        out = out + residual  # Residual connection
        out = self.relu3(out)
        out = self.dropout3(out)
        
        # Output layer
        out = self.fc4(out)
        return out

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=200, patience=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            model.load_state_dict(best_model_state)
            break
    
    if best_model_state is not None and epochs_no_improve > 0:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_model(model, X_test, y_test, y_scaler, file_names_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    X_test = X_test.to(device)
    with torch.no_grad():
        y_pred_scaled = model(X_test)
    
    y_pred_scaled = y_pred_scaled.cpu().numpy()
    y_test = y_test.cpu().numpy()
    
    y_test_actual = y_scaler.inverse_transform(y_test)
    y_pred_actual = y_scaler.inverse_transform(y_pred_scaled)
    
    print("\nPredicted vs Actual Execution Times for Test Files:")
    for i, file_name in enumerate(file_names_test):
        print(f"File: {file_name}")
        print(f"  Actual Execution Time: {y_test_actual[i][0]:.2f} ms")
        print(f"  Predicted Execution Time: {y_pred_actual[i][0]:.2f} ms")
    
    return y_test_actual, y_pred_actual

def main(main_dir):
    print(f"Processing main directory: {main_dir}")
    train_features, test_features, test_file_names = process_main_directory(main_dir)
    print(f"Total training samples: {len(train_features)} (30 files from each program)")
    print(f"Test samples: {len(test_features)} (2 files from each program)")
    
    X_train, y_train, X_test, y_test, y_scaler, input_size = prepare_data_for_lstm(train_features, test_features)
    
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32)
    model = ResidualDNN(input_size=input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    print("Building and training Residual DNN model...")
    model = train_model(model, train_loader, test_loader, criterion, optimizer)
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(model, X_test, y_test, y_scaler, test_file_names)
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "Output_Programs"
    model, y_scaler, y_test_actual, y_pred_actual = main(main_dir)
    print("\nModel training and prediction completed!")
