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
        if execution_time is None or execution_time < 0:
            print(f"Invalid execution time in {file_path}: {execution_time}")
            return None
        
        nodes = data["programming_details"]["Nodes"]
        edges = data["programming_details"]["Edges"]
        scheduling = data["scheduling_data"]
        
        node_dict = {node["Name"]: node["Details"] for node in nodes}
        sched_dict = {item["Name"]: item["Details"]["scheduling_feature"] for item in scheduling if "Name" in item and "Details" in item}
        
        features = {'execution_time': execution_time}
        
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
        
        features['nodes_count'] = len(nodes)
        features['edges_count'] = len(edges)
        features['avg_inputs'] = np.mean([len([e for e in edges if e["To"] == n["Name"]]) for n in nodes]) if nodes else 0
        features['reductions_count'] = sum(1 for e in edges if ".update(" in e["To"])
        
        loop_features = {
            'parallel_count': 0, 'tile_count': 0, 'total_tile_factor': 0,
            'vectorize_count': 0, 'total_vector_size': 0, 'unroll_count': 0, 'total_unroll_factor': 0
        }
        for node_name in node_dict:
            sched = sched_dict.get(node_name, {})
            for _ in range(2):
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

def process_main_directory(main_dir, train_ratio=0.9):
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

def prepare_data_for_model(train_features, test_features):
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
    
    orig_y_train = y_train.copy()
    orig_y_test = y_test.copy()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_y, X_train_scaled.shape[1], X.columns.tolist(), orig_y_train, orig_y_test

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, output_size=1):
        super(EnhancedLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        self.bn = nn.BatchNorm1d(hidden_size2)  # Apply BN after LSTM on hidden size
        self.fc = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()  # Ensure non-negative output
    
    def forward(self, x):
        out, _ = self.lstm1(x)  # [batch_size, seq_len=1, hidden_size1]
        out = self.dropout1(out)
        out, _ = self.lstm2(out)  # [batch_size, seq_len=1, hidden_size2]
        out = self.dropout2(out)
        out = out[:, -1, :]  # Take last timestep: [batch_size, hidden_size2]
        out = self.bn(out)  # Apply batch norm: [batch_size, hidden_size2]
        out = self.fc(out)  # [batch_size, output_size]
        out = self.relu(out)  # Clamp to non-negative
        return out

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=200, patience=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        scheduler.step(val_loss)
        
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
    
    return train_losses, val_losses

def evaluate_model(model, X_test, y_test, y_scaler, file_names_test, orig_y_test):
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
    
    y_pred_actual = np.clip(y_pred_actual, 0, None)
    
    print("\nPredicted vs Actual Execution Times for Test Files:")
    for i, file_name in enumerate(file_names_test):
        print(f"File: {file_name}")
        print(f"  Actual Execution Time: {y_test_actual[i][0]:.2f} ms")
        print(f"  Predicted Execution Time: {y_pred_actual[i][0]:.2f} ms")
    
    return y_test_actual, y_pred_actual

def main(main_dir, model_type='lstm', train_ratio=0.9):
    print(f"Processing main directory: {main_dir}")
    train_features, test_features, test_file_names = process_main_directory(main_dir, train_ratio)
    
    print(f"Total training samples: {len(train_features)}")
    print(f"Total test samples: {len(test_features)}")
    
    if len(train_features) == 0 or len(test_features) == 0:
        print("Error: No valid training or test data found")
        return None, None, None
    
    data = prepare_data_for_model(train_features, test_features)
    X_train, y_train, X_test, y_test, y_scaler, input_size, feature_names, orig_y_train, orig_y_test = data
    
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16)
    
    if model_type == 'lstm':
        model = EnhancedLSTMModel(input_size=input_size, hidden_size1=128, hidden_size2=64)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Using 'lstm' for now.")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    print(f"Training {model_type} model...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=200, patience=20
    )
    
    print("\nEvaluating the model...")
    y_test_actual, y_pred_actual = evaluate_model(
        model, X_test, y_test, y_scaler, test_file_names, orig_y_test
    )
    
    return model, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "Output_Programs"
    model, y_test_actual, y_pred_actual = main(main_dir, model_type='lstm')
    if model is not None:
        print("\nModel training and prediction completed!")
