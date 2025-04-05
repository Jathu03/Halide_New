import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from tqdm import tqdm

# Define operation types (modify according to your actual operations)
OP_TYPES = [
    'op_load', 'op_store', 'op_add', 'op_mul', 'op_div', 
    'op_fma', 'op_sqrt', 'op_exp', 'op_log', 'op_sin',
    'op_cos', 'op_tanh', 'op_conv', 'op_pool', 'op_matmul'
]

def get_execution_time(data):
    """Extract execution time from JSON data"""
    if 'scheduling_data' not in data:
        return None
    
    schedules = data["scheduling_data"]
    for item in schedules:
        if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
            return float(item.get('value'))
    
    if schedules and isinstance(schedules[-1], dict) and 'value' in schedules[-1]:
        return float(schedules[-1]['value'])
    
    return None

def process_nested_directory(root_dir):
    """Process all JSON files in nested directory structure"""
    all_features = []
    file_paths = []
    
    # Walk through all subdirectories
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    execution_time = get_execution_time(data)
                    if execution_time is None:
                        continue
                        
                    # Extract features
                    features = extract_features_from_json(data)
                    features['execution_time'] = execution_time
                    
                    # Store relative path for identification
                    rel_path = os.path.relpath(file_path, root_dir)
                    all_features.append(features)
                    file_paths.append(rel_path)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                    continue
    
    return all_features, file_paths

def extract_features_from_json(data):
    """Extract features from a single JSON file's data"""
    features = {
        'node_features': [],
        'edge_features': [],
        'schedule_features': []
    }
    
    programming_details = data.get("programming_details", {})
    
    # Process nodes (operations)
    for node in programming_details.get("Nodes", []):
        node_feat = {'op_type_counts': {}}
        if 'Details' in node and 'Op histogram' in node['Details']:
            for op_line in node['Details']['Op histogram']:
                parts = op_line.strip().split(':')
                if len(parts) == 2:
                    op_name = 'op_' + parts[0].strip().lower()
                    op_count = int(parts[1].strip())
                    node_feat['op_type_counts'][op_name] = op_count
        features['node_features'].append(node_feat)
    
    # Process edges (data dependencies)
    for edge in programming_details.get("Edges", []):
        edge_feat = {
            'data_size': edge.get('Size', 0),
            'is_vector': int('vector' in edge.get('Name', '').lower())
        }
        features['edge_features'].append(edge_feat)
    
    # Process schedules
    scheduling_data = data.get("scheduling_data", programming_details.get("Schedules", []))
    for schedule in scheduling_data:
        sched_feat = {
            'bytes': schedule.get('bytes_at_production', 0),
            'parallelism': schedule.get('inner_parallelism', 1) * schedule.get('outer_parallelism', 1),
            'working_set': schedule.get('working_set', 0),
            'vectors': schedule.get('num_vectors', 0)
        }
        features['schedule_features'].append(sched_feat)
    
    # Compute aggregated features
    features['num_nodes'] = len(features['node_features'])
    features['num_edges'] = len(features['edge_features'])
    features['num_schedules'] = len(features['schedule_features'])
    
    # Node statistics
    if features['node_features']:
        op_counts = {}
        for node in features['node_features']:
            for op, count in node['op_type_counts'].items():
                op_counts[op] = op_counts.get(op, 0) + count
        
        features['total_ops'] = sum(op_counts.values())
        features['unique_ops'] = len(op_counts)
        features['op_diversity'] = features['unique_ops'] / features['num_nodes'] if features['num_nodes'] > 0 else 0
        
        # Add top operation counts
        for op, count in sorted(op_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            features[op] = count
    
    # Edge statistics
    if features['edge_features']:
        features['total_data_size'] = sum(e['data_size'] for e in features['edge_features'])
        features['vector_edges'] = sum(e['is_vector'] for e in features['edge_features'])
    
    # Schedule statistics
    if features['schedule_features']:
        features['total_bytes'] = sum(s['bytes'] for s in features['schedule_features'])
        features['total_parallelism'] = sum(s['parallelism'] for s in features['schedule_features'])
        features['total_working_set'] = sum(s['working_set'] for s in features['schedule_features'])
        features['total_vectors'] = sum(s['vectors'] for s in features['schedule_features'])
        
        features['bytes_per_vector'] = features['total_bytes'] / (features['total_vectors'] + 1e-8)
        features['memory_pressure'] = features['total_working_set'] / (features['total_bytes'] + 1e-8)
    
    return features

def create_sequential_representation(features, max_nodes=50, max_edges=100, max_schedules=20):
    """Convert features into sequential format suitable for LSTM"""
    seq_features = []
    
    # 1. Node features sequence
    node_feats = []
    for node in features['node_features'][:max_nodes]:
        op_vector = [0] * len(OP_TYPES)
        for op in node['op_type_counts']:
            if op in OP_TYPES:
                op_vector[OP_TYPES.index(op)] = 1
        node_feats.append(op_vector)
    
    # Pad node sequence
    while len(node_feats) < max_nodes:
        node_feats.append([0] * len(OP_TYPES))
    
    # 2. Edge features sequence
    edge_feats = []
    for edge in features['edge_features'][:max_edges]:
        edge_feats.append([edge['data_size'], edge['is_vector']])
    
    # Pad edge sequence
    while len(edge_feats) < max_edges:
        edge_feats.append([0, 0])
    
    # 3. Schedule features sequence
    sched_feats = []
    for sched in features['schedule_features'][:max_schedules]:
        sched_feats.append([
            sched['bytes'],
            sched['parallelism'],
            sched['working_set'],
            sched['vectors']
        ])
    
    # Pad schedule sequence
    while len(sched_feats) < max_schedules:
        sched_feats.append([0, 0, 0, 0])
    
    # Combine all sequences
    seq_features.extend(node_feats)
    seq_features.extend(edge_feats)
    seq_features.extend(sched_feats)
    
    # Add aggregated features
    seq_features.append([
        features['num_nodes'] / max_nodes,
        features['num_edges'] / max_edges,
        features['num_schedules'] / max_schedules,
        np.log1p(features['total_ops']),
        features['unique_ops'] / len(OP_TYPES),
        features['op_diversity'],
        np.log1p(features['total_data_size']),
        features['vector_edges'] / max_edges,
        np.log1p(features['total_bytes']),
        np.log1p(features['total_parallelism']),
        np.log1p(features['total_working_set']),
        np.log1p(features['total_vectors'] + 1),
        np.log1p(features['bytes_per_vector'] + 1),
        features['memory_pressure']
    ])
    
    return np.array(seq_features, dtype=np.float32)

def prepare_data(all_features, test_size=0.2):
    """Prepare train/test split and scale data"""
    # Convert features to sequential format
    X = np.array([create_sequential_representation(f) for f in all_features])
    
    # Get execution times and apply log transform
    y = np.log1p(np.array([f['execution_time'] for f in all_features], dtype=np.float32))
    
    # Split data
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Reshape for scaling
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Scale features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_flat)
    X_test_scaled = scaler_X.transform(X_test_flat)
    
    # Scale targets
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).reshape(-1, 1, X_train_scaled.shape[1])
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled).reshape(-1, 1, X_test_scaled.shape[1])
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            scaler_X, scaler_y, X_train_scaled.shape[1])

class ProgramExecutionPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super(ProgramExecutionPredictor, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.node_head = nn.Linear(hidden_size, hidden_size//2)
        self.edge_head = nn.Linear(hidden_size, hidden_size//2)
        self.sched_head = nn.Linear(hidden_size, hidden_size//2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Feature-specific processing
        node_features = F.relu(self.node_head(context))
        edge_features = F.relu(self.edge_head(context))
        sched_features = F.relu(self.sched_head(context))
        
        # Combine features
        combined = torch.cat([context, node_features, edge_features, sched_features], dim=1)
        
        return self.fc(combined)

def train_and_evaluate(X_train, y_train, X_test, y_test, input_size, epochs=150, patience=20):
    device = torch.device('cpu')
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = ProgramExecutionPredictor(input_size=input_size).to(device)
    
    # Training setup
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item() * inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                model.load_state_dict(torch.load('best_model.pt'))
                break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test.to(device)).cpu().numpy()
    
    return model, y_pred_scaled

def main():
    # Configuration
    DATA_DIR = "synthetic_data"
    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Process all JSON files in nested directory structure
    print("Processing JSON files...")
    all_features, file_paths = process_nested_directory(DATA_DIR)
    
    if len(all_features) < 100:
        raise ValueError(f"Insufficient data - only {len(all_features)} samples found (need at least 100)")
    
    print(f"\nSuccessfully processed {len(all_features)} program files")
    print(f"Total features per sample: {len(create_sequential_representation(all_features[0]))}")
    
    # Prepare data
    print("\nPreparing training data...")
    X_train, y_train, X_test, y_test, scaler_X, scaler_y, input_size = prepare_data(all_features, TEST_SIZE)
    
    # Train and evaluate model
    print("\nTraining model...")
    model, y_pred_scaled = train_and_evaluate(X_train, y_train, X_test, y_test, input_size)
    
    # Inverse transform predictions
    y_pred = np.expm1(scaler_y.inverse_transform(y_pred_scaled))
    y_true = np.expm1(scaler_y.inverse_transform(y_test.numpy()))
    
    # Calculate metrics
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print("\nFinal Evaluation Results:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Save model and scalers
    torch.save(model.state_dict(), 'execution_predictor.pt')
    print("\nSaved model to 'execution_predictor.pt'")
    
    # Save scalers for inference
    import joblib
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')
    print("Saved scalers for inference")

if __name__ == "__main__":
    main()
