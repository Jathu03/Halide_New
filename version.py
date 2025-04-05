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

# Configuration constants
MAX_NODES = 50
MAX_EDGES = 100
MAX_SCHEDULES = 20
NODE_FEATURES = len(OP_TYPES)
EDGE_FEATURES = 2  # data_size, is_vector
SCHEDULE_FEATURES = 4  # bytes, parallelism, working_set, vectors
AGG_FEATURES = 14  # Number of aggregated features

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

def process_nested_directory(root_dir, max_samples=None):
    """Process all JSON files in nested directory structure"""
    all_features = []
    file_paths = []
    
    # Walk through all subdirectories
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                if max_samples and len(all_features) >= max_samples:
                    break
                    
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
    
    return features

def create_sequential_representation(features):
    """Convert features into fixed-size sequential format"""
    # Initialize empty arrays for each feature type
    node_feats = np.zeros((MAX_NODES, NODE_FEATURES), dtype=np.float32)
    edge_feats = np.zeros((MAX_EDGES, EDGE_FEATURES), dtype=np.float32)
    sched_feats = np.zeros((MAX_SCHEDULES, SCHEDULE_FEATURES), dtype=np.float32)
    
    # Fill node features
    for i, node in enumerate(features['node_features'][:MAX_NODES]):
        for op, count in node['op_type_counts'].items():
            if op in OP_TYPES:
                node_feats[i, OP_TYPES.index(op)] = min(count, 1)  # Binary indicator
    
    # Fill edge features
    for i, edge in enumerate(features['edge_features'][:MAX_EDGES]):
        edge_feats[i, 0] = np.log1p(edge['data_size'])
        edge_feats[i, 1] = edge['is_vector']
    
    # Fill schedule features
    for i, sched in enumerate(features['schedule_features'][:MAX_SCHEDULES]):
        sched_feats[i, 0] = np.log1p(sched['bytes'] + 1)
        sched_feats[i, 1] = np.log1p(sched['parallelism'] + 1)
        sched_feats[i, 2] = np.log1p(sched['working_set'] + 1)
        sched_feats[i, 3] = np.log1p(sched['vectors'] + 1)
    
    # Compute aggregated features
    num_nodes = min(len(features['node_features']), MAX_NODES)
    num_edges = min(len(features['edge_features']), MAX_EDGES)
    num_schedules = min(len(features['schedule_features']), MAX_SCHEDULES)
    
    # Count operations
    op_counts = {}
    for node in features['node_features'][:MAX_NODES]:
        for op, count in node['op_type_counts'].items():
            op_counts[op] = op_counts.get(op, 0) + count
    
    agg_feats = np.array([
        num_nodes / MAX_NODES,
        num_edges / MAX_EDGES,
        num_schedules / MAX_SCHEDULES,
        np.log1p(sum(op_counts.values())),
        len(op_counts) / len(OP_TYPES),
        len(op_counts) / (num_nodes + 1e-8),
        np.log1p(sum(e['data_size'] for e in features['edge_features'][:MAX_EDGES]) + 1),
        sum(e['is_vector'] for e in features['edge_features'][:MAX_EDGES]) / (num_edges + 1e-8),
        np.log1p(sum(s['bytes'] for s in features['schedule_features'][:MAX_SCHEDULES]) + 1),
        np.log1p(sum(s['parallelism'] for s in features['schedule_features'][:MAX_SCHEDULES]) + 1),
        np.log1p(sum(s['working_set'] for s in features['schedule_features'][:MAX_SCHEDULES]) + 1),
        np.log1p(sum(s['vectors'] for s in features['schedule_features'][:MAX_SCHEDULES]) + 1),
        np.log1p((sum(s['bytes'] for s in features['schedule_features'][:MAX_SCHEDULES]) / 
                (sum(s['vectors'] for s in features['schedule_features'][:MAX_SCHEDULES]) + 1e-8) + 1),
        (sum(s['working_set'] for s in features['schedule_features'][:MAX_SCHEDULES]) / 
         (sum(s['bytes'] for s in features['schedule_features'][:MAX_SCHEDULES]) + 1e-8)
    ], dtype=np.float32)
    
    # Combine all features into a single flat array
    sequential_features = np.concatenate([
        node_feats.flatten(),
        edge_feats.flatten(),
        sched_feats.flatten(),
        agg_feats.flatten()
    ])
    
    return sequential_features

def prepare_data(all_features, file_paths, test_size=10):
    """Prepare data with fixed test size of 10 samples"""
    # Convert features to sequential format
    X = np.array([create_sequential_representation(f) for f in all_features])
    
    # Get execution times and apply log transform
    y = np.log1p(np.array([f['execution_time'] for f in all_features], dtype=np.float32))
    
    # Split data - last 10 samples for test
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    test_files = file_paths[-test_size:]
    
    # Scale features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
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
            scaler_X, scaler_y, X_train_scaled.shape[1], test_files)

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
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, 1)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        return self.fc(context)

def train_and_evaluate(X_train, y_train, X_test, y_test, input_size, test_files, epochs=50, patience=10):
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
    
    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test.to(device)).cpu().numpy()
    
    return y_pred_scaled

def main():
    # Configuration
    DATA_DIR = "synthetic_data"
    TEST_SIZE = 10  # Test on exactly 10 samples
    SAMPLE_LIMIT = 1000  # Limit total samples for faster testing
    
    # Process JSON files (limit to SAMPLE_LIMIT for testing)
    print(f"Processing up to {SAMPLE_LIMIT} JSON files...")
    all_features, file_paths = process_nested_directory(DATA_DIR, max_samples=SAMPLE_LIMIT)
    
    if len(all_features) < TEST_SIZE + 10:  # Need at least TEST_SIZE + 10 for meaningful training
        raise ValueError(f"Insufficient data - only {len(all_features)} samples found (need at least {TEST_SIZE + 10})")
    
    print(f"\nSuccessfully processed {len(all_features)} program files")
    
    # Prepare data with fixed test size
    print("\nPreparing training data...")
    X_train, y_train, X_test, y_test, scaler_X, scaler_y, input_size, test_files = prepare_data(
        all_features, file_paths, test_size=TEST_SIZE
    )
    
    print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Input feature dimension: {input_size}")
    
    # Train and evaluate model
    print("\nTraining model...")
    y_pred_scaled = train_and_evaluate(X_train, y_train, X_test, y_test, input_size, test_files)
    
    # Inverse transform predictions
    y_pred = np.expm1(scaler_y.inverse_transform(y_pred_scaled))
    y_true = np.expm1(scaler_y.inverse_transform(y_test.numpy()))
    
    # Calculate metrics
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print("\nTest Set Evaluation Results:")
    print(f"MSE: {mse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Print individual test results
    print("\nDetailed Predictions for Test Samples:")
    for i in range(len(test_files)):
        print(f"\nFile: {test_files[i]}")
        print(f"Actual: {y_true[i][0]:.2f} ms, Predicted: {y_pred[i][0]:.2f} ms")
        print(f"Error: {abs(y_true[i][0] - y_pred[i][0]) / y_true[i][0] * 100:.2f}%")
    
    # Save model and scalers
    torch.save({
        'model_state_dict': torch.load('best_model.pt'),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'op_types': OP_TYPES,
        'input_size': input_size
    }, 'execution_predictor.pth')
    print("\nSaved model and scalers to 'execution_predictor.pth'")

if __name__ == "__main__":
    main()
