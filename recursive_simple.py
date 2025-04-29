import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import random

# ---- Fixed set of features ----
FIXED_FEATURES = [
    'cache_hits', 'cache_misses', 'execution_time_ms',
    'sched_num_realizations', 'sched_num_productions', 'sched_points_computed_total',
    'sched_innermost_loop_extent', 'sched_inner_parallelism', 'sched_outer_parallelism',
    'sched_bytes_at_realization', 'sched_bytes_at_production', 'sched_bytes_at_root',
    'sched_unique_bytes_read_per_realization', 'sched_working_set', 'sched_vector_size',
    'sched_num_vectors', 'sched_num_scalars', 'sched_bytes_at_task', 'sched_working_set_at_task',
    'sched_working_set_at_production', 'sched_working_set_at_realization', 'sched_working_set_at_root',
    'total_parallelism', 'scheduling_count', 'total_bytes_at_production', 'total_vectors',
    'computation_efficiency', 'memory_pressure', 'memory_utilization_ratio', 'bytes_processing_rate',
    'bytes_per_parallelism', 'bytes_per_vector', 'nodes_count', 'edges_count', 'node_edge_ratio',
    'nodes_per_schedule', 'op_diversity', 'op_add', 'op_sub', 'op_mul', 'op_div'
    # Add more as needed
]

def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_features(json_data):
    """Extracts a fixed set of features from a JSON structure."""
    features = {}
    global_node = next((child for child in json_data['children'] if child['name'] == 'Global Features'), None)
    if global_node:
        features['cache_hits'] = global_node.get('cache_hits', 0)
        features['cache_misses'] = global_node.get('cache_misses', 0)
        features['execution_time_ms'] = global_node.get('execution_time_ms', 0)
    # Op histogram
    for node in json_data['children']:
        if 'op_histogram' in node:
            for op, count in node['op_histogram'].items():
                features[f'op_{op.lower()}'] = features.get(f'op_{op.lower()}', 0) + count
    # Scheduling features
    scheduling_keys = [
        'num_realizations', 'num_productions', 'points_computed_total', 'innermost_loop_extent',
        'inner_parallelism', 'outer_parallelism', 'bytes_at_realization', 'bytes_at_production',
        'bytes_at_root', 'unique_bytes_read_per_realization', 'working_set', 'vector_size',
        'num_vectors', 'num_scalars', 'bytes_at_task', 'working_set_at_task', 'working_set_at_production',
        'working_set_at_realization', 'working_set_at_root'
    ]
    scheduling_sums = {}
    for node in json_data['children']:
        if 'scheduling' in node:
            for key in scheduling_keys:
                scheduling_sums[key] = scheduling_sums.get(key, 0) + node['scheduling'].get(key, 0)
    for key in scheduling_keys:
        features[f'sched_{key}'] = scheduling_sums.get(key, 0)
    # Derived features
    features['total_parallelism'] = features.get('sched_inner_parallelism', 0) + features.get('sched_outer_parallelism', 0)
    features['scheduling_count'] = features.get('sched_num_realizations', 0) + features.get('sched_num_productions', 0)
    features['total_bytes_at_production'] = features.get('sched_bytes_at_production', 0)
    features['total_vectors'] = features.get('sched_num_vectors', 0)
    features['computation_efficiency'] = features.get('sched_points_computed_total', 0) / (features.get('sched_bytes_at_realization', 1) or 1)
    features['memory_pressure'] = features.get('sched_working_set', 0) / (features.get('sched_bytes_at_root', 1) or 1)
    features['memory_utilization_ratio'] = features.get('sched_unique_bytes_read_per_realization', 0) / (features.get('sched_bytes_at_task', 1) or 1)
    features['bytes_processing_rate'] = features.get('sched_bytes_at_realization', 0) / (features.get('execution_time_ms', 1) or 1)
    features['bytes_per_parallelism'] = features.get('sched_bytes_at_task', 0) / (features.get('total_parallelism', 1) or 1)
    features['bytes_per_vector'] = features.get('sched_bytes_at_realization', 0) / (features.get('sched_num_vectors', 1) or 1)
    nodes_count = len(json_data['children'])
    edges_count = sum(len(node.get('children', [])) for node in json_data['children'])
    features['nodes_count'] = nodes_count
    features['edges_count'] = edges_count
    features['node_edge_ratio'] = nodes_count / (edges_count + 1)
    features['nodes_per_schedule'] = nodes_count / (features.get('scheduling_count', 1) or 1)
    features['op_diversity'] = len([k for k, v in features.items() if k.startswith('op_') and v > 0])
    # Fill missing features with 0
    fixed_features = {key: features.get(key, 0.0) for key in FIXED_FEATURES}
    return fixed_features

def process_tree_output_directory(main_dir):
    """Walks the directory, extracts features from all valid JSON files."""
    all_features, file_names, skipped_files = [], [], []
    for root, dirs, files in os.walk(main_dir):
        if 'tree_representation.json' in files:
            file_path = os.path.join(root, 'tree_representation.json')
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                features = extract_features(json_data)
                if features['execution_time_ms'] > 0 and np.isfinite(features['execution_time_ms']):
                    all_features.append(features)
                    file_names.append(file_path)
                else:
                    skipped_files.append(file_path)
            except Exception:
                skipped_files.append(file_path)
    if not all_features:
        raise ValueError("No valid JSON files with valid execution times found in Tree_Output directory.")
    combined = list(zip(all_features, file_names))
    random.shuffle(combined)
    all_features, file_names = zip(*combined)
    test_size = min(30, len(all_features))
    train_features = all_features[:-test_size]
    test_features = all_features[-test_size:]
    train_file_names = file_names[:-test_size]
    test_file_names = file_names[-test_size:]
    print(f"Training files: {len(train_features)}")
    print(f"Testing files: {len(test_features)}")
    return train_features, test_features, list(test_file_names)

def prepare_data_for_model(train_features, test_features):
    """Prepares PyTorch tensors and scalers for model input."""
    train_sequences = [np.array([[features.get(key, 0.0) for key in FIXED_FEATURES]]) for features in train_features]
    test_sequences = [np.array([[features.get(key, 0.0) for key in FIXED_FEATURES]]) for features in test_features]
    train_sequences_padded = torch.FloatTensor(np.array(train_sequences))
    test_sequences_padded = torch.FloatTensor(np.array(test_sequences))
    train_scalar_df = pd.DataFrame(train_features).fillna(0)
    test_scalar_df = pd.DataFrame(test_features).fillna(0)
    constant_columns = [col for col in train_scalar_df.columns if train_scalar_df[col].nunique() == 1]
    train_scalar_df = train_scalar_df.drop(columns=constant_columns)
    test_scalar_df = test_scalar_df.drop(columns=constant_columns)
    y_train_raw = np.array([f['execution_time_ms'] for f in train_features])
    y_test_raw = np.array([f['execution_time_ms'] for f in test_features])
    y_train = np.log1p(y_train_raw).reshape(-1, 1)
    y_test = np.log1p(y_test_raw).reshape(-1, 1)
    scaler_X_scalar = RobustScaler()
    scaler_y = RobustScaler()
    train_scalar_scaled = scaler_X_scalar.fit_transform(train_scalar_df)
    test_scalar_scaled = scaler_X_scalar.transform(test_scalar_df)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    train_scalar_tensor = torch.FloatTensor(train_scalar_scaled)
    test_scalar_tensor = torch.FloatTensor(test_scalar_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    return (train_sequences_padded, train_scalar_tensor, y_train_tensor,
            test_sequences_padded, test_scalar_tensor, y_test_tensor,
            scaler_y, train_sequences_padded.shape[2], train_scalar_tensor.shape[1], train_scalar_df.columns)

class EnhancedRecursiveLSTMModel(nn.Module):
    """Bidirectional multi-layer LSTM with scalar feature concatenation."""
    def __init__(self, seq_input_size, scalar_input_size, hidden_sizes=[128, 64], output_size=1, dropout_rate=0.2):
        super().__init__()
        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(nn.LSTM(seq_input_size, hidden_sizes[0], batch_first=True, bidirectional=True))
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(nn.LSTM(hidden_sizes[i-1]*2, hidden_sizes[i], batch_first=True, bidirectional=True))
        combined_size = hidden_sizes[-1]*2 + scalar_input_size
        self.fc1 = nn.Linear(combined_size, 64)
        self.fc2 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, seq_input, scalar_input):
        lstm_out = seq_input
        for lstm in self.lstm_layers:
            lstm_out, _ = lstm(lstm_out)
            lstm_out = self.dropout(lstm_out)
        context = lstm_out.mean(dim=1)
        combined = torch.cat((context, scalar_input), dim=1)
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        output = self.fc2(x)
        return output

def train_model(model, train_loader, test_loader, optimizer, num_epochs=100, patience=10, use_amp=False):
    """Trains the model with early stopping and optional mixed precision."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    for epoch in range(num_epochs):
        model.train()
        for seq_inputs, scalar_inputs, targets in train_loader:
            seq_inputs, scalar_inputs, targets = seq_inputs.to(device), scalar_inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(seq_inputs, scalar_inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(seq_inputs, scalar_inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq_inputs, scalar_inputs, targets in test_loader:
                seq_inputs, scalar_inputs, targets = seq_inputs.to(device), scalar_inputs.to(device), targets.to(device)
                outputs = model(seq_inputs, scalar_inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * seq_inputs.size(0)
        val_loss /= len(test_loader.dataset)
        print(f"Epoch {epoch+1}: Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_lstm.pth")
        else:
            patience_counter += 1
        if patience_counter > patience:
            print("Early stopping.")
            break
    model.load_state_dict(torch.load("best_lstm.pth"))
    return model

def evaluate_model(model, X_test_seq, X_test_scalar, y_test, y_scaler, file_names_test):
    """Evaluates model and prints detailed metrics."""
    from sklearn.metrics import r2_score
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    X_test_seq, X_test_scalar = X_test_seq.to(device), X_test_scalar.to(device)
    with torch.no_grad():
        y_pred_scaled = model(X_test_seq, X_test_scalar)
    y_pred_scaled = y_pred_scaled.cpu().numpy()
    y_test = y_test.cpu().numpy()
    y_test_transformed = y_scaler.inverse_transform(y_test)
    y_pred_transformed = y_scaler.inverse_transform(y_pred_scaled)
    y_test_actual = np.expm1(y_test_transformed)
    y_pred_actual = np.expm1(y_pred_transformed)
    for i, file_path in enumerate(file_names_test):
        pred = max(y_pred_actual[i][0], 0)
        actual = y_test_actual[i][0]
        error_pct = abs(actual - pred) / actual * 100 if actual > 0 else 0
        print(f"{file_path}: Actual={actual:.2f} ms, Predicted={pred:.2f} ms, Error={error_pct:.2f}%")
    mse = np.mean((y_test_actual - y_pred_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / (y_test_actual + 1e-8))) * 100
    r2 = r2_score(y_test_actual, y_pred_actual)
    print("\nOverall Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2 Score: {r2:.4f}")
    return y_test_actual, y_pred_actual

def main(main_dir,
         batch_size=64,
         learning_rate=0.0005,
         weight_decay=1e-4,
         num_epochs=100,
         patience=10,
         use_amp=False,
         seed=42):
    """Main pipeline for data extraction, training, and evaluation."""
    print(f"Processing main directory: {main_dir}")
    set_random_seed(seed)
    train_features, test_features, test_file_names = process_tree_output_directory(main_dir)
    if len(train_features) == 0 or len(test_features) == 0:
        print("Error: No valid training or test data found")
        return None
    (train_sequences, train_scalar, y_train,
     test_sequences, test_scalar, y_test,
     y_scaler, seq_input_size, scalar_input_size, feature_columns) = prepare_data_for_model(train_features, test_features)
    train_loader = DataLoader(TensorDataset(train_sequences, train_scalar, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_sequences, test_scalar, y_test), batch_size=batch_size)
    model = EnhancedRecursiveLSTMModel(seq_input_size=seq_input_size, scalar_input_size=scalar_input_size)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model = train_model(model, train_loader, test_loader, optimizer, num_epochs=num_epochs, patience=patience, use_amp=use_amp)
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(
        model, test_sequences, test_scalar, y_test,
        y_scaler, test_file_names
    )
    print(f"\nSummary for Comparison:")
    print(f"Model: EnhancedRecursiveLSTM")
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "Tree_Output"
    main(main_dir)
