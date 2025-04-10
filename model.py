import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Extract execution time from JSON
def get_execution_time(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw_content = f.read()
            content = raw_content.decode('utf-8', errors='replace').replace('\0', '')
            data = json.loads(content)
        
        if 'programming_details' not in data:
            print(f"Error: 'programming_details' key not found in {file_path}")
            return None
        
        schedules = data["scheduling_data"]
        for item in schedules:
            if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
                execution_time = item.get('value')
                if execution_time is not None:
                    return float(execution_time)
        
        print(f"Warning: 'total_execution_time_ms' not found in {file_path}")
        return schedules[-1]["value"]
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

# Save scaler parameters (corrected for RobustScaler)
def save_scaler_params(scaler_X, scaler_y, is_log_transformed):
    scaler_X_data = {
        "feature_names": list(scaler_X.feature_names_in_),
        "centers": scaler_X.center_.tolist(),
        "scales": scaler_X.scale_.tolist()
    }
    with open("scaler_X.json", "w") as f:
        json.dump(scaler_X_data, f)

    scaler_y_data = {
        "mean": float(scaler_y.mean_[0]),
        "scale": float(scaler_y.scale_[0]),
        "is_log_transformed": is_log_transformed
    }
    with open("scaler_y.json", "w") as f:
        json.dump(scaler_y_data, f)

# Extract features from a single file
def extract_features_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    execution_time = get_execution_time(file_path)
    if execution_time is None or execution_time <= 0:
        print(f"Warning: Invalid execution time in {file_path}")
        return None
    
    nodes_features = []
    edges_features = []
    programming_details = data.get("programming_details", {})
    
    if 'Nodes' in programming_details:
        for node in programming_details['Nodes']:
            node_feature = {'Name': node.get('Name', '')}
            if 'Details' in node and 'Op histogram' in node['Details']:
                op_hist = node['Details']['Op histogram']
                for op_line in op_hist:
                    parts = op_line.strip().split(':')
                    if len(parts) == 2:
                        op_name = parts[0].strip().lower()
                        op_count = int(parts[1].strip())
                        node_feature[f'op_{op_name}'] = op_count
            nodes_features.append(node_feature)
    
    if 'Edges' in programming_details:
        for edge in programming_details['Edges']:
            edge_feature = {
                'From': edge.get('From', ''),
                'To': edge.get('To', ''),
                'Name': edge.get('Name', '')
            }
            edges_features.append(edge_feature)
    
    scheduling_features = data.get("scheduling_data", programming_details.get('Schedules', []))
    features = {
        'execution_time': execution_time,
        'nodes_count': len(nodes_features),
        'edges_count': len(edges_features),
        'scheduling_count': len(scheduling_features),
        'node_edge_ratio': len(nodes_features) / (len(edges_features) + 1e-8),
    }
    
    op_counts = {}
    for node in nodes_features:
        for key, value in node.items():
            if key.startswith('op_'):
                op_counts[key] = op_counts.get(key, 0) + value
    features.update(op_counts)
    
    if scheduling_features:
        metrics = [
            'bytes_at_production', 'bytes_at_realization', 'bytes_at_root', 'bytes_at_task',
            'inner_parallelism', 'outer_parallelism', 'num_productions', 'num_realizations',
            'num_scalars', 'num_vectors', 'points_computed_total', 'working_set'
        ]
        for metric in metrics:
            if metric in scheduling_features[0]:
                features[f'sched_{metric}'] = scheduling_features[0][metric]
        
        total_bytes = sum(sf.get('bytes_at_production', 0) for sf in scheduling_features)
        total_vectors = sum(sf.get('num_vectors', 0) for sf in scheduling_features)
        total_parallelism = sum(sf.get('inner_parallelism', 0) * sf.get('outer_parallelism', 1) for sf in scheduling_features)
        
        features.update({
            'total_bytes_at_production': total_bytes,
            'total_vectors': total_vectors,
            'total_parallelism': total_parallelism,
            'bytes_per_vector': total_bytes / (total_vectors + 1e-8),
            'memory_pressure': scheduling_features[0].get('working_set', 0) / (scheduling_features[0].get('bytes_at_production', 1) + 1e-8),
            'parallelism_per_node': total_parallelism / (len(nodes_features) + 1e-8),
        })
    
    if nodes_features:
        features.update({
            'avg_ops_per_node': sum(op_counts.values()) / len(nodes_features),
            'op_diversity': len(op_counts) / len(nodes_features),
        })
    
    return features

# Process a single directory
def process_directory(directory_path):
    all_features = []
    file_names = []
    json_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.json')])
    
    for filename in json_files:
        file_path = os.path.join(directory_path, filename)
        features = extract_features_from_file(file_path)
        if features:
            all_features.append(features)
            file_names.append(filename)
    
    return all_features, file_names

# Process main directory with subdirectories
def process_main_directory(main_dir):
    all_features = []
    all_file_names = []
    subdirs = sorted([d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))])
    
    if not subdirs:
        raise ValueError(f"No subdirectories found in {main_dir}")
    
    for subdir in subdirs:
        subdir_path = os.path.join(main_dir, subdir)
        features, file_names = process_directory(subdir_path)
        all_features.extend(features)
        all_file_names.extend([os.path.join(subdir, fname) for fname in file_names])
        print(f"Processed {subdir}: {len(features)} files")
    
    if len(all_features) < 50:
        raise ValueError(f"Found {len(all_features)} files, expected at least 50")
    
    combined = list(zip(all_features, all_file_names))
    random.shuffle(combined)
    all_features, all_file_names = zip(*combined)
    
    test_size = 50
    train_features = all_features[:-test_size]
    test_features = all_features[-test_size:]
    train_file_names = all_file_names[:-test_size]
    test_file_names = all_file_names[-test_size:]
    
    print(f"Total: {len(all_features)}, Train: {len(train_features)}, Test: {len(test_features)}")
    return train_features, test_features, list(test_file_names)

# Clean and transform features
def clean_and_transform_features(train_features, test_features):
    all_features_df = pd.DataFrame(train_features + test_features).fillna(0)
    
    constant_cols = [col for col in all_features_df.columns if col != 'execution_time' and all_features_df[col].nunique() <= 1]
    all_features_df.drop(columns=constant_cols, inplace=True)
    print(f"Dropped {len(constant_cols)} constant columns")
    
    all_features_df['execution_time'] = np.clip(all_features_df['execution_time'], 1e-3, 1e6)
    all_features_df['execution_time_log'] = np.log1p(all_features_df['execution_time'])
    
    if 'total_vectors' in all_features_df and all_features_df['total_vectors'].max() > 0:
        all_features_df['log_bytes_per_vector'] = np.log1p(all_features_df['total_bytes_at_production'] / (all_features_df['total_vectors'] + 1e-8))
    if 'nodes_count' in all_features_df and 'edges_count' in all_features_df:
        all_features_df['node_edge_interaction'] = all_features_df['nodes_count'] * all_features_df['edges_count']
    
    numeric_cols = all_features_df.select_dtypes(include=['number']).columns
    all_features_df = all_features_df[numeric_cols]
    
    train_df = all_features_df.iloc[:len(train_features)]
    test_df = all_features_df.iloc[len(train_features):]
    
    return train_df, test_df

# Prepare data with robust scaling
def prepare_data_for_model(train_features, test_features):
    train_df, test_df = clean_and_transform_features(train_features, test_features)
    
    y_train = train_df['execution_time_log'].values.reshape(-1, 1)
    y_test = test_df['execution_time_log'].values.reshape(-1, 1)
    X_train = train_df.drop(['execution_time', 'execution_time_log'], axis=1)
    X_test = test_df.drop(['execution_time', 'execution_time_log'], axis=1)
    
    scaler_X = RobustScaler(quantile_range=(25.0, 75.0))
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Input feature dimension: {X_train_scaled.shape[1]}")
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_X, scaler_y, X_train_scaled.shape[1], True

# Enhanced LSTM model (unchanged from your snippet)
class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], output_size=1, num_heads=4, dropout_rate=0.4):
        super(EnhancedLSTMModel, self).__init__()
        
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        self.lstm_layers.append(nn.LSTM(input_size, hidden_sizes[0], batch_first=True, bidirectional=True))
        self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(nn.LSTM(hidden_sizes[i-1] * 2, hidden_sizes[i], batch_first=True, bidirectional=True))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        self.attention = nn.MultiheadAttention(hidden_sizes[-1] * 2, num_heads=num_heads, dropout=dropout_rate)
        self.attn_fc = nn.Linear(hidden_sizes[-1] * 2, hidden_sizes[-1])
        
        self.fc_layers = nn.ModuleList([
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.Linear(hidden_sizes[-1] // 2, hidden_sizes[-1] // 4),
            nn.Linear(hidden_sizes[-1] // 4, output_size)
        ])
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_sizes[-1] // 2),
            nn.BatchNorm1d(hidden_sizes[-1] // 4)
        ])
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        for i, (lstm, dropout) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            lstm_out, _ = lstm(x if i == 0 else lstm_out)
            lstm_out = dropout(lstm_out)
        
        lstm_out = lstm_out.transpose(0, 1)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.transpose(0, 1)
        context = self.attn_fc(attn_out[:, -1, :])
        
        out = context
        for i, (fc, bn) in enumerate(zip(self.fc_layers[:-1], self.bn_layers)):
            out = fc(out)
            out = bn(out)
            out = self.relu(out)
            out = self.dropout(out)
        
        out = self.fc_layers[-1](out)
        return out

# (Assuming train_model, evaluate_model, create_data_loaders are defined elsewhere as in your previous snippets)

# Main function
def main(main_dir):
    print(f"Processing {main_dir}")
    train_features, test_features, test_file_names = process_main_directory(main_dir)
    
    X_train, y_train, X_test, y_test, scaler_X, scaler_y, input_size, is_log_transformed = prepare_data_for_model(train_features, test_features)
    save_scaler_params(scaler_X, scaler_y, is_log_transformed)
    
    model = EnhancedLSTMModel(input_size=input_size, hidden_sizes=[256, 128, 64], num_heads=4, dropout_rate=0.4)
    criterion = nn.HuberLoss(delta=1.0)
    
    print("Training with cross-validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    best_model = None
    best_val_loss = float('inf')
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\nFold {fold+1}/5")
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        fold_model = EnhancedLSTMModel(input_size=input_size, hidden_sizes=[256, 128, 64], num_heads=4, dropout_rate=0.4)
        _, val_losses, trained_model = train_model(fold_model, X_tr, y_tr, X_val, y_val, criterion)
        
        fold_val_loss = min(val_losses)
        cv_scores.append(fold_val_loss)
        if fold_val_loss < best_val_loss:
            best_val_loss = fold_val_loss
            best_model = trained_model
    
    print(f"\nCross-validation MAPE scores: {cv_scores}, Mean: {np.mean(cv_scores):.4f}")
    
    print("\nFinal training on full dataset...")
    train_losses, val_losses, best_model = train_model(best_model, X_train, y_train, X_test, y_test, criterion)
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual, avg_actual, avg_predicted = evaluate_model(best_model, X_test, y_test, scaler_y, test_file_names, is_log_transformed)
    
    torch.jit.save(torch.jit.trace(best_model.cpu(), torch.randn(1, 1, input_size)), "lstm_model.pt")
    print("Model saved as 'lstm_model.pt'")
    
    print(f"\nSummary: Avg Actual: {avg_actual:.2f} ms, Avg Predicted: {avg_predicted:.2f} ms")
    return best_model, y_scaler, y_test_actual, y_pred_actual, avg_actual, avg_predicted

if __name__ == "__main__":
    main_dir = "synthetic_data"
    model, y_scaler, y_test_actual, y_pred_actual, avg_actual, avg_predicted = main(main_dir)
