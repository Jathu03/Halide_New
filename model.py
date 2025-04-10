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

# Save scaler parameters
def save_scaler_params(scaler_X, scaler_y, is_log_transformed):
    scaler_X_data = {
        "feature_names": list(scaler_X.feature_names_in_),
        "means": scaler_X.mean_.tolist(),
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

# Enhanced feature extraction
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

# Process directory
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

# Enhanced data cleaning and transformation
def clean_and_transform_features(train_features, test_features):
    all_features_df = pd.DataFrame(train_features + test_features).fillna(0)
    
    # Remove constant columns
    constant_cols = [col for col in all_features_df.columns if col != 'execution_time' and all_features_df[col].nunique() <= 1]
    all_features_df.drop(columns=constant_cols, inplace=True)
    print(f"Dropped {len(constant_cols)} constant columns")
    
    # Clip execution time outliers
    all_features_df['execution_time'] = np.clip(all_features_df['execution_time'], 1e-3, 1e6)
    all_features_df['execution_time_log'] = np.log1p(all_features_df['execution_time'])
    
    # Additional feature engineering
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

# Enhanced LSTM model with bidirectional LSTM and multi-head attention
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
        
        # Multi-head attention
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
        
        # Multi-head attention
        lstm_out = lstm_out.transpose(0, 1)  # [seq_len, batch, hidden]
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.transpose(0, 1)  # [batch, seq_len, hidden]
        context = self.attn_fc(attn_out[:, -1, :])  # Take last output
        
        # Fully connected layers
        out = context
        for i, (fc, bn) in enumerate(zip(self.fc_layers[:-1], self.bn_layers)):
            out = fc(out)
            out = bn(out)
            out = self.relu(out)
            out = self.dropout(out)
        
        out = self.fc_layers[-1](out)
        return out

# Data loaders
def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Training with mixed precision and cross-validation
def train_model(model, X_train, y_train, X_test, y_test, criterion, num_epochs=200, batch_size=32, patience=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()
    
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
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    model.load_state_dict(best_model_state)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('lstm_loss_enhanced.png')
    plt.show()
    
    return train_losses, val_losses, model

# Enhanced evaluation
def evaluate_model(model, X_test, y_test, y_scaler, file_names_test, is_log_transformed=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    X_test = X_test.to(device)
    with torch.no_grad():
        with autocast():
            y_pred_scaled = model(X_test)
    
    y_pred_scaled = y_pred_scaled.cpu().numpy()
    y_test = y_test.cpu().numpy()
    
    y_test_transformed = y_scaler.inverse_transform(y_test)
    y_pred_transformed = y_scaler.inverse_transform(y_pred_scaled)
    
    y_test_actual = np.expm1(y_test_transformed) if is_log_transformed else y_test_transformed
    y_pred_actual = np.expm1(y_pred_transformed) if is_log_transformed else y_pred_transformed
    y_pred_actual = np.clip(y_pred_actual, 0, None)  # Ensure non-negative predictions
    
    results_by_subfolder = {}
    for i, file_path in enumerate(file_names_test):
        subfolder = file_path.split('/')[0]
        results_by_subfolder.setdefault(subfolder, []).append({
            'file': file_path,
            'actual': y_test_actual[i][0],
            'predicted': y_pred_actual[i][0],
            'error_percentage': abs(y_test_actual[i][0] - y_pred_actual[i][0]) / (y_test_actual[i][0] + 1e-8) * 100
        })
    
    mse = np.mean((y_test_actual - y_pred_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    mape = np.mean([r['error_percentage'] for subfolder in results_by_subfolder.values() for r in subfolder])
    
    print("\nDetailed Results:")
    for subfolder, results in results_by_subfolder.items():
        print(f"\n{subfolder}:")
        for r in results:
            print(f"  File: {r['file']}, Actual: {r['actual']:.2f} ms, Predicted: {r['predicted']:.2f} ms, Error: {r['error_percentage']:.2f}%")
    
    print(f"\nOverall Performance: MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
    
    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_actual, y_pred_actual, alpha=0.5)
    plt.plot([y_test_actual.min(), y_test_actual.max()], [y_test_actual.min(), y_test_actual.max()], 'r--')
    plt.xlabel('Actual Execution Time (ms)')
    plt.ylabel('Predicted Execution Time (ms)')
    plt.title('Actual vs Predicted')
    plt.grid(True)
    plt.savefig('actual_vs_predicted.png')
    plt.show()
    
    return y_test_actual, y_pred_actual, np.mean(y_test_actual), np.mean(y_pred_actual)

# Main function with cross-validation
def main(main_dir):
    print(f"Processing {main_dir}")
    train_features, test_features, test_file_names = process_main_directory(main_dir)
    
    X_train, y_train, X_test, y_test, scaler_X, scaler_y, input_size, is_log_transformed = prepare_data_for_model(train_features, test_features)
    save_scaler_params(scaler_X, scaler_y, is_log_transformed)
    
    model = EnhancedLSTMModel(input_size=input_size, hidden_sizes=[256, 128, 64], num_heads=4, dropout_rate=0.4)
    criterion = nn.HuberLoss(delta=1.0)
    
    print("Training with cross-validation...")
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
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
