import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random
import matplotlib.pyplot as plt

# 1. Feature extraction
FIXED_FEATURES = [
    # Add all features you want to extract from each node
    'cache_hits', 'cache_misses', 'execution_time_ms', 'sched_num_realizations',
    'sched_num_productions', 'sched_points_computed_total', 'sched_innermost_loop_extent',
    'sched_inner_parallelism', 'sched_outer_parallelism', 'sched_bytes_at_realization',
    'sched_bytes_at_production', 'sched_bytes_at_root', 'sched_unique_bytes_read_per_realization',
    'sched_working_set', 'sched_vector_size', 'sched_num_vectors', 'sched_num_scalars',
    'sched_bytes_at_task', 'sched_working_set_at_task', 'sched_working_set_at_production',
    'sched_working_set_at_realization', 'sched_working_set_at_root', 'total_parallelism',
    'scheduling_count', 'total_bytes_at_production', 'total_vectors', 'computation_efficiency',
    'memory_pressure', 'memory_utilization_ratio', 'bytes_processing_rate', 'bytes_per_parallelism',
    'bytes_per_vector', 'nodes_count', 'edges_count', 'node_edge_ratio', 'nodes_per_schedule',
    'op_diversity', 'op_add', 'op_sub', 'op_mul', 'op_div', 'op_mod', 'op_eq', 'op_ne',
    'op_lt', 'op_le', 'op_or', 'op_and', 'op_not', 'op_min', 'op_max', 'op_constant',
    'op_variable', 'op_funccall', 'op_imagecall', 'op_externcall', 'op_let', 'op_param',
    'memory_transpose_0', 'memory_transpose_1', 'memory_transpose_2', 'memory_transpose_3',
    'memory_slice_0', 'memory_slice_1', 'memory_slice_2', 'memory_slice_3',
    'memory_broadcast_0', 'memory_broadcast_1', 'memory_broadcast_2', 'memory_broadcast_3',
    'memory_pointwise_0', 'memory_pointwise_1', 'memory_pointwise_2', 'memory_pointwise_3'
]

def extract_features(json_data):
    features = {}
    global_node = next((child for child in json_data['children'] if child['name'] == 'Global Features'), None)
    if global_node:
        features['cache_hits'] = global_node.get('cache_hits', 0)
        features['cache_misses'] = global_node.get('cache_misses', 0)
        features['execution_time_ms'] = global_node.get('execution_time_ms', 0)
    op_histogram = {}
    for node in json_data['children']:
        if 'op_histogram' in node:
            for op, count in node['op_histogram'].items():
                features[f'op_{op.lower()}'] = features.get(f'op_{op.lower()}', 0) + count
    memory_patterns = {}
    for node in json_data['children']:
        if 'memory_patterns' in node:
            for pattern, values in node['memory_patterns'].items():
                for i, val in enumerate(values):
                    k = f'memory_{pattern.lower()}_{i}'
                    features[k] = features.get(k, 0) + val
    scheduling_keys = [
        'num_realizations', 'num_productions', 'points_computed_total', 'innermost_loop_extent',
        'inner_parallelism', 'outer_parallelism', 'bytes_at_realization', 'bytes_at_production',
        'bytes_at_root', 'unique_bytes_read_per_realization', 'working_set', 'vector_size',
        'num_vectors', 'num_scalars', 'bytes_at_task', 'working_set_at_task', 'working_set_at_production',
        'working_set_at_realization', 'working_set_at_root'
    ]
    scheduling_sums = {}
    node_count = 0
    for node in json_data['children']:
        if 'scheduling' in node:
            node_count += 1
            for key in scheduling_keys:
                scheduling_sums[key] = scheduling_sums.get(key, 0) + node['scheduling'].get(key, 0)
    for key in scheduling_keys:
        features[f'sched_{key}'] = scheduling_sums.get(key, 0)
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
    fixed_features = {key: features.get(key, 0.0) for key in FIXED_FEATURES}
    return fixed_features

def process_tree_output_directory(main_dir):
    all_features = []
    file_names = []
    skipped_files = []
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
            except Exception as e:
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
    train_sequences = [np.array([[features.get(key, 0.0) for key in FIXED_FEATURES]]) for features in train_features]
    test_sequences = [np.array([[features.get(key, 0.0) for key in FIXED_FEATURES]]) for features in test_features]
    train_sequences_padded = torch.FloatTensor(np.array(train_sequences))
    test_sequences_padded = torch.FloatTensor(np.array(test_sequences))
    train_scalar_df = pd.DataFrame(train_features)
    test_scalar_df = pd.DataFrame(test_features)
    train_scalar_df = train_scalar_df.fillna(0)
    test_scalar_df = test_scalar_df.fillna(0)
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

# 2. Model
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
    def forward(self, x):
        batch_size = x.shape[0]
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.transpose(-1, -2)) / self.scale.to(x.device)
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        out = torch.matmul(attention, V).permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, -1, self.hidden_size)
        out = self.fc_out(out)
        return out

class EnhancedRecursiveLSTMModel(nn.Module):
    def __init__(self, seq_input_size, scalar_input_size, hidden_sizes=[512, 256, 128], output_size=1, dropout_rate=0.2, num_heads=8):
        super(EnhancedRecursiveLSTMModel, self).__init__()
        self.lstm_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        self.lstm_layers.append(nn.LSTM(seq_input_size, hidden_sizes[0], batch_first=True, bidirectional=True))
        self.ln_layers.append(nn.LayerNorm(hidden_sizes[0] * 2))
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(nn.LSTM(hidden_sizes[i-1] * 2, hidden_sizes[i], batch_first=True, bidirectional=True))
            self.ln_layers.append(nn.LayerNorm(hidden_sizes[i] * 2))
        self.attention = MultiHeadAttention(hidden_sizes[-1] * 2, num_heads, dropout_rate)
        combined_size = hidden_sizes[-1] * 2 + scalar_input_size
        self.fc1 = nn.Linear(combined_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.ln3 = nn.LayerNorm(64)
        self.output_layer = nn.Linear(64, output_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.residual_proj = nn.Linear(combined_size, 64) if combined_size != 64 else None
    def forward(self, seq_input, scalar_input):
        lstm_out = seq_input
        for lstm, ln in zip(self.lstm_layers, self.ln_layers):
            lstm_out, _ = lstm(lstm_out)
            lstm_out = ln(lstm_out)
            lstm_out = self.dropout(lstm_out)
        attn_out = self.attention(lstm_out)
        context = attn_out.mean(dim=1)
        combined = torch.cat((context, scalar_input), dim=1)
        x = self.fc1(combined)
        x = self.bn1(x)
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.ln2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.ln3(x)
        x = self.gelu(x)
        residual = combined if self.residual_proj is None else self.residual_proj(combined)
        x = x + residual
        x = self.dropout(x)
        output = self.output_layer(x)
        return output

def custom_loss(outputs, targets, scalar_inputs, feature_indices, feature_importances, huber_delta=0.5, mae_weight=0.3, l1_lambda=1e-5):
    huber = nn.HuberLoss(delta=huber_delta)(outputs, targets)
    mae = torch.mean(torch.abs(outputs - targets))
    l1_reg = sum(param.abs().sum() for param in model.parameters()) * l1_lambda
    weights = torch.ones_like(targets)
    for feature, idx in feature_indices.items():
        if idx != -1 and feature in feature_importances:
            feature_vals = scalar_inputs[:, idx]
            importance = feature_importances[feature]
            weights = torch.where(
                feature_vals > 1.0,
                weights * (1.0 + importance * 2.0),
                weights
            )
    weighted_huber = (huber * weights).mean()
    weighted_mae = (mae * weights).mean()
    return weighted_huber + mae_weight * weighted_mae + l1_reg

def create_data_loaders(train_sequences, train_scalar, y_train, test_sequences, test_scalar, y_test, batch_size=64):
    train_dataset = TensorDataset(train_sequences, train_scalar, y_train)
    test_dataset = TensorDataset(test_sequences, test_scalar, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, feature_indices, feature_importances, num_epochs=1000, patience=50, accumulation_steps=2, checkpoint_path='recursive.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for i, (seq_inputs, scalar_inputs, targets) in enumerate(train_loader):
            seq_inputs, scalar_inputs, targets = seq_inputs.to(device), scalar_inputs.to(device), targets.to(device)
            outputs = model(seq_inputs, scalar_inputs)
            loss = criterion(outputs, targets, scalar_inputs, feature_indices, feature_importances)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss detected at epoch {epoch+1}, batch {i+1}")
                return None, None
            loss = loss / accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item() * accumulation_steps * seq_inputs.size(0)
        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq_inputs, scalar_inputs, targets in test_loader:
                seq_inputs, scalar_inputs, targets = seq_inputs.to(device), scalar_inputs.to(device), targets.to(device)
                outputs = model(seq_inputs, scalar_inputs)
                loss = criterion(outputs, targets, scalar_inputs, feature_indices, feature_importances)
                val_loss += loss.item() * seq_inputs.size(0)
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        if val_loss < best_val_loss and not np.isnan(val_loss) and not np.isinf(val_loss):
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
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.close()
    return train_losses, val_losses

def evaluate_model(model, X_test_seq, X_test_scalar, y_test, y_scaler, file_names_test):
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
    print("\nOverall Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    return y_test_actual, y_pred


def main(main_dir):
    print(f"Processing main directory: {main_dir}")
    train_features, test_features, test_file_names = process_tree_output_directory(main_dir)
    if len(train_features) == 0 or len(test_features) == 0:
        print("Error: No valid training or test data found")
        return None
    (train_sequences, train_scalar, y_train,
     test_sequences, test_scalar, y_test,
     y_scaler, seq_input_size, scalar_input_size, feature_columns) = prepare_data_for_model(train_features, test_features)
    train_loader = DataLoader(TensorDataset(train_sequences, train_scalar, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_sequences, test_scalar, y_test), batch_size=64)
    model = EnhancedRecursiveLSTMModel(seq_input_size=seq_input_size, scalar_input_size=scalar_input_size)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    model = train_model(model, train_loader, test_loader, optimizer, num_epochs=100, patience=10)
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
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    main(main_dir)
