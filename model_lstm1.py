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
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
from torch.jit import script

# Define fixed set of features for each node in the sequence
NODE_FEATURES = [
    'cache_hits', 'cache_misses', 'sched_num_realizations', 'sched_num_productions',
    'sched_points_computed_total', 'sched_innermost_loop_extent', 'sched_inner_parallelism',
    'sched_outer_parallelism', 'sched_bytes_at_realization', 'sched_bytes_at_production',
    'sched_bytes_at_root', 'sched_unique_bytes_read_per_realization', 'sched_working_set',
    'sched_vector_size', 'sched_num_vectors', 'sched_num_scalars', 'sched_bytes_at_task',
    'sched_working_set_at_task', 'sched_working_set_at_production', 'sched_working_set_at_realization',
    'sched_working_set_at_root', 'op_add', 'op_sub', 'op_mul', 'op_div', 'op_mod', 'op_eq',
    'op_ne', 'op_lt', 'op_le', 'op_or', 'op_and', 'op_not', 'op_min', 'op_max', 'op_constant',
    'op_variable', 'op_funccall', 'op_imagecall', 'op_externcall', 'op_let', 'op_param',
    'memory_transpose_0', 'memory_transpose_1', 'memory_transpose_2', 'memory_transpose_3',
    'memory_slice_0', 'memory_slice_1', 'memory_slice_2', 'memory_slice_3',
    'memory_broadcast_0', 'memory_broadcast_1', 'memory_broadcast_2', 'memory_broadcast_3',
    'memory_pointwise_0', 'memory_pointwise_1', 'memory_pointwise_2', 'memory_pointwise_3'
]

# Scalar features for the entire sample
SCALAR_FEATURES = [
    'execution_time_ms', 'total_parallelism', 'scheduling_count', 'total_bytes_at_production',
    'total_vectors', 'computation_efficiency', 'memory_pressure', 'memory_utilization_ratio',
    'bytes_processing_rate', 'bytes_per_parallelism', 'bytes_per_vector', 'nodes_count',
    'edges_count', 'node_edge_ratio', 'nodes_per_schedule', 'op_diversity'
]

# Feature extraction function for node-level and scalar features
def extract_features(json_data):
    node_sequences = []
    scalar_features = {}
    
    # Helper function to extract node features
    def extract_node_features(node, is_global=False):
        features = {}
        if is_global:
            features['cache_hits'] = json_data.get('Global Features', {}).get('cache_hits', 0)
            features['cache_misses'] = json_data.get('Global Features', {}).get('cache_misses', 0)
        else:
            features['cache_hits'] = node.get('cache_hits', 0)
            features['cache_misses'] = node.get('cache_misses', 0)
        
        # Extract scheduling features
        scheduling = node.get('scheduling', {}) if not is_global else {}
        for key in [
            'num_realizations', 'num_productions', 'points_computed_total', 'innermost_loop_extent',
            'inner_parallelism', 'outer_parallelism', 'bytes_at_realization', 'bytes_at_production',
            'bytes_at_root', 'unique_bytes_read_per_realization', 'working_set', 'vector_size',
            'num_vectors', 'num_scalars', 'bytes_at_task', 'working_set_at_task',
            'working_set_at_production', 'working_set_at_realization', 'working_set_at_root'
        ]:
            features[f'sched_{key}'] = scheduling.get(key, 0)
        
        # Extract op_histogram
        op_histogram = node.get('op_histogram', {}) if not is_global else {}
        for op in ['add', 'sub', 'mul', 'div', 'mod', 'eq', 'ne', 'lt', 'le', 'or', 'and', 'not', 'min', 'max',
                   'constant', 'variable', 'funccall', 'imagecall', 'externcall', 'let', 'param']:
            features[f'op_{op}'] = op_histogram.get(op, 0)
        
        # Extract memory patterns
        memory_patterns = node.get('memory_patterns', {}) if not is_global else {}
        for pattern in ['transpose', 'slice', 'broadcast', 'pointwise']:
            for i in range(4):
                features[f'memory_{pattern}_{i}'] = memory_patterns.get(f'{pattern}_{i}', 0)
        
        return {key: features.get(key, 0.0) for key in NODE_FEATURES}
    
    # Recursive function to traverse tree
    def traverse_tree(node):
        if not node:
            return
        node_features = extract_node_features(node)
        node_sequences.append(node_features)
        for child in node.get('children', []):
            traverse_tree(child)
    
    # Start with global features
    global_node = {}
    node_sequences.append(extract_node_features(global_node, is_global=True))
    
    # Traverse the tree starting from root
    traverse_tree(json_data.get('root', {}))
    
    # Extract scalar features
    scalar_data = json_data.get('Scalar Features', {})
    for feature in SCALAR_FEATURES:
        scalar_features[feature] = scalar_data.get(feature, 0)
    
    # Ensure execution_time_ms is set
    scalar_features['execution_time_ms'] = json_data.get('Global Features', {}).get('execution_time_ms', 0)
    
    return node_sequences, scalar_features

def safe_div(a, b):
    return a / b if b != 0 else 0

# Process directory containing tree_representation.json files
def process_tree_output_directory(main_dir):
    all_node_sequences = []
    all_scalar_features = []
    file_names = []
    skipped_files = []
    zero_time_files = []
    
    for root, dirs, files in os.walk(main_dir):
        if 'tree_representation.json' in files:
            file_path = os.path.join(root, 'tree_representation.json')
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                node_sequences, scalar_features = extract_features(json_data)
                if np.isfinite(scalar_features['execution_time_ms']):
                    if scalar_features['execution_time_ms'] == 0:
                        zero_time_files.append(file_path)
                        print(f"Warning: {file_path} has execution_time_ms = 0, but including it")
                    all_node_sequences.append(node_sequences)
                    all_scalar_features.append(scalar_features)
                    file_names.append(file_path)
                else:
                    skipped_files.append(file_path)
                    print(f"Skipped {file_path} due to non-finite execution time: {scalar_features['execution_time_ms']}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                skipped_files.append(file_path)
    
    if not all_node_sequences:
        raise ValueError("No valid JSON files with finite execution times found in directory.")
    
    log_path = os.path.join(main_dir, 'skipped_files_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("Files skipped due to non-finite execution times or errors:\n")
        for file_path in skipped_files:
            f.write(f"{file_path}\n")
        f.write("\nFiles with execution_time_ms = 0 (included but potentially problematic):\n")
        for file_path in zero_time_files:
            f.write(f"{file_path}\n")
    
    total_files = len(all_node_sequences) + len(skipped_files)
    print(f"Total files found: {total_files}")
    print(f"Files skipped: {len(skipped_files)}")
    print(f"Files with zero execution time: {len(zero_time_files)}")
    print(f"Valid files retained: {len(all_node_sequences)}")
    if len(all_node_sequences) < 10:
        raise ValueError(f"Expected at least 10 valid files, found {len(all_node_sequences)}")
    
    combined = list(zip(all_node_sequences, all_scalar_features, file_names))
    random.shuffle(combined)
    all_node_sequences, all_scalar_features, file_names = zip(*combined)
    
    test_size = min(50, len(all_node_sequences))
    train_node_sequences = all_node_sequences[:-test_size]
    test_node_sequences = all_node_sequences[-test_size:]
    train_scalar_features = all_scalar_features[:-test_size]
    test_scalar_features = all_scalar_features[-test_size:]
    train_file_names = file_names[:-test_size]
    test_file_names = file_names[-test_size:]
    
    print(f"Training files: {len(train_node_sequences)}")
    print(f"Testing files: {len(test_node_sequences)}")
    
    return train_node_sequences, test_node_sequences, train_scalar_features, test_scalar_features, list(test_file_names)

# Prepare data for model
def prepare_data_for_model(train_node_sequences, test_node_sequences, train_scalar_features, test_scalar_features):
    # Convert node sequences to tensors
    train_sequences = []
    test_sequences = []
    
    # Scaler for node features
    scaler_node = RobustScaler()
    
    # Flatten node features for fitting the scaler
    all_node_features = []
    for seq in train_node_sequences:
        for node in seq:
            all_node_features.append([node[key] for key in NODE_FEATURES])
    all_node_features = np.array(all_node_features)
    scaler_node.fit(all_node_features)
    
    # Process training sequences
    for seq in train_node_sequences:
        node_features = np.array([[node[key] for key in NODE_FEATURES] for node in seq])
        node_features_scaled = scaler_node.transform(node_features)
        train_sequences.append(torch.FloatTensor(node_features_scaled))
    
    # Process test sequences
    for seq in test_node_sequences:
        node_features = np.array([[node[key] for key in NODE_FEATURES] for node in seq])
        node_features_scaled = scaler_node.transform(node_features)
        test_sequences.append(torch.FloatTensor(node_features_scaled))
    
    # Pad sequences
    train_sequences_padded = pad_sequence(train_sequences, batch_first=True, padding_value=0.0)
    test_sequences_padded = pad_sequence(test_sequences, batch_first=True, padding_value=0.0)
    
    # Process scalar features
    train_scalar_df = pd.DataFrame(train_scalar_features)
    test_scalar_df = pd.DataFrame(test_scalar_features)
    
    # Drop low-importance scalar features
    low_importance_features = [
        'op_cast', 'op_selfcall', 'memory_pointwise_1', 'memory_transpose_1', 'memory_broadcast_1',
        'memory_slice_1', 'op_select', 'op_not', 'op_and', 'op_ne', 'op_mod', 'memory_pointwise_2',
        'memory_broadcast_2', 'memory_slice_2', 'memory_transpose_2', 'op_externcall', 'op_imagecall',
        'op_param', 'memory_pointwise_3', 'memory_transpose_3', 'op_sub', 'memory_pointwise_0', 'op_let'
    ]
    train_scalar_df = train_scalar_df.drop(columns=[col for col in low_importance_features if col in train_scalar_df.columns])
    test_scalar_df = test_scalar_df.drop(columns=[col for col in low_importance_features if col in test_scalar_df.columns])
    
    # Log transform skewed scalar features
    skewed_features = ['bytes_processing_rate', 'total_bytes_at_production', 'computation_efficiency']
    for feature in skewed_features:
        if feature in train_scalar_df.columns:
            train_scalar_df[f'log_{feature}'] = np.log1p(train_scalar_df[feature])
            test_scalar_df[f'log_{feature}'] = np.log1p(test_scalar_df[feature])
            train_scalar_df = train_scalar_df.drop(columns=[feature])
            test_scalar_df = test_scalar_df.drop(columns=[feature])
    
    train_scalar_df = train_scalar_df.fillna(0)
    test_scalar_df = test_scalar_df.fillna(0)
    
    # Remove constant columns
    constant_columns = [col for col in train_scalar_df.columns if train_scalar_df[col].nunique() == 1]
    train_scalar_df = train_scalar_df.drop(columns=constant_columns)
    test_scalar_df = test_scalar_df.drop(columns=constant_columns)
    
    # Scale scalar features
    scaler_scalar = RobustScaler()
    train_scalar_scaled = scaler_scalar.fit_transform(train_scalar_df)
    test_scalar_scaled = scaler_scalar.transform(test_scalar_df)
    train_scalar_scaled = np.nan_to_num(train_scalar_scaled, nan=0.0)
    test_scalar_scaled = np.nan_to_num(test_scalar_scaled, nan=0.0)
    
    # Extract and scale execution times
    y_train_raw = np.array([f['execution_time_ms'] for f in train_scalar_features])
    y_test_raw = np.array([f['execution_time_ms'] for f in test_scalar_features])
    y_train_raw = np.clip(y_train_raw, 0, np.percentile(y_train_raw, 99))
    y_test_raw = np.clip(y_test_raw, 0, np.percentile(y_test_raw, 99))
    
    y_train = np.log1p(y_train_raw).reshape(-1, 1)
    y_test = np.log1p(y_test_raw).reshape(-1, 1)
    
    y_scaler = RobustScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)
    y_train_scaled = np.nan_to_num(y_train_scaled, nan=0.0)
    y_test_scaled = np.nan_to_num(y_test_scaled, nan=0.0)
    
    # Data augmentation
    train_sequences_aug = []
    train_scalar_aug = []
    y_train_aug = []
    for i in range(len(train_node_sequences)):
        train_sequences_aug.append(train_sequences_padded[i])
        train_scalar_aug.append(train_scalar_scaled[i])
        y_train_aug.append(y_train_scaled[i])
        
        bytes_rate_idx = train_scalar_df.columns.get_loc('log_bytes_processing_rate') if 'log_bytes_processing_rate' in train_scalar_df.columns else -1
        
        is_significant = False
        if bytes_rate_idx != -1 and train_scalar_scaled[i, bytes_rate_idx] > np.percentile(train_scalar_scaled[:, bytes_rate_idx], 75):
            is_significant = True
        
        augment_count = 3 if is_significant else 1
        for _ in range(augment_count):
            noise_seq = torch.normal(mean=0.0, std=0.05, size=train_sequences_padded[i].shape)
            noise_scalar = np.random.normal(0, 0.05, train_scalar_scaled[i].shape)
            noise_y = np.random.normal(0, 0.05, y_train_scaled[i].shape)
            
            seq_aug = train_sequences_padded[i] + noise_seq
            scalar_aug = train_scalar_scaled[i] + noise_scalar
            if bytes_rate_idx != -1:
                scalar_aug[bytes_rate_idx] += np.random.normal(0, 0.05)
            
            train_sequences_aug.append(seq_aug)
            train_scalar_aug.append(scalar_aug)
            y_train_aug.append(y_train_scaled[i] + noise_y)
    
    train_sequences_padded = torch.stack(train_sequences_aug)
    train_scalar_scaled = np.array(train_scalar_aug)
    y_train_aug = np.array(y_train_aug)
    
    train_scalar_tensor = torch.FloatTensor(train_scalar_scaled)
    test_scalar_tensor = torch.FloatTensor(test_scalar_scaled)
    y_train_tensor = torch.FloatTensor(y_train_aug)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    # Save scalers
    with open('scaler_node.pkl', 'wb') as f:
        pickle.dump(scaler_node, f)
    with open('scaler_scalar.pkl', 'wb') as f:
        pickle.dump(scaler_scalar, f)
    with open('scaler_y.pkl', 'wb') as f:
        pickle.dump(y_scaler, f)
    
    # Save scaler parameters as JSON for C++ compatibility
    scaler_node_params = {
        "center": scaler_node.center_.tolist(),
        "scale": scaler_node.scale_.tolist()
    }
    scaler_scalar_params = {
        "center": scaler_scalar.center_.tolist(),
        "scale": scaler_scalar.scale_.tolist()
    }
    scaler_y_params = {
        "center": y_scaler.center_.tolist(),
        "scale": y_scaler.scale_.tolist()
    }
    with open('scaler_node_params.json', 'w') as f:
        json.dump(scaler_node_params, f)
    with open('scaler_scalar_params.json', 'w') as f:
        json.dump(scaler_scalar_params, f)
    with open('scaler_y_params.json', 'w') as f:
        json.dump(scaler_y_params, f)
    
    # Save metadata
    metadata = {
        "max_sequence_length": int(train_sequences_padded.shape[1]),
        "seq_input_size": int(train_sequences_padded.shape[2]),
        "scalar_input_size": int(train_scalar_tensor.shape[1]),
        "node_features": NODE_FEATURES,
        "scalar_features": train_scalar_df.columns.tolist(),
        "dropped_features": low_importance_features + constant_columns,
        "skewed_features": skewed_features
    }
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f)
    
    print(f"Sequence input size: {train_sequences_padded.shape[2]}")
    print(f"Max sequence length: {train_sequences_padded.shape[1]}")
    print(f"Scalar input size: {train_scalar_tensor.shape[1]}")
    
    return (train_sequences_padded, train_scalar_tensor, y_train_tensor,
            test_sequences_padded, test_scalar_tensor, y_test_tensor,
            y_scaler, train_sequences_padded.shape[2], train_scalar_tensor.shape[1], train_scalar_df.columns)

# Model definition
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

# Custom loss function
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
                feature_vals > torch.quantile(feature_vals, 0.75),
                weights * (1.0 + importance * 2.0),
                weights
            )
    
    weighted_huber = (huber * weights).mean()
    weighted_mae = (mae * weights).mean()
    return weighted_huber + mae_weight * weighted_mae + l1_reg

# Create data loaders
def create_data_loaders(train_sequences, train_scalar, y_train, test_sequences, test_scalar, y_test, batch_size=64):
    train_dataset = TensorDataset(train_sequences, train_scalar, y_train)
    test_dataset = TensorDataset(test_sequences, test_scalar, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# Train model with checkpoint handling
def train_model(model, train_loader, test_loader, criterion, optimizer, feature_indices, feature_importances, num_epochs=1000, patience=50, accumulation_steps=2, checkpoint_path='recursive.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        model.to(device)
        for lstm in model.lstm_layers:
            lstm.flatten_parameters()
    except RuntimeError as e:
        print(f"Error moving model to CUDA: {e}. Falling back to CPU.")
        device = torch.device('cpu')
        model.to(device)
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    start_epoch = 0
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            epochs_no_improve = checkpoint['epochs_no_improve']
            best_model_state = checkpoint['best_model_state']
            print(f"Resuming training from epoch {start_epoch}")
        except RuntimeError as e:
            print(f"Checkpoint incompatible: {e}. Starting training from scratch.")
            os.rename(checkpoint_path, checkpoint_path + '.backup')
            print(f"Renamed incompatible checkpoint to {checkpoint_path}.backup")
    
    for epoch in range(start_epoch, num_epochs):
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
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epochs_no_improve': epochs_no_improve,
            'best_model_state': best_model_state
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")
        
        if val_loss < best_val_loss and not np.isnan(val_loss) and not np.isinf(val_loss):
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            
            # Save model for LibTorch
            model.eval()
            example_seq = torch.randn(1, train_sequences_padded.shape[1], seq_input_size).to(device)
            example_scalar = torch.randn(1, scalar_input_size).to(device)
            traced_model = torch.jit.trace(model, (example_seq, example_scalar))
            traced_model.save('recursive_model.pt')
            print("Model saved for LibTorch as recursive_model.pt")
            
            # Save example input shapes for reference
            input_shapes = {
                "seq_input_shape": [1, int(train_sequences_padded.shape[1]), int(seq_input_size)],
                "scalar_input_shape": [1, int(scalar_input_size)]
            }
            with open('input_shapes.json', 'w') as f:
                json.dump(input_shapes, f)
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

# Resume training
def resume_training(model, train_loader, test_loader, criterion, optimizer, feature_indices, feature_importances, num_epochs=1000, patience=50, accumulation_steps=2, checkpoint_path='recursive.pth'):
    print(f"Attempting to resume training from checkpoint: {checkpoint_path}")
    return train_model(model, train_loader, test_loader, criterion, optimizer, feature_indices, feature_importances, num_epochs, patience, accumulation_steps, checkpoint_path)

# Evaluate model
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
    
    results_by_subfolder = {}
    for i, file_path in enumerate(file_names_test):
        subfolder = '/'.join(file_path.split('/')[:-1])
        if subfolder not in results_by_subfolder:
            results_by_subfolder[subfolder] = []
        
        pred = max(y_pred_actual[i][0], 0)
        results_by_subfolder[subfolder].append({
            'file': file_path,
            'actual': y_test_actual[i][0],
            'predicted': pred,
            'error_percentage': abs(y_test_actual[i][0] - pred) / y_test_actual[i][0] * 100 if y_test_actual[i][0] > 0 else 0
        })
    
    for subfolder, results in results_by_subfolder.items():
        print(f"\nResults for {subfolder}:")
        for result in results:
            print(f"File: {result['file']}")
            print(f"  Actual execution time: {result['actual']:.2f} ms")
            print(f"  Predicted execution time: {result['predicted']:.2f} ms")
            print(f"  Error percentage: {result['error_percentage']:.2f}%")
    
    mse = np.mean((y_test_actual - y_pred_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / (y_test_actual + 1e-8))) * 100
    
    print("\nOverall Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    return y_test_actual, y_pred_actual

# Main function
def main(main_dir):
    if torch.cuda.is_available():
        torch.cuda.init()
        print(f"CUDA initialized. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Using CPU.")
    
    print(f"Processing main directory: {main_dir}")
    train_node_sequences, test_node_sequences, train_scalar_features, test_scalar_features, test_file_names = process_tree_output_directory(main_dir)
    
    if len(train_node_sequences) == 0 or len(test_node_sequences) == 0:
        print("Error: No valid training or test data found")
        return None
    
    global seq_input_size, scalar_input_size, train_sequences_padded
    (train_sequences_padded, train_scalar, y_train,
     test_sequences_padded, test_scalar, y_test,
     y_scaler, seq_input_size, scalar_input_size, feature_columns) = prepare_data_for_model(
         train_node_sequences, test_node_sequences, train_scalar_features, test_scalar_features)
    
    train_loader, test_loader = create_data_loaders(
        train_sequences_padded, train_scalar, y_train,
        test_sequences_padded, test_scalar, y_test,
        batch_size=64
    )
    
    global model
    model = EnhancedRecursiveLSTMModel(
        seq_input_size=seq_input_size,
        scalar_input_size=scalar_input_size,
        hidden_sizes=[512, 256, 128],
        output_size=1,
        dropout_rate=0.2,
        num_heads=8
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-4)
    
    feature_importances = {
        'bytes_processing_rate': 0.2893,
        'total_bytes_at_production': 0.0422,
        'nodes_count': 0.0248,
        'computation_efficiency': 0.0055,
        'memory_utilization_ratio': 0.0049
    }
    
    feature_indices = {}
    for feature in feature_importances.keys():
        log_feature = f'log_{feature}' if feature in ['bytes_processing_rate', 'total_bytes_at_production', 'computation_efficiency'] else feature
        if log_feature in feature_columns:
            feature_indices[feature] = feature_columns.get_loc(log_feature)
        else:
            feature_indices[feature] = feature_columns.get_loc(feature) if feature in feature_columns else -1
    
    print("Building and training Enhanced Recursive LSTM model...")
    train_losses, val_losses = resume_training(
        model, train_loader, test_loader,
        custom_loss, optimizer, feature_indices, feature_importances,
        num_epochs=1000, patience=50, accumulation_steps=2, checkpoint_path='recursive.pth'
    )
    
    if train_losses is None or val_losses is None:
        print("Training failed due to invalid values")
        return None
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(
        model, test_sequences_padded, test_scalar, y_test,
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
    model, y_scaler, y_test_actual, y_pred_actual = main(main_dir)
