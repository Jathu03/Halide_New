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

# Define important metrics for scheduling sequence
important_metrics = [
    'bytes_at_production', 'bytes_at_realization', 'bytes_at_root', 'bytes_at_task',
    'inner_parallelism', 'outer_parallelism', 'num_productions', 'num_realizations',
    'num_scalars', 'num_vectors', 'points_computed_total', 'working_set'
]

# Define memory patterns (dummy values for compatibility with C++ code)
memory_patterns_template = {
    "pointwise": [0.0, 0.0, 0.0, 0.0],
    "transpose": [0.0, 0.0, 0.0, 0.0],
    "broadcast": [0.0, 0.0, 0.0, 0.0],
    "slice": [0.0, 0.0, 0.0, 0.0]
}

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
                if execution_time is not None and execution_time > 0:
                    return float(execution_time)
        
        print(f"Warning: 'total_execution_time_ms' not found in 'Schedules' of {file_path}")
        last_value = schedules[-1]["value"]
        return float(last_value) if last_value > 0 else None
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def extract_features_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    execution_time = get_execution_time(file_path)
    if execution_time is None or not np.isfinite(execution_time):
        print(f"Warning: Invalid execution time in {file_path}")
        return None
    
    nodes_features = []
    edges_features = []
    programming_details = data.get("programming_details", None)
    
    if programming_details:
        if 'Nodes' in programming_details:
            for node in programming_details['Nodes']:
                node_feature = {
                    'name': node.get('Name', ''),
                    'cache_hits': 0.0,  # Dummy values for C++ compatibility
                    'cache_misses': 0.0,
                    'scheduling': {},
                    'op_histogram': {},
                    'memory_patterns': memory_patterns_template.copy()
                }
                if 'Details' in node and 'Op histogram' in node['Details']:
                    op_hist = node['Details']['Op histogram']
                    for op_line in op_hist:
                        parts = op_line.strip().split(':')
                        if len(parts) == 2:
                            op_name = parts[0].strip().lower()
                            op_count = int(parts[1].strip())
                            node_feature['op_histogram'][op_name] = float(op_count)
                nodes_features.append(node_feature)
        
        if 'Edges' in programming_details:
            for edge in programming_details['Edges']:
                edge_feature = {
                    'From': edge.get('From', ''),
                    'To': edge.get('To', ''),
                    'Name': edge.get('Name', '')
                }
                edges_features.append(edge_feature)
    
    scheduling_features = []
    scheduling_data = data.get("scheduling_data", None)
    if not scheduling_data and programming_details and 'Schedules' in programming_details:
        scheduling_data = programming_details['Schedules']
    
    if scheduling_data:
        for sched in scheduling_data:
            sched_feature = {'name': sched.get('Name', '')}
            if 'Details' in sched and 'scheduling_feature' in sched['Details']:
                sf = sched['Details']['scheduling_feature']
                sched_feature.update(sf)
            scheduling_features.append(sched_feature)
    
    # Build tree structure
    tree = {
        "name": "Root",
        "children": [],
        "cache_hits": 0.0,
        "cache_misses": 0.0,
        "scheduling": {},
        "op_histogram": {},
        "memory_patterns": memory_patterns_template.copy()
    }
    for i, node in enumerate(nodes_features):
        node_tree = node.copy()
        node_tree["children"] = []
        tree["children"].append(node_tree)
    
    # Add Global Features node
    global_features = {
        "name": "Global Features",
        "execution_time_ms": execution_time,
        "children": []
    }
    tree["children"].append(global_features)
    
    # Compute significant features for the sequence
    scheduling_sequence = []
    for sf in scheduling_features:
        seq_vector = [float(sf.get(metric, 0.0)) for metric in important_metrics]
        bytes_prod = sf.get('bytes_at_production', 0.0)
        bytes_real = sf.get('bytes_at_realization', 0.0)
        num_vec = sf.get('num_vectors', 0.0)
        points_total = sf.get('points_computed_total', 0.0)
        working_set = sf.get('working_set', 0.0)
        inner_p = sf.get('inner_parallelism', 0.0)
        outer_p = sf.get('outer_parallelism', 0.0)
        comp_efficiency = points_total / max(bytes_prod, 1e-4) if bytes_prod != 0 else 0.0
        mem_util_ratio = working_set / max(bytes_prod, 1e-4) if bytes_prod != 0 else 0.0
        seq_vector.append(np.log1p(abs(bytes_prod)) / np.log1p(max(abs(bytes_real), 1e-4)) if bytes_real != 0 else 0.0)
        seq_vector.append(np.log1p(bytes_prod) / np.log1p(max(num_vec, 1e-4)) if num_vec != 0 else 0.0)
        seq_vector.append(np.log1p(points_total) / np.log1p(max(num_vec, 1e-4)) if num_vec != 0 else 0.0)
        seq_vector.append(np.log1p(working_set) / np.log1p(max(bytes_prod, 1e-4)) if bytes_prod != 0 else 0.0)
        seq_vector.append(np.log1p(inner_p * outer_p))
        seq_vector.append(np.log1p(bytes_prod) / np.log1p(max(points_total, 1e-4)) if points_total != 0 else 0.0)
        seq_vector.append(np.log1p(comp_efficiency))
        seq_vector.append(np.log1p(inner_p) ** 2)
        seq_vector.append(np.log1p(mem_util_ratio))
        scheduling_sequence.append(seq_vector)
    if not scheduling_sequence:
        scheduling_sequence = [[0.0] * (len(important_metrics) + 9)]
    
    seq_array = np.array(scheduling_sequence)
    scaler_seq = RobustScaler()
    scheduling_sequence = scaler_seq.fit_transform(seq_array)
    scheduling_sequence = np.nan_to_num(scheduling_sequence, nan=0.0).tolist()
    
    op_counts = {}
    for node in nodes_features:
        for key, value in node['op_histogram'].items():
            if key.startswith('op_'):
                op_counts[key] = op_counts.get(key, 0) + value
    
    total_ops = sum(op_counts.values())
    num_nodes = max(len(nodes_features), 1)
    num_edges = len(edges_features)
    total_bytes = sum(sf.get('bytes_at_production', 0) for sf in scheduling_features)
    total_vectors = sum(sf.get('num_vectors', 0) for sf in scheduling_features)
    total_parallelism = sum(sf.get('inner_parallelism', 0) * sf.get('outer_parallelism', 1) for sf in scheduling_features)
    comp_efficiency = sum(sf.get('points_computed_total', 0) for sf in scheduling_features) / max(total_bytes, 1e-4) if total_bytes != 0 else 0.0
    bytes_processing_rate = total_bytes / max(execution_time, 1e-4) if execution_time != 0 else 0.0
    mem_util_ratio = sum(sf.get('working_set', 0) for sf in scheduling_features) / max(total_bytes, 1e-4) if total_bytes != 0 else 0.0
    scheduling_count = len(scheduling_features)
    
    scalar_features = {
        'nodes_count': num_nodes,
        'edges_count': num_edges,
        'node_edge_ratio': num_nodes / max(num_edges, 1),
        'total_ops': total_ops,
        'op_diversity': len(op_counts) / num_nodes,
        'avg_ops_per_node': total_ops / num_nodes,
        'edge_density': num_edges / max(num_nodes * (num_nodes - 1), 1),
        'total_parallelism': total_parallelism,
        'avg_bytes_per_node': total_bytes / num_nodes,
        'vector_op_ratio': op_counts.get('op_vector', 0) / max(total_ops, 1),
        'bytes_per_vector': total_bytes / max(total_vectors, 1e-4),
        'ops_per_byte': total_ops / max(total_bytes, 1e-4),
        'computation_efficiency': comp_efficiency,
        'bytes_processing_rate': bytes_processing_rate,
        'memory_utilization_ratio': mem_util_ratio,
        'scheduling_count': scheduling_count,
        'comp_efficiency_total_vectors': comp_efficiency * total_vectors,
        'inner_parallelism_total_parallelism': sum(sf.get('inner_parallelism', 0) for sf in scheduling_features) * total_parallelism,
        'sched_inner_parallelism_squared': sum(sf.get('inner_parallelism', 0) ** 2 for sf in scheduling_features),
        'computation_efficiency_squared': comp_efficiency ** 2
    }
    scalar_features.update(op_counts)
    
    for key in scalar_features:
        if not np.isfinite(scalar_features[key]):
            scalar_features[key] = 0.0
    
    return {
        'tree': tree,
        'scheduling_sequence': scheduling_sequence,
        'scalar_features': scalar_features,
        'execution_time': execution_time
    }

def process_directory(directory_path):
    all_features = []
    file_names = []
    json_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.json')])
    
    for filename in json_files:
        file_path = os.path.join(directory_path, filename)
        features = extract_features_from_file(file_path)
        if features is not None:
            all_features.append(features)
            file_names.append(filename)
    
    return all_features, file_names

def process_main_directory(main_dir):
    all_features = []
    all_file_names = []
    subdirs = sorted([d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))])
    
    if len(subdirs) < 1:
        raise ValueError(f"Expected at least 1 subdirectory in {main_dir}, found {len(subdirs)}")
    
    for subdir in subdirs:
        subdir_path = os.path.join(main_dir, subdir)
        features, file_names = process_directory(subdir_path)
        if not features:
            print(f"Skipping {subdir} due to no valid data")
            continue
        all_features.extend(features)
        all_file_names.extend([os.path.join(subdir, fname) for fname in file_names])
        print(f"Processed subdir {subdir}: {len(features)} files")
    
    total_files = len(all_features)
    if total_files < 50:
        raise ValueError(f"Expected at least 50 files total, found {total_files}")
    
    combined = list(zip(all_features, all_file_names))
    random.shuffle(combined)
    all_features, all_file_names = zip(*combined)
    
    test_size = 50
    train_features = all_features[:-test_size]
    test_features = all_features[-test_size:]
    train_file_names = all_file_names[:-test_size]
    test_file_names = all_file_names[-test_size:]
    
    print(f"Total files: {total_files}")
    print(f"Training files: {len(train_features)}")
    print(f"Testing files: {len(test_features)}")
    
    return train_features, test_features, list(test_file_names)

def prepare_data_for_model(train_features, test_features):
    train_sequences = [torch.FloatTensor(f['scheduling_sequence']) for f in train_features]
    test_sequences = [torch.FloatTensor(f['scheduling_sequence']) for f in test_features]
    
    train_sequences_padded = pad_sequence(train_sequences, batch_first=True)
    test_sequences_padded = pad_sequence(test_sequences, batch_first=True)
    
    train_scalar_df = pd.DataFrame([f['scalar_features'] for f in train_features])
    test_scalar_df = pd.DataFrame([f['scalar_features'] for f in test_features])
    
    low_importance_features = [
        'op_cast', 'op_eq', 'op_ne', 'op_or', 'op_and', 'op_le', 'op_lt', 'op_not',
        'sched_num_scalars', 'sched_bytes_at_realization', 'sched_outer_parallelism',
        'sched_num_realizations', 'sched_num_productions', 'sched_bytes_at_root'
    ]
    train_scalar_df = train_scalar_df.drop(columns=[col for col in low_importance_features if col in train_scalar_df.columns])
    test_scalar_df = test_scalar_df.drop(columns=[col for col in low_importance_features if col in test_scalar_df.columns])
    
    skewed_features = ['computation_efficiency', 'bytes_processing_rate', 'total_parallelism', 'bytes_per_vector']
    for feature in skewed_features:
        if feature in train_scalar_df.columns:
            train_scalar_df[f'log_{feature}'] = np.log1p(train_scalar_df[feature])
            test_scalar_df[f'log_{feature}'] = np.log1p(test_scalar_df[feature])
            train_scalar_df = train_scalar_df.drop(columns=[feature])
            test_scalar_df = test_scalar_df.drop(columns=[feature])
    
    train_scalar_df = train_scalar_df.fillna(0)
    test_scalar_df = test_scalar_df.fillna(0)
    
    constant_columns = [col for col in train_scalar_df.columns if train_scalar_df[col].nunique() == 1]
    train_scalar_df = train_scalar_df.drop(columns=constant_columns)
    test_scalar_df = test_scalar_df.drop(columns=constant_columns)
    
    y_train_raw = np.array([f['execution_time'] for f in train_features])
    y_test_raw = np.array([f['execution_time'] for f in test_features])
    y_train_raw = np.clip(y_train_raw, 0, np.percentile(y_train_raw, 99))
    y_test_raw = np.clip(y_test_raw, 0, np.percentile(y_test_raw, 99))
    
    y_train = np.log1p(y_train_raw).reshape(-1, 1)
    y_test = np.log1p(y_test_raw).reshape(-1, 1)
    
    scaler_X_scalar = RobustScaler()
    scaler_y = RobustScaler()
    
    train_scalar_scaled = scaler_X_scalar.fit_transform(train_scalar_df)
    test_scalar_scaled = scaler_X_scalar.transform(test_scalar_df)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    train_scalar_scaled = np.nan_to_num(train_scalar_scaled, nan=0.0)
    test_scalar_scaled = np.nan_to_num(test_scalar_scaled, nan=0.0)
    y_train_scaled = np.nan_to_num(y_train_scaled, nan=0.0)
    y_test_scaled = np.nan_to_num(y_test_scaled, nan=0.0)
    
    train_sequences_aug = []
    train_scalar_aug = []
    y_train_aug = []
    for i in range(len(train_features)):
        train_sequences_aug.append(train_sequences_padded[i])
        train_scalar_aug.append(train_scalar_scaled[i])
        y_train_aug.append(y_train_scaled[i])
        
        inner_parallelism_idx = train_scalar_df.columns.get_loc('inner_parallelism_total_parallelism') if 'inner_parallelism_total_parallelism' in train_scalar_df.columns else -1
        comp_efficiency_idx = train_scalar_df.columns.get_loc('log_computation_efficiency') if 'log_computation_efficiency' in train_scalar_df.columns else -1
        
        is_significant = False
        if inner_parallelism_idx != -1 and train_scalar_scaled[i, inner_parallelism_idx] > np.percentile(train_scalar_scaled[:, inner_parallelism_idx], 75):
            is_significant = True
        if comp_efficiency_idx != -1 and train_scalar_scaled[i, comp_efficiency_idx] > np.percentile(train_scalar_scaled[:, comp_efficiency_idx], 75):
            is_significant = True
        
        augment_count = 3 if is_significant else 1
        for _ in range(augment_count):
            noise_seq = torch.normal(mean=0.0, std=0.05, size=train_sequences_padded[i].shape)
            noise_scalar = np.random.normal(0, 0.05, train_scalar_scaled[i].shape)
            noise_y = np.random.normal(0, 0.05, y_train_scaled[i].shape)
            train_sequences_aug.append(train_sequences_padded[i] + noise_seq)
            train_scalar_aug.append(train_scalar_scaled[i] + noise_scalar)
            y_train_aug.append(y_train_scaled[i] + noise_y)
    
    train_sequences_padded = pad_sequence(train_sequences_aug, batch_first=True)
    
    train_scalar_scaled = np.array(train_scalar_aug)
    y_train_scaled = np.array(y_train_aug)
    
    train_scalar_tensor = torch.FloatTensor(train_scalar_scaled)
    test_scalar_tensor = torch.FloatTensor(test_scalar_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Sequence input size: {train_sequences_padded.shape[2]}")
    print(f"Scalar input size: {train_scalar_tensor.shape[1]}")
    
    # Save metadata and scalers
    metadata = {
        "max_sequence_length": train_sequences_padded.shape[1],
        "seq_input_size": train_sequences_padded.shape[2],
        "scalar_input_size": train_scalar_tensor.shape[1],
        "node_features": [
            "cache_hits", "cache_misses",
            *[f"sched_{metric}" for metric in important_metrics],
            *[f"op_{op}" for op in ["add", "sub", "mul", "div", "mod", "eq", "ne", "lt", "le", "or", "and", "not",
                                    "min", "max", "constant", "variable", "funccall", "imagecall", "externcall", "let", "param"]],
            *[f"memory_{pattern}_{i}" for pattern in ["pointwise", "transpose", "broadcast", "slice"] for i in range(4)]
        ],
        "scalar_features": list(train_scalar_df.columns),
        "skewed_features": [f"log_{f}" for f in skewed_features],
        "dropped_features": low_importance_features
    }
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f)
    
    scaler_node_params = {"center": scaler_seq.center_.tolist(), "scale": scaler_seq.scale_.tolist()}
    with open("scaler_node_params.json", "w") as f:
        json.dump(scaler_node_params, f)
    
    scaler_scalar_params = {"center": scaler_X_scalar.center_.tolist(), "scale": scaler_X_scalar.scale_.tolist()}
    with open("scaler_scalar_params.json", "w") as f:
        json.dump(scaler_scalar_params, f)
    
    scaler_y_params = {"center": scaler_y.center_.tolist(), "scale": scaler_y.scale_.tolist()}
    with open("scaler_y_params.json", "w") as f:
        json.dump(scaler_y_params, f)
    
    return (train_sequences_padded, train_scalar_tensor, y_train_tensor,
            test_sequences_padded, test_scalar_tensor, y_test_tensor,
            scaler_y, train_sequences_padded.shape[2], train_scalar_tensor.shape[1], train_scalar_df.columns)

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

def train_model(model, train_loader, test_loader, criterion, optimizer, feature_indices, feature_importances, num_epochs=700, patience=50, accumulation_steps=2):
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
    
    results_by_subfolder = {}
    for i, file_path in enumerate(file_names_test):
        subfolder = file_path.split('/')[0]
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

def main(main_dir):
    if torch.cuda.is_available():
        torch.cuda.init()
        print(f"CUDA initialized. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Using CPU.")
    
    print(f"Processing main directory: {main_dir}")
    train_features, test_features, test_file_names = process_main_directory(main_dir)
    
    print(f"Total training samples: {len(train_features)} (randomly selected)")
    print(f"Total test samples: {len(test_features)} (50 randomly selected)")
    
    if len(train_features) == 0 or len(test_features) == 0:
        print("Error: No valid training or test data found")
        return None
    
    (train_sequences, train_scalar, y_train,
     test_sequences, test_scalar, y_test,
     y_scaler, seq_input_size, scalar_input_size, feature_columns) = prepare_data_for_model(train_features, test_features)
    
    train_loader, test_loader = create_data_loaders(
        train_sequences, train_scalar, y_train,
        test_sequences, test_scalar, y_test,
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
        'computation_efficiency': 0.6064,
        'inner_parallelism_total_parallelism': 0.2135,
        'total_parallelism': 0.0038,
        'bytes_per_vector': 0.0138,
        'scheduling_count': 0.0454,
        'avg_bytes_per_node': 0.0357,
        'bytes_processing_rate': 0.0064
    }
    
    feature_indices = {}
    for feature in feature_importances.keys():
        log_feature = f'log_{feature}' if feature in ['computation_efficiency', 'bytes_processing_rate', 'total_parallelism', 'bytes_per_vector'] else feature
        if log_feature in feature_columns:
            feature_indices[feature] = feature_columns.get_loc(log_feature)
        else:
            feature_indices[feature] = feature_columns.get_loc(feature) if feature in feature_columns else -1
    
    print("Building and training Enhanced Recursive LSTM model...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader,
        custom_loss, optimizer, feature_indices, feature_importances,
        num_epochs=700, patience=50, accumulation_steps=2
    )
    
    if train_losses is None or val_losses is None:
        print("Training failed due to invalid values")
        return None
    
    # Export model for LibTorch
    model.eval()
    example_seq = torch.zeros(1, train_sequences.shape[1], seq_input_size)
    example_scalar = torch.zeros(1, scalar_input_size)
    traced_model = torch.jit.trace(model, (example_seq, example_scalar))
    traced_model.save("recursive_model.pt")
    
    # Save a test JSON file
    if test_features:
        with open("tree_representation.json", "w") as f:
            json.dump(test_features[0]['tree'], f)
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(
        model, test_sequences, test_scalar, y_test,
        y_scaler, test_file_names
    )
    
    print(f"\nSummary for Comparison:")
    print(f"Model: EnhancedRecursiveLSTM")
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "synthetic_data"
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    model, y_scaler, y_test_actual, y_pred_actual = main(main_dir)
