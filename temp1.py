import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import psutil
import time

# Define fixed sets of features
PROGRAM_FEATURES = [
    'op_add', 'op_sub', 'op_mul', 'op_div', 'op_mod', 'op_eq', 'op_ne', 'op_lt', 'op_le',
    'op_or', 'op_and', 'op_not', 'op_min', 'op_max', 'op_constant', 'op_variable',
    'op_funccall', 'op_imagecall', 'op_externcall', 'op_let', 'op_param',
    'memory_transpose_0', 'memory_transpose_1', 'memory_transpose_2', 'memory_transpose_3',
    'memory_slice_0', 'memory_slice_1', 'memory_slice_2', 'memory_slice_3',
    'memory_broadcast_0', 'memory_broadcast_1', 'memory_broadcast_2', 'memory_broadcast_3',
    'memory_pointwise_0', 'memory_pointwise_1', 'memory_pointwise_2', 'memory_pointwise_3',
    'op_diversity', 'nodes_count', 'edges_count', 'node_edge_ratio'
]

SCHEDULE_FEATURES = [
    'cache_hits', 'cache_misses', 'execution_time_ms', 'sched_num_realizations',
    'sched_num_productions', 'sched_points_computed_total', 'sched_innermost_loop_extent',
    'sched_inner_parallelism', 'sched_outer_parallelism', 'sched_bytes_at_realization',
    'sched_bytes_at_production', 'sched_bytes_at_root', 'sched_unique_bytes_read_per_realization',
    'sched_working_set', 'sched_vector_size', 'sched_num_vectors', 'sched_num_scalars',
    'sched_bytes_at_task', 'sched_working_set_at_task', 'sched_working_set_at_production',
    'sched_working_set_at_realization', 'sched_working_set_at_root', 'total_parallelism',
    'scheduling_count', 'total_bytes_at_production', 'total_vectors', 'computation_efficiency',
    'memory_pressure', 'memory_utilization_ratio', 'bytes_processing_rate', 'bytes_per_parallelism',
    'bytes_per_vector', 'nodes_per_schedule'
]

# Device Manager Class to handle CPU/GPU workload distribution
class DeviceManager:
    def __init__(self, usage_threshold=80.0):
        self.devices = []
        self.usage_threshold = usage_threshold  # % usage above which device is considered busy
        self.device_loads = {}
        
        # Check CPU availability
        self.devices.append(torch.device('cpu'))
        self.device_loads['cpu'] = {'usage': 0.0, 'memory': 0.0}
        
        # Check GPU availability
        if torch.cuda.is_available():
            torch.cuda.init()
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                device = torch.device(f'cuda:{i}')
                self.devices.append(device)
                self.device_loads[f'cuda:{i}'] = {'usage': 0.0, 'memory': 0.0}
                print(f"Found GPU: {torch.cuda.get_device_name(i)}")
        else:
            print("No GPUs available. Using CPU only.")
        
        print(f"Available devices: {[str(d) for d in self.devices]}")

    def update_device_loads(self):
        # Update CPU load
        cpu_usage = psutil.cpu_percent(interval=0.1)
        cpu_memory = psutil.virtual_memory().percent
        self.device_loads['cpu']['usage'] = cpu_usage
        self.device_loads['cpu']['memory'] = cpu_memory
        
        # Update GPU loads
        for device in self.devices:
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
                total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
                memory_usage = (memory_allocated / total_memory) * 100 if total_memory > 0 else 0
                # Approximate GPU usage (requires NVIDIA tools or approximation)
                # Here, we use memory usage as a proxy for load
                self.device_loads[str(device)]['usage'] = memory_usage
                self.device_loads[str(device)]['memory'] = memory_usage

    def get_least_loaded_device(self):
        self.update_device_loads()
        least_load = float('inf')
        best_device = self.devices[0]  # Default to CPU
        
        for device in self.devices:
            device_str = str(device)
            load = max(self.device_loads[device_str]['usage'], self.device_loads[device_str]['memory'])
            if load < least_load and load < self.usage_threshold:
                least_load = load
                best_device = device
        
        print(f"Selected device: {best_device} with load {least_load:.2f}%")
        return best_device

    def distribute_batch(self, batch, batch_size):
        # Split batch across devices based on load
        self.update_device_loads()
        available_devices = [d for d in self.devices if max(self.device_loads[str(d)]['usage'], 
                                                            self.device_loads[str(d)]['memory']) < self.usage_threshold]
        if not available_devices:
            print("All devices are busy. Using least loaded device.")
            available_devices = [self.get_least_loaded_device()]
        
        num_devices = len(available_devices)
        if num_devices == 1:
            return [(available_devices[0], batch)]
        
        # Split batch proportionally based on device load
        batch_splits = []
        total_load = sum(max(self.device_loads[str(d)]['usage'], self.device_loads[str(d)]['memory']) for d in available_devices)
        if total_load == 0:
            total_load = 1e-8  # Avoid division by zero
        
        remaining_size = batch_size
        for i, device in enumerate(available_devices):
            device_load = max(self.device_loads[str(device)]['usage'], self.device_loads[str(device)]['memory'])
            # Inverse load: less loaded devices get more data
            inverse_load = (total_load - device_load) if total_load > device_load else 1e-8
            proportion = inverse_load / sum((total_load - max(self.device_loads[str(d)]['usage'], 
                                                              self.device_loads[str(d)]['memory'])) 
                                            for d in available_devices)
            split_size = int(proportion * batch_size) if i < num_devices - 1 else remaining_size
            batch_splits.append((device, split_size))
            remaining_size -= split_size
        
        return batch_splits

# Feature extraction function
def extract_features(json_data):
    program_features = {}
    schedule_features = {}
    
    global_node = next((child for child in json_data['children'] if child['name'] == 'Global Features'), None)
    if global_node:
        schedule_features['cache_hits'] = global_node.get('cache_hits', 0)
        schedule_features['cache_misses'] = global_node.get('cache_misses', 0)
        schedule_features['execution_time_ms'] = global_node.get('execution_time_ms', 0)
    
    op_histogram = defaultdict(int)
    for node in json_data['children']:
        if 'op_histogram' in node:
            for op, count in node['op_histogram'].items():
                op_histogram[op.lower()] += count
    for op, count in op_histogram.items():
        program_features[f'op_{op.lower()}'] = count
    
    memory_patterns = defaultdict(lambda: [0, 0, 0, 0])
    for node in json_data['children']:
        if 'memory_patterns' in node:
            for pattern, values in node['memory_patterns'].items():
                memory_patterns[pattern] = [sum(x) for x in zip(memory_patterns[pattern], values)]
    for pattern, values in memory_patterns.items():
        for i, val in enumerate(values):
            program_features[f'memory_{pattern.lower()}_{i}'] = val
    
    scheduling_keys = [
        'num_realizations', 'num_productions', 'points_computed_total', 'innermost_loop_extent',
        'inner_parallelism', 'outer_parallelism', 'bytes_at_realization', 'bytes_at_production',
        'bytes_at_root', 'unique_bytes_read_per_realization', 'working_set', 'vector_size',
        'num_vectors', 'num_scalars', 'bytes_at_task', 'working_set_at_task', 'working_set_at_production',
        'working_set_at_realization', 'working_set_at_root'
    ]
    scheduling_sums = defaultdict(float)
    node_count = 0
    for node in json_data['children']:
        if 'scheduling' in node:
            node_count += 1
            for key in scheduling_keys:
                scheduling_sums[key] += node['scheduling'].get(key, 0)
    for key in scheduling_keys:
        if key in ['inner_parallelism', 'outer_parallelism'] and node_count > 0:
            schedule_features[f'sched_{key}'] = scheduling_sums[key] / node_count
        else:
            schedule_features[f'sched_{key}'] = scheduling_sums[key]
    
    schedule_features['total_parallelism'] = schedule_features.get('sched_inner_parallelism', 0) + schedule_features.get('sched_outer_parallelism', 0)
    schedule_features['scheduling_count'] = schedule_features.get('sched_num_realizations', 0) + schedule_features.get('sched_num_productions', 0)
    schedule_features['total_bytes_at_production'] = schedule_features.get('sched_bytes_at_production', 0)
    schedule_features['total_vectors'] = schedule_features.get('sched_num_vectors', 0)
    schedule_features['computation_efficiency'] = (schedule_features.get('sched_points_computed_total', 0) /
                                                 schedule_features.get('sched_bytes_at_realization', 1)) if schedule_features.get('sched_bytes_at_realization', 0) != 0 else 0
    schedule_features['memory_pressure'] = (schedule_features.get('sched_working_set', 0) /
                                          schedule_features.get('sched_bytes_at_root', 1)) if schedule_features.get('sched_bytes_at_root', 0) != 0 else 0
    schedule_features['memory_utilization_ratio'] = (schedule_features.get('sched_unique_bytes_read_per_realization', 0) /
                                                   schedule_features.get('sched_bytes_at_task', 1)) if schedule_features.get('sched_bytes_at_task', 0) != 0 else 0
    schedule_features['bytes_processing_rate'] = (schedule_features.get('sched_bytes_at_realization', 0) /
                                                schedule_features.get('execution_time_ms', 1)) if schedule_features.get('execution_time_ms', 0) != 0 else 0
    schedule_features['bytes_per_parallelism'] = (schedule_features.get('sched_bytes_at_task', 0) /
                                                schedule_features.get('total_parallelism', 1)) if schedule_features.get('total_parallelism', 0) != 0 else 0
    schedule_features['bytes_per_vector'] = (schedule_features.get('sched_bytes_at_realization', 0) /
                                           schedule_features.get('sched_num_vectors', 1)) if schedule_features.get('sched_num_vectors', 0) != 0 else 0
    nodes_count = len(json_data['children'])
    edges_count = sum(len(node.get('children', [])) for node in json_data['children'])
    program_features['nodes_count'] = nodes_count
    program_features['edges_count'] = edges_count
    program_features['node_edge_ratio'] = nodes_count / (edges_count + 1)
    schedule_features['nodes_per_schedule'] = nodes_count / (schedule_features.get('scheduling_count', 1)) if schedule_features.get('scheduling_count', 0) != 0 else 0
    program_features['op_diversity'] = len([k for k, v in program_features.items() if k.startswith('op_') and v > 0])
    
    fixed_program_features = {key: program_features.get(key, 0.0) for key in PROGRAM_FEATURES}
    fixed_schedule_features = {key: schedule_features.get(key, 0.0) for key in SCHEDULE_FEATURES}
    return fixed_program_features, fixed_schedule_features

# Process Tree_Output directory
def process_tree_output_directory(main_dir):
    all_program_features = []
    all_schedule_features = []
    file_names = []
    invalid_files = []
    
    for root, dirs, files in os.walk(main_dir):
        if 'tree_representation.json' in files:
            file_path = os.path.join(root, 'tree_representation.json')
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                program_features, schedule_features = extract_features(json_data)
                if schedule_features['execution_time_ms'] <= 0 or not np.isfinite(schedule_features['execution_time_ms']):
                    invalid_files.append(file_path)
                    print(f"Skipped file with invalid execution time: {file_path}")
                    continue
                all_program_features.append(program_features)
                all_schedule_features.append(schedule_features)
                file_names.append(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                invalid_files.append(file_path)
    
    if not all_program_features:
        raise ValueError("No valid JSON files with valid execution times found in Tree_Output directory.")
    
    log_path = os.path.join(main_dir, 'invalid_files_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("Files with invalid execution times or errors (skipped):\n")
        for file_path in invalid_files:
            f.write(f"{file_path}\n")
    
    total_files = len(all_program_features)
    print(f"Total valid files found: {total_files}")
    print(f"Files skipped due to invalid execution times or errors: {len(invalid_files)}")
    if total_files < 50:
        raise ValueError(f"Expected at least 50 valid files, found {total_files}")
    
    combined = list(zip(all_program_features, all_schedule_features, file_names))
    random.shuffle(combined)
    all_program_features, all_schedule_features, file_names = zip(*combined)
    
    test_size = min(50, total_files)
    train_program_features = all_program_features[:-test_size]
    train_schedule_features = all_schedule_features[:-test_size]
    test_program_features = all_program_features[-test_size:]
    test_schedule_features = all_schedule_features[-test_size:]
    train_file_names = file_names[:-test_size]
    test_file_names = file_names[-test_size:]
    
    print(f"Training files: {len(train_program_features)}")
    print(f"Testing files: {len(test_program_features)}")
    
    return train_program_features, train_schedule_features, test_program_features, test_schedule_features, list(test_file_names)

# Prepare data for model
def prepare_data_for_model(train_program_features, train_schedule_features, test_program_features, test_schedule_features):
    important_features = [
        'cache_hits', 'bytes_processing_rate', 'sched_bytes_at_task', 'sched_working_set_at_root',
        'sched_bytes_at_realization', 'sched_unique_bytes_read_per_realization'
    ]
    
    train_program_df = pd.DataFrame(train_program_features)
    train_schedule_df = pd.DataFrame(train_schedule_features)
    test_program_df = pd.DataFrame(test_program_features)
    test_schedule_df = pd.DataFrame(test_schedule_features)
    
    low_importance_features = [
        'op_cast', 'op_selfcall', 'memory_pointwise_1', 'memory_transpose_1', 'memory_broadcast_1',
        'memory_slice_1', 'op_select', 'op_not', 'op_and', 'op_ne', 'op_mod', 'memory_pointwise_2',
        'memory_broadcast_2', 'memory_slice_2', 'memory_transpose_2', 'op_externcall', 'op_imagecall',
        'op_param', 'memory_pointwise_3', 'memory_transpose_3', 'op_sub', 'memory_pointwise_0', 'op_let'
    ]
    train_program_df = train_program_df.drop(columns=[col for col in low_importance_features if col in train_program_df.columns])
    test_program_df = test_program_df.drop(columns=[col for col in low_importance_features if col in test_program_df.columns])
    train_schedule_df = train_schedule_df.drop(columns=[col for col in low_importance_features if col in train_schedule_df.columns])
    test_schedule_df = test_schedule_df.drop(columns=[col for col in low_importance_features if col in test_schedule_df.columns])
    
    skewed_features = ['cache_hits', 'bytes_processing_rate', 'sched_bytes_at_task', 'computation_efficiency']
    for feature in skewed_features:
        if feature in train_schedule_df.columns:
            train_schedule_df[f'log_{feature}'] = np.log1p(train_schedule_df[feature])
            test_schedule_df[f'log_{feature}'] = np.log1p(test_schedule_df[feature])
            train_schedule_df = train_schedule_df.drop(columns=[feature])
            test_schedule_df = test_schedule_df.drop(columns=[feature])
    
    train_program_df = train_program_df.fillna(0)
    test_program_df = test_program_df.fillna(0)
    train_schedule_df = train_schedule_df.fillna(0)
    test_schedule_df = test_schedule_df.fillna(0)
    
    constant_columns_program = [col for col in train_program_df.columns if train_program_df[col].nunique() == 1]
    train_program_df = train_program_df.drop(columns=constant_columns_program)
    test_program_df = test_program_df.drop(columns=constant_columns_program)
    
    constant_columns_schedule = [col for col in train_schedule_df.columns if train_schedule_df[col].nunique() == 1]
    train_schedule_df = train_schedule_df.drop(columns=constant_columns_schedule)
    test_schedule_df = test_schedule_df.drop(columns=constant_columns_schedule)
    
    y_train_raw = np.array([f['execution_time_ms'] for f in train_schedule_features])
    y_test_raw = np.array([f['execution_time_ms'] for f in test_schedule_features])
    y_train_raw = np.clip(y_train_raw, 0, np.percentile(y_train_raw, 99))
    y_test_raw = np.clip(y_test_raw, 0, np.percentile(y_test_raw, 99))
    
    y_train = np.log1p(y_train_raw).reshape(-1, 1)
    y_test = np.log1p(y_test_raw).reshape(-1, 1)
    
    scaler_program = RobustScaler()
    scaler_schedule = RobustScaler()
    scaler_y = RobustScaler()
    
    train_program_scaled = scaler_program.fit_transform(train_program_df)
    test_program_scaled = scaler_program.transform(test_program_df)
    train_schedule_scaled = scaler_schedule.fit_transform(train_schedule_df)
    test_schedule_scaled = scaler_schedule.transform(test_schedule_df)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    train_program_scaled = np.nan_to_num(train_program_scaled, nan=0.0)
    test_program_scaled = np.nan_to_num(test_program_scaled, nan=0.0)
    train_schedule_scaled = np.nan_to_num(train_schedule_scaled, nan=0.0)
    test_schedule_scaled = np.nan_to_num(test_schedule_scaled, nan=0.0)
    y_train_scaled = np.nan_to_num(y_train_scaled, nan=0.0)
    y_test_scaled = np.nan_to_num(y_test_scaled, nan=0.0)
    
    train_program_aug = []
    train_schedule_aug = []
    y_train_aug = []
    for i in range(len(train_program_features)):
        train_program_aug.append(train_program_scaled[i])
        train_schedule_aug.append(train_schedule_scaled[i])
        y_train_aug.append(y_train_scaled[i])
        
        cache_hits_idx = train_schedule_df.columns.get_loc('log_cache_hits') if 'log_cache_hits' in train_schedule_df.columns else -1
        bytes_rate_idx = train_schedule_df.columns.get_loc('log_bytes_processing_rate') if 'log_bytes_processing_rate' in train_schedule_df.columns else -1
        
        is_significant = False
        if cache_hits_idx != -1 and train_schedule_scaled[i, cache_hits_idx] > np.percentile(train_schedule_scaled[:, cache_hits_idx], 75):
            is_significant = True
        if bytes_rate_idx != -1 and train_schedule_scaled[i, bytes_rate_idx] > np.percentile(train_schedule_scaled[:, bytes_rate_idx], 75):
            is_significant = True
        
        augment_count = 3 if is_significant else 1
        for _ in range(augment_count):
            noise_program = np.random.normal(0, 0.05, train_program_scaled[i].shape)
            noise_schedule = np.random.normal(0, 0.05, train_schedule_scaled[i].shape)
            noise_y = np.random.normal(0, 0.05, y_train_scaled[i].shape)
            train_program_aug.append(train_program_scaled[i] + noise_program)
            train_schedule_aug.append(train_schedule_scaled[i] + noise_schedule)
            y_train_aug.append(y_train_scaled[i] + noise_y)
    
    train_program_tensor = torch.FloatTensor(np.array(train_program_aug))
    train_schedule_tensor = torch.FloatTensor(np.array(train_schedule_aug))
    y_train_tensor = torch.FloatTensor(np.array(y_train_aug))
    test_program_tensor = torch.FloatTensor(test_program_scaled)
    test_schedule_tensor = torch.FloatTensor(test_schedule_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Program input size: {train_program_tensor.shape[1]}")
    print(f"Schedule input size: {train_schedule_tensor.shape[1]}")
    
    return (train_program_tensor, train_schedule_tensor, y_train_tensor,
            test_program_tensor, test_schedule_tensor, y_test_tensor,
            scaler_y, train_program_tensor.shape[1], train_schedule_tensor.shape[1],
            train_program_df.columns, train_schedule_df.columns)

# Multi-Head Attention mechanism
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

# Modified LSTM Model
class SimpleLSTMModel(nn.Module):
    def __init__(self, program_input_size, schedule_input_size, hidden_sizes=[512, 256, 128], output_size=1, dropout_rate=0.2, num_heads=8):
        super(SimpleLSTMModel, self).__init__()
        self.program_fc = nn.Linear(program_input_size, hidden_sizes[0])
        self.schedule_fc = nn.Linear(schedule_input_size, hidden_sizes[0])
        self.program_bn = nn.BatchNorm1d(hidden_sizes[0])
        self.schedule_bn = nn.BatchNorm1d(hidden_sizes[0])
        
        self.attention = MultiHeadAttention(hidden_sizes[0] * 2, num_heads, dropout_rate)
        
        combined_size = hidden_sizes[0] * 2
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
    
    def forward(self, program_input, schedule_input):
        program_out = self.program_fc(program_input)
        program_out = self.program_bn(program_out)
        program_out = self.gelu(program_out)
        program_out = self.dropout(program_out)
        
        schedule_out = self.schedule_fc(schedule_input)
        schedule_out = self.schedule_bn(schedule_out)
        schedule_out = self.gelu(schedule_out)
        schedule_out = self.dropout(schedule_out)
        
        combined = torch.cat((program_out, schedule_out), dim=1)
        combined = self.attention(combined.unsqueeze(1)).squeeze(1)
        
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
def custom_loss(outputs, targets, schedule_inputs, feature_indices, feature_importances, huber_delta=0.5, mae_weight=0.3, l1_lambda=1e-5):
    huber = nn.HuberLoss(delta=huber_delta)(outputs, targets)
    mae = torch.mean(torch.abs(outputs - targets))
    l1_reg = sum(param.abs().sum() for param in model.parameters()) * l1_lambda
    
    weights = torch.ones_like(targets)
    for feature, idx in feature_indices.items():
        if idx != -1 and feature in feature_importances:
            feature_vals = schedule_inputs[:, idx]
            importance = feature_importances[feature]
            weights = torch.where(
                feature_vals > 1.0,
                weights * (1.0 + importance * 2.0),
                weights
            )
    
    weighted_huber = (huber * weights).mean()
    weighted_mae = (mae * weights).mean()
    return weighted_huber + mae_weight * weighted_mae + l1_reg

# Create data loaders with optimized workers
def create_data_loaders(train_program, train_schedule, y_train, test_program, test_schedule, y_test, batch_size=64):
    # Optimize number of workers based on CPU cores
    num_workers = min(os.cpu_count(), 4)  # Limit to 4 to avoid overloading CPU
    print(f"Using {num_workers} workers for DataLoader")
    
    train_dataset = TensorDataset(train_program, train_schedule, y_train)
    test_dataset = TensorDataset(test_program, test_schedule, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader

# Train the model with distributed workload
def train_model(model, train_loader, test_loader, criterion, optimizer, feature_indices, feature_importances, 
                device_manager, num_epochs=1000, patience=50, accumulation_steps=2):
    # Create a model copy for each device
    models = {}
    optimizers = {}
    for device in device_manager.devices:
        model_copy = SimpleLSTMModel(
            program_input_size=model.program_fc.in_features,
            schedule_input_size=model.schedule_fc.in_features,
            hidden_sizes=[512, 256, 128],
            output_size=1,
            dropout_rate=0.2,
            num_heads=8
        )
        model_copy.load_state_dict(model.state_dict())
        model_copy.to(device)
        model_copy.train()
        models[device] = model_copy
        optimizers[device] = optim.AdamW(model_copy.parameters(), lr=0.00005, weight_decay=1e-4)
    
    schedulers = {device: CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-6) for device, opt in optimizers.items()}
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_samples = 0
        
        for i, (program_inputs, schedule_inputs, targets) in enumerate(train_loader):
            batch_size = program_inputs.size(0)
            # Distribute batch across devices
            batch_splits = device_manager.distribute_batch((program_inputs, schedule_inputs, targets), batch_size)
            
            batch_loss = 0.0
            batch_samples = 0
            
            for device, split_size in batch_splits:
                if split_size == 0:
                    continue
                
                # Split the batch
                start_idx = batch_samples
                end_idx = start_idx + split_size
                prog_batch = program_inputs[start_idx:end_idx].to(device)
                sched_batch = schedule_inputs[start_idx:end_idx].to(device)
                target_batch = targets[start_idx:end_idx].to(device)
                
                model_copy = models[device]
                optimizer_copy = optimizers[device]
                optimizer_copy.zero_grad()
                
                outputs = model_copy(prog_batch, sched_batch)
                loss = criterion(outputs, target_batch, sched_batch, feature_indices, feature_importances)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Invalid loss detected at epoch {epoch+1}, batch {i+1} on {device}")
                    return None, None, None
                
                loss = loss / accumulation_steps
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model_copy.parameters(), max_norm=1.0)
                    optimizer_copy.step()
                    optimizer_copy.zero_grad()
                
                batch_loss += loss.item() * accumulation_steps * split_size
                batch_samples += split_size
            
            running_loss += batch_loss
            total_samples += batch_samples
        
        # Synchronize models across devices by averaging weights
        with torch.no_grad():
            avg_state_dict = models[device_manager.devices[0]].state_dict()
            for key in avg_state_dict.keys():
                avg_state_dict[key] = torch.zeros_like(avg_state_dict[key])
                for device in device_manager.devices:
                    avg_state_dict[key] += models[device].state_dict()[key].cpu()
                avg_state_dict[key] = avg_state_dict[key] / float(len(device_manager.devices))
            
            for device in device_manager.devices:
                models[device].load_state_dict(avg_state_dict)
        
        if len(train_loader) % accumulation_steps != 0:
            for device, optimizer_copy in optimizers.items():
                torch.nn.utils.clip_grad_norm_(models[device].parameters(), max_norm=1.0)
                optimizer_copy.step()
                optimizer_copy.zero_grad()
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase (use least loaded device)
        val_device = device_manager.get_least_loaded_device()
        models[val_device].eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for program_inputs, schedule_inputs, targets in test_loader:
                prog_batch = program_inputs.to(val_device)
                sched_batch = schedule_inputs.to(val_device)
                target_batch = targets.to(val_device)
                outputs = models[val_device](prog_batch, sched_batch)
                loss = criterion(outputs, target_batch, sched_batch, feature_indices, feature_importances)
                val_loss += loss.item() * program_inputs.size(0)
                val_samples += program_inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        
        for device in device_manager.devices:
            schedulers[device].step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss and not np.isnan(val_loss) and not np.isinf(val_loss):
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = models[val_device].state_dict().copy()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
        
        # Clear GPU memory
        for device in device_manager.devices:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    if best_model_state is not None and epochs_no_improve > 0:
        for device in device_manager.devices:
            models[device].load_state_dict(best_model_state)
    
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
    
    return train_losses, val_losses, models[device_manager.devices[0]]

# Evaluate the model with distributed inference
def evaluate_model(model, X_test_program, X_test_schedule, y_test, y_scaler, file_names_test, device_manager):
    model.eval()
    y_pred_actual_all = []
    y_test_actual_all = []
    
    # Create a DataLoader for inference
    test_dataset = TensorDataset(X_test_program, X_test_schedule, y_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=min(os.cpu_count(), 4))
    
    for program_inputs, schedule_inputs, targets in test_loader:
        batch_size = program_inputs.size(0)
        batch_splits = device_manager.distribute_batch((program_inputs, schedule_inputs, targets), batch_size)
        
        y_pred_scaled_batch = []
        y_test_batch = []
        idx = 0
        
        for device, split_size in batch_splits:
            if split_size == 0:
                continue
            
            start_idx = idx
            end_idx = start_idx + split_size
            prog_batch = program_inputs[start_idx:end_idx].to(device)
            sched_batch = schedule_inputs[start_idx:end_idx].to(device)
            target_batch = targets[start_idx:end_idx].to(device)
            
            with torch.no_grad():
                y_pred_scaled = model(prog_batch, sched_batch)
            
            y_pred_scaled_batch.append(y_pred_scaled.cpu())
            y_test_batch.append(target_batch.cpu())
            idx += split_size
        
        y_pred_scaled_batch = torch.cat(y_pred_scaled_batch, dim=0)
        y_test_batch = torch.cat(y_test_batch, dim=0)
        
        y_pred_transformed = y_scaler.inverse_transform(y_pred_scaled_batch.numpy())
        y_test_transformed = y_scaler.inverse_transform(y_test_batch.numpy())
        
        y_pred_actual = np.expm1(y_pred_transformed)
        y_test_actual = np.expm1(y_test_transformed)
        
        y_pred_actual_all.extend(y_pred_actual)
        y_test_actual_all.extend(y_test_actual)
    
    y_test_actual_all = np.array(y_test_actual_all)
    y_pred_actual_all = np.array(y_pred_actual_all)
    
    results_by_subfolder = {}
    for i, file_path in enumerate(file_names_test):
        subfolder = '/'.join(file_path.split('/')[:-1])
        if subfolder not in results_by_subfolder:
            results_by_subfolder[subfolder] = []
        
        pred = max(y_pred_actual_all[i][0], 0)
        results_by_subfolder[subfolder].append({
            'file': file_path,
            'actual': y_test_actual_all[i][0],
            'predicted': pred,
            'error_percentage': abs(y_test_actual_all[i][0] - pred) / y_test_actual_all[i][0] * 100 if y_test_actual_all[i][0] > 0 else 0
        })
    
    for subfolder, results in results_by_subfolder.items():
        print(f"\nResults for {subfolder}:")
        for result in results:
            print(f"File: {result['file']}")
            print(f"  Actual execution time: {result['actual']:.2f} ms")
            print(f"  Predicted execution time: {result['predicted']:.2f} ms")
            print(f"  Error percentage: {result['error_percentage']:.2f}%")
    
    mse = np.mean((y_test_actual_all - y_pred_actual_all) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_actual_all - y_pred_actual_all))
    mape = np.mean(np.abs((y_test_actual_all - y_pred_actual_all) / (y_test_actual_all + 1e-8))) * 100
    
    print("\nOverall Model Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    return y_test_actual_all, y_pred_actual_all

# Main function to save the model
def main(main_dir):
    # Initialize device manager
    device_manager = DeviceManager(usage_threshold=80.0)
    
    print(f"Processing main directory: {main_dir}")
    train_program_features, train_schedule_features, test_program_features, test_schedule_features, test_file_names = process_tree_output_directory(main_dir)
    
    if len(train_program_features) == 0 or len(test_program_features) == 0:
        print("Error: No valid training or test data found")
        return None
    
    (train_program_tensor, train_schedule_tensor, y_train,
     test_program_tensor, test_schedule_tensor, y_test,
     y_scaler, program_input_size, schedule_input_size, program_columns, schedule_columns) = prepare_data_for_model(
        train_program_features, train_schedule_features, test_program_features, test_schedule_features
    )
    
    train_loader, test_loader = create_data_loaders(
        train_program_tensor, train_schedule_tensor, y_train,
        test_program_tensor, test_schedule_tensor, y_test,
        batch_size=64
    )
    
    global model
    model = SimpleLSTMModel(
        program_input_size=program_input_size,
        schedule_input_size=schedule_input_size,
        hidden_sizes=[512, 256, 128],
        output_size=1,
        dropout_rate=0.2,
        num_heads=8
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-4)
    
    feature_importances = {
        'cache_hits': 0.5860,
        'bytes_processing_rate': 0.2893,
        'sched_bytes_at_task': 0.0422,
        'sched_working_set_at_root': 0.0248,
        'sched_bytes_at_realization': 0.0055,
        'sched_unique_bytes_read_per_realization': 0.0049
    }
    
    feature_indices = {}
    for feature in feature_importances.keys():
        log_feature = f'log_{feature}' if feature in ['cache_hits', 'bytes_processing_rate'] else feature
        if log_feature in schedule_columns:
            feature_indices[feature] = schedule_columns.get_loc(log_feature)
        else:
            feature_indices[feature] = schedule_columns.get_loc(feature) if feature in schedule_columns else -1
    
    print("Building and training Simple LSTM model...")
    train_losses, val_losses, trained_model = train_model(
        model, train_loader, test_loader,
        custom_loss, optimizer, feature_indices, feature_importances,
        device_manager, num_epochs=1000, patience=50, accumulation_steps=2
    )
    
    if train_losses is None or val_losses is None:
        print("Training failed due to invalid values")
        return None, None, None, None
    
    torch.save(trained_model.state_dict(), "model.pt")
    print("Model saved to model.pt")
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(
        trained_model, test_program_tensor, test_schedule_tensor, y_test,
        y_scaler, test_file_names, device_manager
    )
    
    print(f"\nSummary for Comparison:")
    print(f"Model: SimpleLSTM")
    
    return trained_model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "Tree_Output"
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    model, y_scaler, y_test_actual, y_pred_actual = main(main_dir)
