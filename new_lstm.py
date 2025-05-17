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
import time
import argparse

# Define fixed set of features
FIXED_FEATURES = [
    'cache_hits', 'cache_misses', 'execution_time_ms', 'sched_num_realizations',
    'sched_num_productions', 'sched_points_computed_total', 'sched_inner_parallelism',
    'sched_outer_parallelism', 'sched_bytes_at_realization', 'sched_bytes_at_production',
    'sched_bytes_at_root', 'sched_bytes_at_task', 'sched_working_set', 'sched_num_vectors',
    'sched_num_scalars', 'total_parallelism', 'scheduling_count', 'total_bytes_at_production',
    'total_vectors', 'computation_efficiency', 'memory_pressure', 'memory_utilization_ratio',
    'bytes_processing_rate', 'bytes_per_parallelism', 'bytes_per_vector', 'nodes_count',
    'edges_count', 'node_edge_ratio', 'nodes_per_schedule', 'op_diversity',
    'op_add', 'op_sub', 'op_mul', 'op_div', 'op_mod', 'op_eq', 'op_ne', 'op_lt', 'op_le',
    'op_or', 'op_and', 'op_not', 'op_min', 'op_max', 'op_constant', 'op_variable',
    'op_funccall', 'op_imagecall', 'op_externcall', 'op_let', 'op_param',
    'memory_transpose_0', 'memory_transpose_1', 'memory_transpose_2', 'memory_transpose_3',
    'memory_slice_0', 'memory_slice_1', 'memory_slice_2', 'memory_slice_3',
    'memory_broadcast_0', 'memory_broadcast_1', 'memory_broadcast_2', 'memory_broadcast_3',
    'memory_pointwise_0', 'memory_pointwise_1', 'memory_pointwise_2', 'memory_pointwise_3'
]

# Feature extraction function
def extract_features(json_data):
    try:
        if not isinstance(json_data, dict) or 'without_extern' not in json_data:
            return None
        without_extern = json_data['without_extern']
        if 'global_features' not in without_extern:
            return None
        global_features = without_extern['global_features']
        
        execution_time_ms = global_features.get('execution_time_ms', None)
        if execution_time_ms is None or not isinstance(execution_time_ms, (int, float)) or execution_time_ms <= 0:
            return None
        
        features = {}
        features['execution_time_ms'] = float(execution_time_ms)
        features['cache_hits'] = global_features.get('cache_hits', 0)
        features['cache_misses'] = global_features.get('cache_misses', 0)
        
        nodes = without_extern.get('nodes', [])
        edges = without_extern.get('edges', [])
        features['nodes_count'] = len(nodes)
        features['edges_count'] = len(edges)
        features['node_edge_ratio'] = features['nodes_count'] / (features['edges_count'] + 1e-8)
        
        op_counts = defaultdict(int)
        memory_patterns = defaultdict(lambda: [0, 0, 0, 0])
        
        for node in nodes:
            stages = node.get('stages', [])
            for stage in stages:
                pipeline_features = stage.get('pipeline_features', {})
                op_hist = pipeline_features.get('op_histogram', {}).get('Float', {})
                for op, count in op_hist.items():
                    op_counts[f'op_{op.lower()}'] += count
                
                mem_access = pipeline_features.get('memory_access_patterns', {}).get('Float', {})
                for pattern, values in mem_access.items():
                    for i, val in enumerate(values[:4]):
                        memory_patterns[pattern][i] += val
        
        features.update(op_counts)
        for pattern, values in memory_patterns.items():
            for i, val in enumerate(values):
                features[f'memory_{pattern.lower()}_{i}'] = val
        
        scheduling_features = []
        for node in nodes:
            stages = node.get('stages', [])
            for stage in stages:
                sched = stage.get('schedule_features', {})
                scheduling_features.append(sched)
        
        features['scheduling_count'] = len(scheduling_features)
        
        if scheduling_features:
            important_metrics = [
                'bytes_at_production', 'bytes_at_realization', 'bytes_at_root', 'bytes_at_task',
                'inner_parallelism', 'outer_parallelism', 'num_productions', 'num_realizations',
                'num_scalars', 'num_vectors', 'points_computed_total', 'working_set'
            ]
            
            for metric in important_metrics:
                features[f'sched_{metric}'] = sum(sf.get(metric, 0) for sf in scheduling_features)
            
            features['total_bytes_at_production'] = features['sched_bytes_at_production']
            features['total_vectors'] = features['sched_num_vectors']
            features['total_parallelism'] = sum(sf.get('inner_parallelism', 0) * sf.get('outer_parallelism', 1) 
                                              for sf in scheduling_features)
            
            features['bytes_per_vector'] = (features['total_bytes_at_production'] / 
                                          (features['total_vectors'] + 1e-8))
            features['memory_pressure'] = (features['sched_working_set'] / 
                                         (features['sched_bytes_at_production'] + 1e-8))
            features['bytes_per_parallelism'] = (features['total_bytes_at_production'] / 
                                               (features['total_parallelism'] + 1e-8))
            features['nodes_per_schedule'] = (features['nodes_count'] / 
                                            (features['scheduling_count'] + 1e-8))
        
        features['op_diversity'] = len([k for k, v in op_counts.items() if v > 0])
        
        features['computation_efficiency'] = (features['sched_points_computed_total'] / 
                                            (features['execution_time_ms'] + 1e-8))
        features['bytes_processing_rate'] = (features['total_bytes_at_production'] / 
                                            (features['execution_time_ms'] + 1e-8))
        features['memory_utilization_ratio'] = (features['sched_working_set'] / 
                                              (features['sched_bytes_at_production'] + 1e-8))
        
        fixed_features = {key: features.get(key, 0.0) for key in FIXED_FEATURES}
        return fixed_features
    
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Multi-Head Attention
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
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([self.head_dim])))
    
    def forward(self, x):
        batch_size = x.shape[0]
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        out = torch.matmul(attention, V).permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, -1, self.hidden_size)
        out = self.fc_out(out)
        return out

# LSTM Model
class SimpleLSTMModel(nn.Module):
    def __init__(self, seq_input_size, scalar_input_size, hidden_sizes=[512, 256, 128], output_size=1, dropout_rate=0.2, num_heads=8, use_attention=True):
        super(SimpleLSTMModel, self).__init__()
        self.use_attention = use_attention
        self.lstm_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        
        self.lstm_layers.append(nn.LSTM(seq_input_size, hidden_sizes[0], batch_first=True, bidirectional=True))
        self.ln_layers.append(nn.LayerNorm(hidden_sizes[0] * 2))
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(nn.LSTM(hidden_sizes[i-1] * 2, hidden_sizes[i], batch_first=True, bidirectional=True))
            self.ln_layers.append(nn.LayerNorm(hidden_sizes[i] * 2))
        
        if self.use_attention:
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
        
        if self.use_attention:
            context = self.attention(lstm_out).mean(dim=1)
        else:
            context = lstm_out.mean(dim=1)
        
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

# Preprocess a single JSON file
def preprocess_single_json(json_file, metadata, scaler_X_seq, scaler_X_scalar):
    try:
        # Load and extract features from the JSON file
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        features = extract_features(json_data)
        if features is None:
            print(f"Invalid JSON file or couldn't extract features: {json_file}")
            return None
        
        # SEQUENCE FEATURES
        seq_length = metadata.get('max_sequence_length', 3)
        seq_features = np.array([[features.get(key, 0.0) for key in FIXED_FEATURES]] * seq_length)
        
        # Reshape and scale sequence features exactly as in training
        seq_flat = seq_features.reshape(-1, len(FIXED_FEATURES))
        seq_flat_scaled = scaler_X_seq.transform(seq_flat)
        seq_tensor = torch.FloatTensor(seq_flat_scaled).view(1, seq_length, -1)
        
        # SCALAR FEATURES
        # Create scalar features dictionary
        scalar_features = {}
        scalar_feature_list = metadata.get('scalar_features', [])
        
        # Process standard and log-transformed features
        for feature in scalar_feature_list:
            if feature.startswith('log_'):
                # For log-transformed features
                base_feature = feature[4:]  # Remove 'log_' prefix
                if base_feature in features:
                    scalar_features[feature] = np.log1p(features[base_feature])
                else:
                    scalar_features[feature] = 0.0
            else:
                # For regular features
                scalar_features[feature] = features.get(feature, 0.0)
        
        # Create DataFrame with columns in the exact order as during training
        scalar_df = pd.DataFrame([scalar_features])
        
        # Ensure all required columns exist (add missing ones with zeros)
        for col in scalar_feature_list:
            if col not in scalar_df.columns:
                scalar_df[col] = 0.0
        
        # Keep only columns in the metadata list and in the right order
        scalar_df = scalar_df[scalar_feature_list]
        
        # Fill NaN values with zeros
        scalar_df = scalar_df.fillna(0)
        
        # Scale the scalar features using the training scaler
        scalar_scaled = scaler_X_scalar.transform(scalar_df)
        scalar_scaled = np.nan_to_num(scalar_scaled, nan=0.0)
        scalar_tensor = torch.FloatTensor(scalar_scaled)
        
        return seq_tensor, scalar_tensor
    
    except Exception as e:
        print(f"Error preprocessing JSON file: {json_file}")
        print(f"Error details: {e}")
        return None

# Main function with mode selection
def main():
    parser = argparse.ArgumentParser(description="LSTM Model for Execution Time Prediction")
    parser.add_argument("mode", choices=["T", "P"], help="Mode: T for training, P for prediction")
    parser.add_argument("--main_dir", default="/home/kowrisaan/jathu/Halide_New/Graph/Graph_Output", help="Main directory for training data")
    parser.add_argument("--json_file", help="Path to JSON file for prediction")
    
    args = parser.parse_args()
    
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    if args.mode == "P":
        if args.json_file is None:
            print("Error: JSON file path is required for prediction mode.")
            return
        
        # Load saved model
        try:
            model = torch.jit.load("model.pt")
            model.eval()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        # Load model metadata
        try:
            with open('model_metadata.json', 'r') as f:
                metadata = json.load(f)
            print("Model metadata loaded successfully.")
        except Exception as e:
            print(f"Error loading model metadata: {e}")
            return
        
        # Load and initialize node/sequence scaler
        try:
            with open('scaler_node_params.json', 'r') as f:
                scaler_node_params = json.load(f)
            scaler_X_seq = RobustScaler()
            scaler_X_seq.center_ = np.array(scaler_node_params['center'])
            scaler_X_seq.scale_ = np.array(scaler_node_params['scale'])
            print("Sequence scaler loaded successfully.")
        except Exception as e:
            print(f"Error loading sequence scaler: {e}")
            return
        
        # Load and initialize scalar features scaler
        try:
            with open('scaler_scalar_params.json', 'r') as f:
                scaler_scalar_params = json.load(f)
            scaler_X_scalar = RobustScaler()
            scaler_X_scalar.center_ = np.array(scaler_scalar_params['center'])
            scaler_X_scalar.scale_ = np.array(scaler_scalar_params['scale'])
            print("Scalar scaler loaded successfully.")
        except Exception as e:
            print(f"Error loading scalar scaler: {e}")
            return
        
        # Load and initialize output scaler
        try:
            with open('scaler_y_params.json', 'r') as f:
                scaler_y_params = json.load(f)
            y_scaler = RobustScaler()
            y_scaler.center_ = np.array(scaler_y_params['center'])
            y_scaler.scale_ = np.array(scaler_y_params['scale'])
            print("Output scaler loaded successfully.")
        except Exception as e:
            print(f"Error loading output scaler: {e}")
            return
        
        # Preprocess the input JSON file
        print(f"Preprocessing JSON file: {args.json_file}")
        result = preprocess_single_json(args.json_file, metadata, scaler_X_seq, scaler_X_scalar)
        if result is None:
            print("Failed to preprocess the JSON file.")
            return
        
        seq_tensor, scalar_tensor = result
        print(f"Sequence tensor shape: {seq_tensor.shape}")
        print(f"Scalar tensor shape: {scalar_tensor.shape}")
        
        # Check if shapes match what the model expects
        seq_input_size = metadata.get('seq_input_size')
        scalar_input_size = metadata.get('scalar_input_size')
        if seq_tensor.shape[2] != seq_input_size:
            print(f"Error: Sequence input size mismatch. Expected: {seq_input_size}, Got: {seq_tensor.shape[2]}")
            return
        if scalar_tensor.shape[1] != scalar_input_size:
            print(f"Error: Scalar input size mismatch. Expected: {scalar_input_size}, Got: {scalar_tensor.shape[1]}")
            return
        
        # Run prediction
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        try:
            model = model.to(device)
            seq_tensor = seq_tensor.to(device)
            scalar_tensor = scalar_tensor.to(device)
            
            with torch.no_grad():
                y_pred_scaled = model(seq_tensor, scalar_tensor)
            
            y_pred_scaled = y_pred_scaled.cpu().numpy()
            y_pred_transformed = y_scaler.inverse_transform(y_pred_scaled)
            y_pred_actual = np.expm1(y_pred_transformed)
            prediction = max(y_pred_actual[0][0], 0)
            
            print(f"Predicted execution time for {args.json_file}: {prediction:.2f} ms")
        except Exception as e:
            print(f"Error during prediction: {e}")
            return

if __name__ == "__main__":
    main()
