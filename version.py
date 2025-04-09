import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import seaborn as sns
from scipy import stats

# Define important metrics for scheduling sequence with additional features
important_metrics = [
    'bytes_at_production', 'bytes_at_realization', 'inner_parallelism', 'outer_parallelism',
    'num_vectors', 'points_computed_total', 'working_set', 'memory_bandwidth', 'compute_intensity'
]

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
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        execution_time = get_execution_time(file_path)
        if execution_time is None or not np.isfinite(execution_time):
            print(f"Warning: Invalid execution time in {file_path}")
            return None
        
        nodes_features = []
        edges_features = []
        programming_details = data.get("programming_details", None)
        
        if not programming_details or 'Nodes' not in programming_details or 'Edges' not in programming_details:
            print(f"Warning: Incomplete programming_details in {file_path}")
            return None
        
        # Extract node and edge details with enhanced feature extraction
        op_counts_per_node = []
        all_op_types = set()
        node_compute_intensities = []
        node_memory_footprints = []
        
        for node in programming_details['Nodes']:
            node_feature = {'Name': node.get('Name', '')}
            op_counts = {}
            
            # Extract operation counts
            if 'Details' in node and 'Op histogram' in node['Details']:
                op_hist = node['Details']['Op histogram']
                for op_line in op_hist:
                    parts = op_line.strip().split(':')
                    if len(parts) == 2:
                        op_name = parts[0].strip().lower()
                        op_count = int(parts[1].strip())
                        op_counts[f'op_{op_name}'] = op_count
                        all_op_types.add(f'op_{op_name}')
            
            # Extract memory usage estimates
            memory_footprint = 0
            if 'Details' in node and 'Memory usage' in node['Details']:
                memory_info = node['Details']['Memory usage']
                if isinstance(memory_info, list):
                    for mem_line in memory_info:
                        if isinstance(mem_line, str) and ':' in mem_line:
                            parts = mem_line.strip().split(':')
                            if len(parts) == 2 and 'bytes' in parts[0].lower():
                                try:
                                    memory_footprint = float(parts[1].strip())
                                except ValueError:
                                    pass
            
            # Compute intensity estimation (ops/byte)
            total_ops = sum(op_counts.values())
            compute_intensity = total_ops / max(memory_footprint, 1)
            
            nodes_features.append(node_feature)
            op_counts_per_node.append(op_counts)
            node_memory_footprints.append(memory_footprint)
            node_compute_intensities.append(compute_intensity)
        
        # Edge features with data flow characteristics
        for edge in programming_details['Edges']:
            edge_feature = {
                'From': edge.get('From', ''), 
                'To': edge.get('To', ''), 
                'Name': edge.get('Name', '')
            }
            
            if 'Details' in edge and 'Size' in edge['Details']:
                try:
                    edge_feature['DataSize'] = float(edge['Details']['Size'])
                except (ValueError, TypeError):
                    edge_feature['DataSize'] = 0.0
            else:
                edge_feature['DataSize'] = 0.0
                
            edges_features.append(edge_feature)
        
        # Enhanced Graph Embedding with Topological Features
        num_nodes = max(len(nodes_features), 1)
        num_edges = len(edges_features)
        total_ops = sum(sum(node.get(f'op_{op}', 0) for op in all_op_types) for node in op_counts_per_node)
        total_memory = sum(node_memory_footprints)
        avg_compute_intensity = sum(node_compute_intensities) / max(len(node_compute_intensities), 1)
        
        node_map = {node['Name']: i for i, node in enumerate(nodes_features)}
        
        adj_matrix = np.zeros((num_nodes, num_nodes))
        in_degree = np.zeros(num_nodes)
        out_degree = np.zeros(num_nodes)
        
        for edge in edges_features:
            from_idx = node_map.get(edge['From'], -1)
            to_idx = node_map.get(edge['To'], -1)
            if from_idx != -1 and to_idx != -1:
                adj_matrix[from_idx, to_idx] = 1
                out_degree[from_idx] += 1
                in_degree[to_idx] += 1
        
        betweenness = np.zeros(num_nodes)
        if num_nodes > 1:
            for k in range(num_nodes):
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i != j and adj_matrix[i, k] > 0 and adj_matrix[k, j] > 0:
                            betweenness[k] += 1
        
        max_possible_edges = num_nodes * (num_nodes - 1)
        graph_density = num_edges / max(max_possible_edges, 1)
        
        fixed_op_size = 15
        op_types = sorted(list(all_op_types))[:fixed_op_size]
        node_features = np.zeros((num_nodes, fixed_op_size))
        for i, op_counts in enumerate(op_counts_per_node):
            for j, op in enumerate(op_types):
                if j < fixed_op_size:
                    node_features[i, j] = op_counts.get(op, 0) / max(total_ops, 1)
        
        if num_nodes > 1:
            eig_centrality = np.ones(num_nodes)
            for _ in range(5):
                eig_centrality = np.dot(adj_matrix, eig_centrality)
                eig_centrality = eig_centrality / max(np.max(eig_centrality), 1e-10)
            
            damping = 0.85
            pagerank = np.ones(num_nodes) / num_nodes
            for _ in range(5):
                pagerank = (1 - damping) / num_nodes + damping * np.dot(adj_matrix.T, pagerank)
                pagerank = pagerank / np.sum(pagerank)
            
            node_importance = (in_degree + out_degree + betweenness + eig_centrality) / 4
            weighted_features = node_features * node_importance.reshape(-1, 1)
            graph_embedding = np.mean(weighted_features, axis=0)
        else:
            graph_embedding = np.mean(node_features, axis=0)
        
        graph_metrics = [
            num_nodes, num_edges, total_ops, len(all_op_types) / max(num_nodes, 1),
            graph_density, np.mean(in_degree), np.max(in_degree),
            np.mean(out_degree), np.max(out_degree), np.mean(betweenness),
            total_memory, avg_compute_intensity
        ]
        
        template_features = np.concatenate([graph_metrics, graph_embedding])
        scaler_template = PowerTransformer(method='yeo-johnson')
        template_features = scaler_template.fit_transform(template_features.reshape(1, -1)).flatten()
        template_features = np.nan_to_num(template_features, nan=0.0)
        
        # Enhanced Schedule-Specific Features
        scheduling_features = []
        scheduling_data = data.get("scheduling_data", None)
        if not scheduling_data and programming_details and 'Schedules' in programming_details:
            scheduling_data = programming_details['Schedules']
        
        if not scheduling_data:
            print(f"Warning: No scheduling data in {file_path}")
            return None
        
        for sched in scheduling_data:
            sched_feature = {'Name': sched.get('Name', '')}
            if 'Details' in sched and 'scheduling_feature' in sched['Details']:
                sf = sched['Details']['scheduling_feature']
                sched_feature.update(sf)
                
                if 'memory_bandwidth' not in sched_feature and 'bytes_at_production' in sf:
                    bytes_prod = sf.get('bytes_at_production', 0)
                    points_total = sf.get('points_computed_total', 1)
                    sched_feature['memory_bandwidth'] = bytes_prod / max(points_total, 1)
                
                if 'compute_intensity' not in sched_feature and 'working_set' in sf:
                    inner_p = sf.get('inner_parallelism', 1)
                    outer_p = sf.get('outer_parallelism', 1)
                    working_set = sf.get('working_set', 1)
                    sched_feature['compute_intensity'] = (inner_p * outer_p) / max(working_set, 1)
            
            scheduling_features.append(sched_feature)
        
        scheduling_sequence = []
        seq_length = len(scheduling_features)
        
        for i, sf in enumerate(scheduling_features):
            sched_vector = [float(sf.get(metric, 0.0)) for metric in important_metrics]
            bytes_prod = sf.get('bytes_at_production', 0.0)
            bytes_real = sf.get('bytes_at_realization', 0.0)
            points_total = sf.get('points_computed_total', 0.0)
            inner_p = sf.get('inner_parallelism', 0.0)
            outer_p = sf.get('outer_parallelism', 0.0)
            working_set = sf.get('working_set', 0.0)
            
            derived_features = [
                np.log1p(inner_p * outer_p + 1e-6),
                np.log1p(bytes_prod / max(points_total, 1e-4) + 1e-6),
                inner_p / max(outer_p, 1e-4),
                bytes_prod / max(bytes_real, 1e-4),
                np.log1p(working_set / max(points_total, 1e-4) + 1e-6),
                i / max(seq_length, 1),
                np.sin(2 * np.pi * i / max(seq_length, 1)),
                np.cos(2 * np.pi * i / max(seq_length, 1))
            ]
            
            sched_vector.extend(derived_features)
            noise = np.random.normal(0, 0.01, len(sched_vector))
            augmented_vector = np.array(sched_vector, dtype=np.float32) + noise
            combined_vector = np.concatenate([template_features, augmented_vector])
            scheduling_sequence.append(combined_vector)
        
        if not scheduling_sequence:
            empty_sched_vector = np.zeros(len(important_metrics) + 8, dtype=np.float32)
            scheduling_sequence = [np.concatenate([template_features, empty_sched_vector])]
        
        seq_array = np.array(scheduling_sequence)
        scaler_seq = PowerTransformer(method='yeo-johnson')
        try:
            scheduling_sequence = scaler_seq.fit_transform(seq_array)
        except:
            scaler_seq = RobustScaler()
            scheduling_sequence = scaler_seq.fit_transform(seq_array)
        
        scheduling_sequence = np.nan_to_num(scheduling_sequence, nan=0.0).tolist()
        
        return {
            'scheduling_sequence': scheduling_sequence,
            'execution_time': execution_time,
            'sequence_length': len(scheduling_sequence),
            'num_nodes': num_nodes,
            'num_edges': num_edges
        }
    except Exception as e:
        print(f"Error in extract_features_from_file: {str(e)} for {file_path}")
        return None

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
    
    execution_times = np.array([f['execution_time'] for f in all_features])
    quantiles = pd.qcut(execution_times, 5, labels=False, duplicates='drop')
    
    combined = list(zip(all_features, all_file_names, quantiles))
    random.shuffle(combined)
    
    quantile_groups = {}
    for feature, file_name, q in combined:
        if q not in quantile_groups:
            quantile_groups[q] = []
        quantile_groups[q].append((feature, file_name))
    
    test_size = 50
    test_features = []
    test_file_names = []
    
    quantiles_unique = sorted(quantile_groups.keys())
    samples_per_quantile = test_size // len(quantiles_unique)
    remaining = test_size - samples_per_quantile * len(quantiles_unique)
    
    for q in quantiles_unique:
        quota = samples_per_quantile + (1 if remaining > 0 else 0)
        remaining -= 1 if remaining > 0 else 0
        group = quantile_groups[q]
        test_subset = group[:quota]
        quantile_groups[q] = group[quota:]
        
        for feature, file_name in test_subset:
            test_features.append(feature)
            test_file_names.append(file_name)
    
    train_features = []
    train_file_names = []
    for q in quantiles_unique:
        for feature, file_name in quantile_groups[q]:
            train_features.append(feature)
            train_file_names.append(file_name)
    
    print(f"\nTotal files: {total_files}")
    print(f"Training files: {len(train_features)}")
    print(f"Testing files: {len(test_features)}")
    
    return train_features, test_features, test_file_names

def prepare_data_for_model(train_features, test_features):
    train_lengths = [f['sequence_length'] for f in train_features]
    test_lengths = [f['sequence_length'] for f in test_features]
    
    train_sequences = [torch.FloatTensor(f['scheduling_sequence']) for f in train_features]
    test_sequences = [torch.FloatTensor(f['scheduling_sequence']) for f in test_features]
    
    train_sequences_padded = pad_sequence(train_sequences, batch_first=True)
    test_sequences_padded = pad_sequence(test_sequences, batch_first=True)
    
    train_sorted_idx = np.argsort([-length for length in train_lengths])
    train_sequences_padded = train_sequences_padded[train_sorted_idx]
    train_lengths = [train_lengths[i] for i in train_sorted_idx]
    
    y_train_raw = np.array([train_features[i]['execution_time'] for i in train_sorted_idx])
    y_test_raw = np.array([f['execution_time'] for f in test_features])
    
    y_train_raw = np.clip(y_train_raw, 1e-6, np.percentile(y_train_raw, 99.5))
    y_test_raw = np.clip(y_test_raw, 1e-6, np.percentile(y_test_raw, 99.5))
    
    pt = PowerTransformer(method='yeo-johnson')
    y_train_transformed = pt.fit_transform(np.log1p(y_train_raw).reshape(-1, 1))
    y_test_transformed = pt.transform(np.log1p(y_test_raw).reshape(-1, 1))
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_transformed)
    y_test_scaled = scaler_y.transform(y_test_transformed)
    
    y_train_scaled = np.nan_to_num(y_train_scaled, nan=0.0)
    y_test_scaled = np.nan_to_num(y_test_scaled, nan=0.0)
    
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    y_transform = {'power_transform': pt, 'scaler': scaler_y}
    
    print(f"Sequence input size: {train_sequences_padded.shape[2]}")
    
    return (train_sequences_padded, y_train_tensor, train_lengths,
            test_sequences_padded, y_test_tensor, test_lengths,
            y_transform, train_sequences_padded.shape[2])

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        residual = x
        attn_output, _ = self.multihead_attn(x, x, x, key_padding_mask=mask)
        attn_output = self.dropout(attn_output)
        output = self.layer_norm(residual + attn_output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return self.norm(residual + x)

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, x, mask=None):
        scores = self.attention(x)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            scores = scores.masked_fill(mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights, dim=1)
        return context

class HybridTemporalNet(nn.Module):
    def __init__(self, seq_input_size, hidden_sizes=[384, 256, 128], 
                 output_size=1, dropout_rate=0.3, num_heads=8, num_layers=3):
        super(HybridTemporalNet, self).__init__()
        
        self.input_proj = nn.Linear(seq_input_size, hidden_sizes[0])
        self.pos_encoder = PositionalEncoding(hidden_sizes[0])
        self.input_dropout = nn.Dropout(dropout_rate)
        self.input_norm = nn.LayerNorm(hidden_sizes[0])
        
        # Bidirectional LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.lstm_norms = nn.ModuleList()
        
        curr_size = hidden_sizes[0]
        for i, hidden_size in enumerate(hidden_sizes):
            if i > 0:
                self.lstm_layers.append(
                    nn.LSTM(curr_size, hidden_size, batch_first=True, bidirectional=True)
                )
                self.lstm_norms.append(nn.LayerNorm(hidden_size * 2))
                curr_size = hidden_size * 2
        
        # Transformer encoder blocks
        transformer_dim = hidden_sizes[-1] * 2
        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            block = nn.ModuleList([
                SelfAttention(transformer_dim, num_heads, dropout_rate),
                FeedForward(transformer_dim, transformer_dim * 4, dropout_rate)
            ])
            self.transformer_blocks.append(block)
            
        # Enhanced attention pooling
        self.attn_pool = AttentionPooling(transformer_dim)
        
        # Output layers with residual connections and layer normalization
        self.fc_layers = nn.ModuleList()
        self.fc_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        sizes = [transformer_dim, 256, 128, 64]
        for i in range(len(sizes) - 1):
            self.fc_layers.append(nn.Linear(sizes[i], sizes[i+1]))
            self.fc_norms.append(nn.LayerNorm(sizes[i+1]))
            self.dropouts.append(nn.Dropout(dropout_rate))
            
        self.output_layer = nn.Linear(sizes[-1], output_size)
        
        self.gelu = nn.GELU()
        
    def forward(self, seq_input, lengths=None):
        # Initial projection and positional encoding
        x = self.input_proj(seq_input)
        x = self.pos_encoder(x)
        x = self.input_dropout(x)
        x = self.input_norm(x)
        
        # Create attention mask from sequence lengths if provided
        mask = None
        if lengths is not None:
            max_len = seq_input.size(1)
            mask = torch.arange(max_len, device=seq_input.device)[None, :] >= torch.tensor(lengths, device=seq_input.device)[:, None]
        
        # Process through LSTM layers
        for lstm, norm in zip(self.lstm_layers, self.lstm_norms):
            if lengths is not None:
                packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
                packed_output, _ = lstm(packed_x)
                x, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=seq_input.size(1))
            else:
                x, _ = lstm(x)
            x = norm(x)
        
        # Process through Transformer blocks
        for attn, ff in self.transformer_blocks:
            x = attn(x, mask=mask)
            x = ff(x)
        
        # Attention pooling to aggregate sequence
        context = self.attn_pool(x, mask=mask)
        
        # Fully connected layers with residual connections
        for fc, norm, dropout in zip(self.fc_layers, self.fc_norms, self.dropouts):
            residual = context if context.size(-1) == fc.in_features else nn.Linear(context.size(-1), fc.in_features)(context)
            x = fc(context)
            x = x + residual
            x = self.gelu(x)
            x = norm(x)
            x = dropout(x)
            context = x
        
        # Final output
        output = self.output_layer(x)
        return output

def combined_loss(outputs, targets, alpha=0.7, gamma=2.0, mse_weight=0.3):
    mse = nn.MSELoss()(outputs, targets)
    focal_mse = (outputs - targets) ** 2
    pt = torch.exp(-focal_mse)
    focal = alpha * (1 - pt) ** gamma * focal_mse
    rel_error = torch.mean(torch.abs(outputs - targets) / (targets.abs() + 1e-6))
    return mse_weight * mse + (1 - mse_weight) * torch.mean(focal) + 0.1 * rel_error

def create_data_loaders(train_sequences, y_train, train_lengths, test_sequences, y_test, test_lengths, batch_size=32):
    train_dataset = TensorDataset(train_sequences, y_train, torch.tensor(train_lengths))
    test_dataset = TensorDataset(test_sequences, y_test, torch.tensor(test_lengths))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=500, patience=50, accumulation_steps=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model.to(device)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for i, (seq_inputs, targets, lengths) in enumerate(train_loader):
            seq_inputs, targets, lengths = seq_inputs.to(device), targets.to(device), lengths.to(device)
            outputs = model(seq_inputs, lengths.cpu().tolist())
            loss = criterion(outputs, targets)
            
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
            for seq_inputs, targets, lengths in test_loader:
                seq_inputs, targets, lengths = seq_inputs.to(device), targets.to(device), lengths.to(device)
                outputs = model(seq_inputs, lengths.cpu().tolist())
                loss = criterion(outputs, targets)
                val_loss += loss.item() * seq_inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
        
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
    plt.show()
    
    return train_losses, val_losses

def evaluate_model(model, X_test_seq, y_test, test_lengths, y_transform, file_names_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    X_test_seq = X_test_seq.to(device)
    with torch.no_grad():
        y_pred_scaled = model(X_test_seq, test_lengths)
    
    y_pred_scaled = y_pred_scaled.cpu().numpy()
    y_test = y_test.cpu().numpy()
    
    y_test_transformed = y_transform['scaler'].inverse_transform(y_test)
    y_pred_transformed = y_transform['scaler'].inverse_transform(y_pred_scaled)
    
    y_test_actual = np.expm1(y_transform['power_transform'].inverse_transform(y_test_transformed))
    y_pred_actual = np.expm1(y_transform['power_transform'].inverse_transform(y_pred_transformed))
    
    y_pred_actual = np.clip(y_pred_actual, 0, np.percentile(y_test_actual, 99))
    
    results_by_subfolder = {}
    for i, file_path in enumerate(file_names_test):
        subfolder = file_path.split('/')[0]
        if subfolder not in results_by_subfolder:
            results_by_subfolder[subfolder] = []
        
        pred = max(y_pred_actual[i][0], 0)
        actual = y_test_actual[i][0]
        results_by_subfolder[subfolder].append({
            'file': file_path,
            'actual': actual,
            'predicted': pred,
            'error_percentage': abs(actual - pred) / actual * 100 if actual > 0 else 0
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
    
    print(f"Total training samples: {len(train_features)}")
    print(f"Total test samples: {len(test_features)}")
    
    if len(train_features) == 0 or len(test_features) == 0:
        print("Error: No valid training or test data found")
        return None
    
    (train_sequences, y_train, train_lengths,
     test_sequences, y_test, test_lengths,
     y_transform, seq_input_size) = prepare_data_for_model(train_features, test_features)
    
    train_loader, test_loader = create_data_loaders(
        train_sequences, y_train, train_lengths,
        test_sequences, y_test, test_lengths,
        batch_size=32
    )
    
    global model
    model = HybridTemporalNet(
        seq_input_size=seq_input_size,
        hidden_sizes=[384, 256, 128],
        output_size=1,
        dropout_rate=0.3,
        num_heads=8,
        num_layers=3
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    
    print("Building and training Hybrid Temporal Network...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader,
        combined_loss, optimizer, scheduler,
        num_epochs=500, patience=50, accumulation_steps=2
    )
    
    if train_losses is None or val_losses is None:
        print("Training failed due to invalid values")
        return None
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(
        model, test_sequences, y_test, test_lengths,
        y_transform, test_file_names
    )
    
    print(f"\nSummary for Comparison:")
    print(f"Model: HybridTemporalNet")
    
    return model, y_transform, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "synthetic_data"
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    model, y_transform, y_test_actual, y_pred_actual = main(main_dir)
