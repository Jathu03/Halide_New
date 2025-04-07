import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Define important metrics for scheduling sequence (schedule-specific)
important_metrics = [
    'bytes_at_production', 'bytes_at_realization', 'inner_parallelism', 'outer_parallelism',
    'num_vectors', 'points_computed_total', 'working_set'
]

# Add derived metrics that may help the model
derived_metrics = [
    'bytes_per_point',        # Ratio of bytes to points
    'parallelism_product',    # Product of inner and outer parallelism
    'bytes_density',          # Bytes per vector
    'compute_efficiency'      # Points per working set
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
    
    # Extract node and edge details with improved feature extraction
    op_counts_per_node = []
    all_op_types = set()
    node_depths = {}  # Track depth of each node in the graph
    node_fanouts = {}  # Track outgoing connections per node
    node_fanins = {}   # Track incoming connections per node
    
    # Initialize counts
    for node in programming_details['Nodes']:
        node_name = node.get('Name', '')
        node_fanouts[node_name] = 0
        node_fanins[node_name] = 0
    
    # Count edges
    for edge in programming_details['Edges']:
        from_node = edge.get('From', '')
        to_node = edge.get('To', '')
        if from_node in node_fanouts:
            node_fanouts[from_node] += 1
        if to_node in node_fanins:
            node_fanins[to_node] += 1
    
    # Build directed graph for depth calculation
    graph = {node.get('Name', ''): [] for node in programming_details['Nodes']}
    for edge in programming_details['Edges']:
        from_node = edge.get('From', '')
        to_node = edge.get('To', '')
        if from_node in graph:
            graph[from_node].append(to_node)
    
    # Calculate node depths using BFS
    def calculate_depths(graph):
        depths = {}
        for start_node in graph:
            if start_node not in node_fanins or node_fanins[start_node] == 0:  # Start with source nodes
                queue = [(start_node, 0)]
                visited = set([start_node])
                
                while queue:
                    node, depth = queue.pop(0)
                    depths[node] = max(depth, depths.get(node, 0))
                    
                    for neighbor in graph.get(node, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, depth + 1))
        return depths
    
    node_depths = calculate_depths(graph)
    
    # Extract detailed node features
    for node in programming_details['Nodes']:
        node_name = node.get('Name', '')
        node_feature = {
            'Name': node_name,
            'Depth': node_depths.get(node_name, 0),
            'FanIn': node_fanins.get(node_name, 0),
            'FanOut': node_fanouts.get(node_name, 0),
        }
        
        op_counts = {}
        if 'Details' in node and 'Op histogram' in node['Details']:
            op_hist = node['Details']['Op histogram']
            for op_line in op_hist:
                parts = op_line.strip().split(':')
                if len(parts) == 2:
                    op_name = parts[0].strip().lower()
                    op_count = int(parts[1].strip())
                    op_counts[f'op_{op_name}'] = op_count
                    all_op_types.add(f'op_{op_name}')
        
        nodes_features.append(node_feature)
        op_counts_per_node.append(op_counts)
    
    # Process edge features
    edge_types = set()
    for edge in programming_details['Edges']:
        edge_name = edge.get('Name', '')
        edge_type = edge_name.split(':')[0] if ':' in edge_name else edge_name
        edge_types.add(edge_type)
        
        edge_feature = {
            'From': edge.get('From', ''), 
            'To': edge.get('To', ''), 
            'Name': edge_name,
            'Type': edge_type
        }
        edges_features.append(edge_feature)
    
    # Graph embedding with improved graph structure analysis
    num_nodes = max(len(nodes_features), 1)
    num_edges = len(edges_features)
    max_depth = max([node.get('Depth', 0) for node in nodes_features]) if nodes_features else 0
    avg_depth = np.mean([node.get('Depth', 0) for node in nodes_features]) if nodes_features else 0
    max_fanout = max([node.get('FanOut', 0) for node in nodes_features]) if nodes_features else 0
    avg_fanout = np.mean([node.get('FanOut', 0) for node in nodes_features]) if nodes_features else 0
    
    # Compute total operations
    total_ops = sum(sum(node.get(f'op_{op}', 0) for op in all_op_types) for node in op_counts_per_node)
    
    # Create adjacency matrix
    node_map = {node['Name']: i for i, node in enumerate(nodes_features)}
    adj_matrix = np.zeros((num_nodes, num_nodes))
    for edge in edges_features:
        from_idx = node_map.get(edge['From'], -1)
        to_idx = node_map.get(edge['To'], -1)
        if from_idx != -1 and to_idx != -1:
            adj_matrix[from_idx, to_idx] = 1
    
    # Create node feature matrix with enhanced features
    fixed_op_size = 15  # Increased from 10 to capture more operation types
    op_types = sorted(list(all_op_types))[:fixed_op_size]
    node_features = np.zeros((num_nodes, fixed_op_size + 3))  # +3 for depth, fanin, fanout
    
    for i, (node, op_counts) in enumerate(zip(nodes_features, op_counts_per_node)):
        # Add operation counts
        for j, op in enumerate(op_types):
            node_features[i, j] = op_counts.get(op, 0) / max(total_ops, 1)
        
        # Add structural features
        node_features[i, fixed_op_size] = node.get('Depth', 0) / max(max_depth, 1)
        node_features[i, fixed_op_size+1] = node.get('FanIn', 0) / max(max_fanout, 1)
        node_features[i, fixed_op_size+2] = node.get('FanOut', 0) / max(max_fanout, 1)
    
    # Graph-level structural features
    graph_density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
    connectedness = np.sum(adj_matrix) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
    
    # Enhanced graph embedding with attention to critical paths
    if num_nodes > 1:
        # Apply attention to nodes based on depth (deeper nodes often more critical)
        depth_attention = np.array([node.get('Depth', 0) for node in nodes_features]) / max(max_depth, 1)
        depth_attention = np.exp(depth_attention) / np.sum(np.exp(depth_attention))  # Softmax normalization
        weighted_features = node_features * depth_attention.reshape(-1, 1)
        graph_embedding = np.mean(weighted_features, axis=0)
        
        # Add path analysis - longest path through the graph
        graph_embedding = np.concatenate([
            graph_embedding,
            [max_depth / num_nodes, graph_density, connectedness]
        ])
    else:
        graph_embedding = np.mean(node_features, axis=0)
        graph_embedding = np.concatenate([
            graph_embedding,
            [0, 0, 0]  # Placeholder values for single-node graphs
        ])
    
    # Create enhanced template features
    template_features = np.concatenate([
        [num_nodes, num_edges, total_ops, len(all_op_types) / max(num_nodes, 1)],
        [max_depth, avg_depth, max_fanout, avg_fanout],
        [graph_density, connectedness],
        graph_embedding
    ])
    
    # Use both standard and robust scaling for better handling of different feature distributions
    scaler_template = StandardScaler()
    template_features = scaler_template.fit_transform(template_features.reshape(1, -1)).flatten()
    template_features = np.nan_to_num(template_features, nan=0.0)
    
    # Schedule-Specific Features with enhanced engineering
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
        scheduling_features.append(sched_feature)
    
    # Enhanced Scheduling Sequence with more sophisticated feature engineering
    scheduling_sequence = []
    for i, sf in enumerate(scheduling_features):
        # Extract base metrics
        sched_vector = [float(sf.get(metric, 0.0)) for metric in important_metrics]
        
        # Calculate derived metrics
        bytes_prod = sf.get('bytes_at_production', 0.0)
        bytes_real = sf.get('bytes_at_realization', 0.0)
        points_total = sf.get('points_computed_total', 0.0)
        inner_p = sf.get('inner_parallelism', 0.0)
        outer_p = sf.get('outer_parallelism', 0.0)
        num_vectors = sf.get('num_vectors', 0.0)
        working_set = sf.get('working_set', 0.0)
        
        # Add derived metrics
        derived_values = [
            bytes_prod / max(points_total, 1e-4),  # bytes_per_point
            inner_p * outer_p,                     # parallelism_product
            bytes_prod / max(num_vectors, 1e-4),   # bytes_density
            points_total / max(working_set, 1e-4)  # compute_efficiency
        ]
        
        # Add logarithmic transformations for better numerical stability
        log_transforms = [
            np.log1p(inner_p * outer_p),
            np.log1p(bytes_prod),
            np.log1p(points_total),
            np.log1p(working_set)
        ]
        
        # Add ratio features
        ratio_features = [
            bytes_prod / max(bytes_real, 1e-4),
            inner_p / max(outer_p, 1e-4),
            bytes_prod / max(working_set, 1e-4)
        ]
        
        # Combine all features
        extended_vector = np.concatenate([
            sched_vector,
            derived_values,
            log_transforms,
            ratio_features
        ])
        
        # Apply more sophisticated data augmentation for regularization
        if i % 2 == 0:  # Only augment half the samples to maintain some clean data
            noise_scale = 0.03  # Reduced noise for better stability
            noise = np.random.normal(0, noise_scale, len(extended_vector))
            extended_vector = np.array(extended_vector, dtype=np.float32) + noise
        
        # Concatenate with template features
        combined_vector = np.concatenate([template_features, extended_vector])
        scheduling_sequence.append(combined_vector)
    
    # Handle empty sequences
    if not scheduling_sequence:
        empty_feature_size = len(template_features) + len(important_metrics) + len(derived_metrics) + 7  # +7 for transforms and ratios
        scheduling_sequence = [np.concatenate([template_features, np.zeros(empty_feature_size - len(template_features), dtype=np.float32)])]
    
    # Normalize sequence features
    seq_array = np.array(scheduling_sequence)
    scaler_seq = RobustScaler()
    scheduling_sequence = scaler_seq.fit_transform(seq_array)
    scheduling_sequence = np.nan_to_num(scheduling_sequence, nan=0.0).tolist()
    
    return {
        'scheduling_sequence': scheduling_sequence,
        'execution_time': execution_time,
        'graph_stats': {  # Store these for potential analysis
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'max_depth': max_depth,
            'graph_density': graph_density
        }
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
    
    # Use better stratification to ensure representative test set
    combined = list(zip(all_features, all_file_names))
    
    # Sort by execution time to ensure a range of values in both train and test sets
    combined.sort(key=lambda x: x[0]['execution_time'])
    
    # Take every nth item for test set (stratified sampling)
    test_size = 50
    test_indices = np.linspace(0, len(combined) - 1, test_size, dtype=int)
    test_set = [combined[i] for i in test_indices]
    train_set = [combined[i] for i in range(len(combined)) if i not in test_indices]
    
    random.shuffle(train_set)  # Shuffle training data
    
    train_features, train_file_names = zip(*train_set)
    test_features, test_file_names = zip(*test_set)
    
    print(f"Total files: {total_files}")
    print(f"Training files: {len(train_features)}")
    print(f"Testing files: {len(test_features)}")
    
    return list(train_features), list(test_features), list(test_file_names)

def prepare_data_for_model(train_features, test_features):
    # Convert sequences to tensors with more sophisticated padding
    max_seq_len = max(
        max([len(f['scheduling_sequence']) for f in train_features]),
        max([len(f['scheduling_sequence']) for f in test_features])
    )
    
    # Create padded sequences with attention masks
    train_sequences = []
    train_masks = []
    for f in train_features:
        seq = torch.FloatTensor(f['scheduling_sequence'])
        seq_len = seq.shape[0]
        
        # Create padding to match max length
        if seq_len < max_seq_len:
            padding = torch.zeros((max_seq_len - seq_len, seq.shape[1]))
            seq_padded = torch.cat([seq, padding], dim=0)
        else:
            seq_padded = seq[:max_seq_len]  # Truncate if too long
            
        train_sequences.append(seq_padded)
        
        # Create attention mask (1 for real data, 0 for padding)
        mask = torch.ones(max_seq_len)
        if seq_len < max_seq_len:
            mask[seq_len:] = 0
        train_masks.append(mask)
    
    test_sequences = []
    test_masks = []
    for f in test_features:
        seq = torch.FloatTensor(f['scheduling_sequence'])
        seq_len = seq.shape[0]
        
        if seq_len < max_seq_len:
            padding = torch.zeros((max_seq_len - seq_len, seq.shape[1]))
            seq_padded = torch.cat([seq, padding], dim=0)
        else:
            seq_padded = seq[:max_seq_len]
            
        test_sequences.append(seq_padded)
        
        mask = torch.ones(max_seq_len)
        if seq_len < max_seq_len:
            mask[seq_len:] = 0
        test_masks.append(mask)
    
    # Stack sequences and masks
    train_sequences_padded = torch.stack(train_sequences)
    test_sequences_padded = torch.stack(test_sequences)
    train_masks = torch.stack(train_masks)
    test_masks = torch.stack(test_masks)
    
    # Prepare target values with better outlier handling
    y_train_raw = np.array([f['execution_time'] for f in train_features])
    y_test_raw = np.array([f['execution_time'] for f in test_features])
    
    # Handle outliers using winsorization (clip at 95th percentile)
    train_percentile_95 = np.percentile(y_train_raw, 95)
    y_train_raw = np.clip(y_train_raw, 0, train_percentile_95)
    y_test_raw = np.clip(y_test_raw, 0, train_percentile_95)
    
    # Apply log transformation to handle skewed distribution
    y_train = np.log1p(y_train_raw).reshape(-1, 1)
    y_test = np.log1p(y_test_raw).reshape(-1, 1)
    
    # Use robust scaler for better handling of remaining outliers
    scaler_y = RobustScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Handle potential NaNs or infinities
    y_train_scaled = np.nan_to_num(y_train_scaled, nan=0.0)
    y_test_scaled = np.nan_to_num(y_test_scaled, nan=0.0)
    
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    input_size = train_sequences_padded.shape[2]
    print(f"Sequence input size: {input_size}")
    
    return (train_sequences_padded, train_masks, y_train_tensor,
            test_sequences_padded, test_masks, y_test_tensor,
            scaler_y, input_size)

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden_size, hidden_size)
        
    def transpose_for_scores(self, x):
        batch_size, seq_len, _ = x.size()
        new_x_shape = (batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.size()
        
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Take the dot product between "query" and "key" to get attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        
        # Apply attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            attention_scores = attention_scores + (1.0 - attention_mask) * -10000.0
        
        # Normalize the attention scores
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = (batch_size, seq_len, self.all_head_size)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        output = self.out(context_layer)
        return output

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, x, mask=None):
        # Calculate attention scores
        attention_scores = self.attention(x)
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1)  # Add feature dimension
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        weights = torch.softmax(attention_scores, dim=1)
        
        # Apply weighted sum
        return torch.sum(x * weights, dim=1)

class EnhancedTransformerModel(nn.Module):
    def __init__(self, seq_input_size, hidden_size=256, num_layers=3, num_heads=8, dropout_rate=0.2, output_size=1):
        super(EnhancedTransformerModel, self).__init__()
        
        # Input projection
        self.input_projection = nn.Linear(seq_input_size, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': SelfAttention(hidden_size, num_heads, dropout_rate),
                'feedforward': nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_size * 4, hidden_size)
                ),
                'attention_norm': nn.LayerNorm(hidden_size),
                'feedforward_norm': nn.LayerNorm(hidden_size)
            }) for _ in range(num_layers)
        ])
        
        # Global attention pooling
        self.global_attention = AttentionPooling(hidden_size)
        
        # Output layers with residual connections
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn2 = nn.BatchNorm1d(hidden_size // 4)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Final projection for regression
        self.output_layer = nn.Linear(hidden_size // 4, output_size)
        
        # Activation functions
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, seq_input, attention_mask=None):
        # Input projection
        x = self.input_projection(seq_input)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            # Self-attention with residual connection and layer norm
            residual = x
            attention_output = layer['attention'](x, attention_mask)
            x = layer['attention_norm'](residual + self.dropout(attention_output))
            
            # Feedforward with residual connection and layer norm
            residual = x
            feedforward_output = layer['feedforward'](x)
            x = layer['feedforward_norm'](residual + self.dropout(feedforward_output))
        
        # Pool sequence to a single vector using attention
        x = self.global_attention(x, attention_mask)
        
        # Output layers
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.gelu(x)
        x = self.dropout2(x)
        
        output = self.output_layer(x)
        return output

# Define improved loss function that handles outliers better
def adaptive_loss(outputs, targets, alpha=0.25, gamma=2.0, beta=0.1):
    # Combine focal loss with Huber loss for robustness
    squared_error = (outputs - targets) ** 2
    abs_error = torch.abs(outputs - targets)
    
    # Huber loss component (less sensitive to outliers)
    huber_loss = torch.where(abs_error < beta, 
                            0.5 * squared_error, 
                            beta * abs_error - 0.5 * beta**2)
    
    # Focal loss component (focus on harder examples)
    pt = torch.exp(-abs_error)
    focal_component = alpha * (1 - pt) ** gamma
    
    # Combined loss
    loss = focal_component * huber_loss
    return torch.mean(loss)

def create_data_loaders(train_sequences, train_masks, y_train, test_sequences, test_masks, y_test, batch_size=16):
    train_dataset = TensorDataset(train_sequences, train_masks, y_train)
    test_dataset = TensorDataset(test_sequences, test_masks, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=500, patience=50, accumulation_steps=2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        model.to(device)
    except RuntimeError as e:
        print(f"Error moving model to CUDA: {e}. Falling back to CPU.")
        device = torch.device('cpu')
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
        
        for i, (seq_inputs, masks, targets) in enumerate(train_loader):
            seq_inputs, masks, targets = seq_inputs.to(device), masks.to(device), targets.to(device)
            outputs = model(seq_inputs, masks)
            loss = criterion(outputs, targets)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss detected at epoch {epoch+1}, batch {i+1}")
                return None, None
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps * seq_inputs.size(0)
        
        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq_inputs, masks, targets in test_loader:
                seq_inputs, masks, targets = seq_inputs.to(device), masks.to(device), targets.to(device)
                outputs = model(seq_inputs, masks)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * seq_inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
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
    plt.show()
    
    return train_losses, val_losses

def evaluate_model(model, X_test_seq, X_test_masks, y_test, y_scaler, file_names_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    X_test_seq = X_test_seq.to(device)
    X_test_masks = X_test_masks.to(device)
    with torch.no_grad():
        y_pred_scaled = model(X_test_seq, X_test_masks)
    
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
    
    mse = mean_squared_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    mape = mean_absolute_percentage_error(y_test_actual, y_pred_actual) * 100
    
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
    print(f"Total test samples: {len(test_features)} (50 selected)")
    
    if len(train_features) == 0 or len(test_features) == 0:
        print("Error: No valid training or test data found")
        return None
    
    (train_sequences, train_masks, y_train,
     test_sequences, test_masks, y_test,
     y_scaler, seq_input_size) = prepare_data_for_model(train_features, test_features)
    
    train_loader, test_loader = create_data_loaders(
        train_sequences, train_masks, y_train,
        test_sequences, test_masks, y_test,
        batch_size=16
    )
    
    global model
    model = EnhancedTransformerModel(
        seq_input_size=seq_input_size,
        hidden_size=256,
        num_layers=3,
        num_heads=8,
        dropout_rate=0.2,
        output_size=1
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    print("Building and training Enhanced Transformer model...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader,
        adaptive_loss, optimizer, scheduler,
        num_epochs=500, patience=50, accumulation_steps=2
    )
    
    if train_losses is None or val_losses is None:
        print("Training failed due to invalid values")
        return None
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(
        model, test_sequences, test_masks, y_test,
        y_scaler, test_file_names
    )
    
    print(f"\nSummary for Comparison:")
    print(f"Model: EnhancedTransformer")
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "synthetic_data"
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    model, y_scaler, y_test_actual, y_pred_actual = main(main_dir)
