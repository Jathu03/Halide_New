import os
import json
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA as SKPCA
from sklearn.feature_selection import VarianceThreshold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import matplotlib.pyplot as plt

# Define important metrics for scheduling sequence
important_metrics = [
    'bytes_at_production', 'bytes_at_realization', 'bytes_at_root', 'bytes_at_task',
    'inner_parallelism', 'outer_parallelism', 'num_productions', 'num_realizations',
    'num_scalars', 'num_vectors', 'points_computed_total', 'working_set'
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
        
        schedules = data.get("scheduling_data", [])
        for item in schedules:
            if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
                execution_time = item.get('value')
                if execution_time is not None and execution_time > 0:
                    return float(execution_time)
        
        print(f"Warning: 'total_execution_time_ms' not found in {file_path}")
        last_value = schedules[-1]["value"]
        return float(last_value) if last_value > 0 else None
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def extract_graph_features(nodes_features, edges_features):
    G = nx.DiGraph()
    for node in nodes_features:
        G.add_node(node['Name'])
    for edge in edges_features:
        G.add_edge(edge['From'], edge['To'], name=edge['Name'])
    
    # Compute graph-based features
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    clustering_coeff = nx.clustering(G)
    
    avg_degree_centrality = np.mean(list(degree_centrality.values())) if degree_centrality else 0.0
    avg_betweenness = np.mean(list(betweenness_centrality.values())) if betweenness_centrality else 0.0
    avg_clustering = np.mean(list(clustering_coeff.values())) if clustering_coeff else 0.0
    
    return {
        'avg_degree_centrality': avg_degree_centrality,
        'avg_betweenness_centrality': avg_betweenness,
        'avg_clustering_coeff': avg_clustering,
        'graph_density': nx.density(G) if G.number_of_nodes() > 0 else 0.0,
        'num_connected_components': nx.number_strongly_connected_components(G)
    }

def extract_features_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    execution_time = get_execution_time(file_path)
    if execution_time is None or not np.isfinite(execution_time):
        print(f"Warning: Invalid execution time in {file_path}")
        return None
    
    # IQR-based outlier clipping
    q1, q3 = np.percentile([execution_time], [25, 75])
    iqr = q3 - q1
    lower_bound = max(1.0, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr
    execution_time = np.clip(execution_time, lower_bound, upper_bound)
    
    nodes_features = []
    edges_features = []
    programming_details = data.get("programming_details", None)
    
    if programming_details:
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
    
    scheduling_features = []
    scheduling_data = data.get("scheduling_data", None)
    if not scheduling_data and programming_details and 'Schedules' in programming_details:
        scheduling_data = programming_details['Schedules']
    
    if scheduling_data:
        for sched in scheduling_data:
            sched_feature = {'Name': sched.get('Name', '')}
            if 'Details' in sched and 'scheduling_feature' in sched['Details']:
                sf = sched['Details']['scheduling_feature']
                sched_feature.update(sf)
            scheduling_features.append(sched_feature)
    
    # Enhanced scheduling sequence (without PCA for now)
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
        seq_vector.append(np.log1p(abs(bytes_prod)) / np.log1p(max(abs(bytes_real), 1e-4)) if bytes_real != 0 else 0.0)
        seq_vector.append(np.log1p(bytes_prod) / np.log1p(max(num_vec, 1e-4)) if num_vec != 0 else 0.0)
        seq_vector.append(np.log1p(points_total) / np.log1p(max(num_vec, 1e-4)) if num_vec != 0 else 0.0)
        seq_vector.append(np.log1p(working_set) / np.log1p(max(bytes_prod, 1e-4)) if bytes_prod != 0 else 0.0)
        seq_vector.append(np.log1p(inner_p * outer_p))
        seq_vector.append(np.log1p(bytes_prod) / np.log1p(max(points_total, 1e-4)) if points_total != 0 else 0.0)
        seq_vector.append(np.log1p(working_set) / np.log1p(max(num_vec, 1e-4)) if num_vec != 0 else 0.0)
        seq_vector.append(inner_p / max(outer_p, 1e-4) if outer_p != 0 else 0.0)
        seq_vector.append(np.log1p(bytes_real) / np.log1p(max(working_set, 1e-4)) if working_set != 0 else 0.0)
        scheduling_sequence.append(seq_vector)
    if not scheduling_sequence:
        scheduling_sequence = [[0.0] * (len(important_metrics) + 9)]
    
    seq_array = np.array(scheduling_sequence)
    # Adjust n_quantiles to be at most the number of samples
    n_samples = seq_array.shape[0]
    scaler_seq = QuantileTransformer(output_distribution='normal', n_quantiles=min(1000, n_samples))
    scheduling_sequence = scaler_seq.fit_transform(seq_array)
    scheduling_sequence = np.nan_to_num(scheduling_sequence, nan=0.0).tolist()
    
    op_counts = {}
    for node in nodes_features:
        for key, value in node.items():
            if key.startswith('op_'):
                op_counts[key] = op_counts.get(key, 0) + value
    
    total_ops = sum(op_counts.values())
    num_nodes = max(len(nodes_features), 1)
    num_edges = len(edges_features)
    total_bytes = sum(sf.get('bytes_at_production', 0) for sf in scheduling_features)
    total_vectors = sum(sf.get('num_vectors', 0) for sf in scheduling_features)
    
    graph_features = extract_graph_features(nodes_features, edges_features)
    
    scalar_features = {
        'nodes_count': num_nodes,
        'edges_count': num_edges,
        'node_edge_ratio': num_nodes / max(num_edges, 1),
        'total_ops': total_ops,
        'op_diversity': len(op_counts) / num_nodes,
        'avg_ops_per_node': total_ops / num_nodes,
        'edge_density': num_edges / max(num_nodes * (num_nodes - 1), 1),
        'total_parallelism': sum(sf.get('inner_parallelism', 0) * sf.get('outer_parallelism', 1) for sf in scheduling_features),
        'avg_bytes_per_node': total_bytes / num_nodes,
        'vector_op_ratio': op_counts.get('op_vector', 0) / max(total_ops, 1),
        'bytes_per_vector': total_bytes / max(total_vectors, 1e-4),
        'ops_per_byte': total_ops / max(total_bytes, 1e-4),
        'parallelism_ratio': sum(sf.get('inner_parallelism', 0) / max(sf.get('outer_parallelism', 1e-4), 1) for sf in scheduling_features)
    }
    scalar_features.update(op_counts)
    scalar_features.update(graph_features)
    
    for key in scalar_features:
        if not np.isfinite(scalar_features[key]):
            scalar_features[key] = 0.0
    
    return {
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

def mixup_data(seq_inputs, scalar_inputs, targets, alpha=0.2):
    batch_size = seq_inputs.size(0)
    indices = torch.randperm(batch_size)
    
    lam = np.random.beta(alpha, alpha)
    seq_inputs_mixed = lam * seq_inputs + (1 - lam) * seq_inputs[indices]
    scalar_inputs_mixed = lam * scalar_inputs + (1 - lam) * scalar_inputs[indices]
    targets_mixed = lam * targets + (1 - lam) * targets[indices]
    
    return seq_inputs_mixed, scalar_inputs_mixed, targets_mixed

def prepare_data_for_model(train_features, test_features):
    # Collect all scheduling sequences
    train_sequences_raw = [np.array(f['scheduling_sequence']) for f in train_features]
    test_sequences_raw = [np.array(f['scheduling_sequence']) for f in test_features]
    
    # Pad sequences to the same length for PCA
    max_seq_length = max(max(len(seq) for seq in train_sequences_raw), max(len(seq) for seq in test_sequences_raw))
    feature_dim = train_sequences_raw[0].shape[1]  # All sequences should have the same feature dimension at this point
    
    # Pad sequences with zeros to the maximum length
    train_sequences_padded = []
    test_sequences_padded = []
    for seq in train_sequences_raw:
        padded_seq = np.zeros((max_seq_length, feature_dim))
        padded_seq[:len(seq), :] = seq
        train_sequences_padded.append(padded_seq)
    for seq in test_sequences_raw:
        padded_seq = np.zeros((max_seq_length, feature_dim))
        padded_seq[:len(seq), :] = seq
        test_sequences_padded.append(padded_seq)
    
    # Convert to numpy arrays
    train_sequences_padded = np.array(train_sequences_padded)  # Shape: (n_samples, max_seq_length, feature_dim)
    test_sequences_padded = np.array(test_sequences_padded)
    
    # Reshape for PCA: (n_samples * max_seq_length, feature_dim)
    train_reshaped = train_sequences_padded.reshape(-1, feature_dim)
    test_reshaped = test_sequences_padded.reshape(-1, feature_dim)
    
    # Apply PCA to reduce dimensionality
    desired_components = 10
    n_samples = train_reshaped.shape[0]
    n_features = train_reshaped.shape[1]
    n_components = min(desired_components, n_features, n_samples)
    if n_components > 0:
        pca = SKPCA(n_components=n_components)
        train_transformed = pca.fit_transform(train_reshaped)
        test_transformed = pca.transform(test_reshaped)
        print(f"Applied PCA with {n_components} components (original features: {n_features})")
    else:
        print("Warning: Skipping PCA as n_components would be 0")
        train_transformed = train_reshaped
        test_transformed = test_reshaped
        n_components = n_features
    
    # Reshape back to (n_samples, max_seq_length, n_components)
    train_transformed = train_transformed.reshape(len(train_features), max_seq_length, n_components)
    test_transformed = test_transformed.reshape(len(test_features), max_seq_length, n_components)
    
    # Convert to tensors
    train_sequences = [torch.FloatTensor(seq) for seq in train_transformed]
    test_sequences = [torch.FloatTensor(seq) for seq in test_transformed]
    
    # Pad sequences (should now have consistent feature dimensions)
    train_sequences_padded = pad_sequence(train_sequences, batch_first=True)
    test_sequences_padded = pad_sequence(test_sequences, batch_first=True)
    
    train_scalar_df = pd.DataFrame([f['scalar_features'] for f in train_features])
    test_scalar_df = pd.DataFrame([f['scalar_features'] for f in test_features])
    
    # Feature selection
    train_scalar_df = train_scalar_df.fillna(0)
    test_scalar_df = test_scalar_df.fillna(0)
    
    # Remove low-variance and highly correlated features
    selector = VarianceThreshold(threshold=0.01)
    train_scalar_df = pd.DataFrame(selector.fit_transform(train_scalar_df), columns=train_scalar_df.columns[selector.get_support()])
    test_scalar_df = pd.DataFrame(selector.transform(test_scalar_df), columns=test_scalar_df.columns[selector.get_support()])
    
    corr_matrix = train_scalar_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.85)]
    train_scalar_df = train_scalar_df.drop(columns=to_drop)
    test_scalar_df = test_scalar_df.drop(columns=to_drop)
    
    y_train_raw = np.array([f['execution_time'] for f in train_features])
    y_test_raw = np.array([f['execution_time'] for f in test_features])
    
    # IQR-based clipping
    q1, q3 = np.percentile(y_train_raw, [25, 75])
    iqr = q3 - q1
    lower_bound = max(1.0, q1 - 1.5 * iqr)
    upper_bound = q3 + 1.5 * iqr
    y_train_raw = np.clip(y_train_raw, lower_bound, upper_bound)
    y_test_raw = np.clip(y_test_raw, lower_bound, upper_bound)
    
    # Label smoothing
    smoothing_factor = 0.03
    y_train_raw = (1 - smoothing_factor) * y_train_raw + smoothing_factor * np.mean(y_train_raw)
    y_test_raw = (1 - smoothing_factor) * y_test_raw + smoothing_factor * np.mean(y_test_raw)
    
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
    
    train_scalar_tensor = torch.FloatTensor(train_scalar_scaled)
    test_scalar_tensor = torch.FloatTensor(test_scalar_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Sequence input size: {train_sequences_padded.shape[2]}")
    print(f"Scalar input size: {train_scalar_tensor.shape[1]}")
    
    return (train_sequences_padded, train_scalar_tensor, y_train_tensor,
            test_sequences_padded, test_scalar_tensor, y_test_tensor,
            scaler_y, train_sequences_padded.shape[2], train_scalar_tensor.shape[1])

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.1, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, h, adj):
        Wh = self.W(h)  # [batch_size, N, out_features]
        batch_size, N = Wh.size(0), Wh.size(1)
        
        Wh1 = Wh.unsqueeze(2).repeat(1, 1, N, 1)  # [batch_size, N, N, out_features]
        Wh2 = Wh.unsqueeze(1).repeat(1, N, 1, 1)  # [batch_size, N, N, out_features]
        e_input = torch.cat([Wh1, Wh2], dim=-1)  # [batch_size, N, N, 2 * out_features]
        
        e = self.leakyrelu(self.a(e_input).squeeze(-1))  # [batch_size, N, N]
        attention = torch.softmax(e, dim=-1)
        attention = self.dropout(attention)
        
        h_prime = torch.matmul(attention, Wh)  # [batch_size, N, out_features]
        return torch.tanh(h_prime)

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

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout_rate)
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class HighlyOptimizedLSTMModel(nn.Module):
    def __init__(self, seq_input_size, scalar_input_size, hidden_sizes=[1024, 768, 512], output_size=1, dropout_rate=0.15, num_heads=8):
        super(HighlyOptimizedLSTMModel, self).__init__()
        
        self.lstm_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        self.lstm_layers.append(nn.LSTM(seq_input_size, hidden_sizes[0], num_layers=3, batch_first=True, bidirectional=True))
        self.ln_layers.append(nn.LayerNorm(hidden_sizes[0] * 2))
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(nn.LSTM(hidden_sizes[i-1] * 2, hidden_sizes[i], num_layers=3, batch_first=True, bidirectional=True))
            self.ln_layers.append(nn.LayerNorm(hidden_sizes[i] * 2))
        
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_sizes[-1] * 2, num_heads, dropout_rate)
            for _ in range(3)  # Deeper Transformer stack
        ])
        
        # Graph Attention for scalar features (simplified as a feature aggregator)
        self.gat = GraphAttentionLayer(scalar_input_size, 256, dropout_rate)
        
        combined_size = hidden_sizes[-1] * 2 + 256
        self.fc1 = nn.Linear(combined_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.ln1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.ln3 = nn.LayerNorm(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.ln4 = nn.LayerNorm(64)
        self.output_layer = nn.Linear(64, output_size)
        
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.residual_proj1 = nn.Linear(combined_size, 512)
        self.residual_proj2 = nn.Linear(512, 256)
        self.residual_proj3 = nn.Linear(256, 128)
        self.residual_proj4 = nn.Linear(128, 64)
    
    def forward(self, seq_input, scalar_input):
        lstm_out = seq_input
        for lstm, ln in zip(self.lstm_layers, self.ln_layers):
            lstm_out, _ = lstm(lstm_out)
            lstm_out = ln(lstm_out)
            lstm_out = self.dropout(lstm_out)
        
        transformer_out = lstm_out
        for transformer in self.transformer_layers:
            transformer_out = transformer(transformer_out)
        context = transformer_out.mean(dim=1)
        
        # Simulate adjacency matrix for GAT (simplified as self-attention on scalar features)
        batch_size = scalar_input.size(0)
        scalar_input_exp = scalar_input.unsqueeze(1)  # [batch_size, 1, scalar_input_size]
        adj = torch.ones(batch_size, 1, 1).to(scalar_input.device)
        scalar_out = self.gat(scalar_input_exp, adj).squeeze(1)
        
        combined = torch.cat((context, scalar_out), dim=1)
        
        x = self.fc1(combined)
        x = x + self.residual_proj1(combined)
        x = self.bn1(x)
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        
        x2 = self.fc2(x)
        x2 = x2 + self.residual_proj2(x)
        x2 = self.bn2(x2)
        x2 = self.ln2(x2)
        x2 = self.gelu(x2)
        x2 = self.dropout(x2)
        
        x3 = self.fc3(x2)
        x3 = x3 + self.residual_proj3(x2)
        x3 = self.bn3(x3)
        x3 = self.ln3(x3)
        x3 = self.gelu(x3)
        x3 = self.dropout(x3)
        
        x4 = self.fc4(x3)
        x4 = x4 + self.residual_proj4(x3)
        x4 = self.bn4(x4)
        x4 = self.ln4(x4)
        x4 = self.gelu(x4)
        
        output = self.output_layer(x4)
        return output

def focal_loss(outputs, targets, gamma=2.0, alpha=0.25):
    epsilon = 1e-7
    outputs = outputs.clamp(epsilon, 1.0 - epsilon)
    pt = torch.where(targets >= 0, outputs, 1 - outputs)
    loss = -alpha * (1 - pt) ** gamma * torch.log(pt)
    return loss.mean()

def custom_loss(outputs, targets, huber_delta=0.3, mape_weight=0.6, focal_weight=0.2, smooth_weight=0.1, l1_lambda=1e-6):
    huber = nn.HuberLoss(delta=huber_delta)(outputs, targets)
    mape = torch.mean(torch.abs((targets - outputs) / (targets + 1e-2))) * 100
    focal = focal_loss(outputs, targets)
    # Smoothness penalty: penalize large differences between consecutive predictions
    smooth_loss = torch.mean(torch.abs(outputs[1:] - outputs[:-1])) if outputs.size(0) > 1 else torch.tensor(0.0).to(outputs.device)
    l1_reg = sum(param.abs().sum() for param in model.parameters()) * l1_lambda
    return huber + mape_weight * mape + focal_weight * focal + smooth_weight * smooth_loss + l1_reg

def create_data_loaders(train_sequences, train_scalar, y_train, test_sequences, test_scalar, y_test, batch_size=128):
    train_dataset = TensorDataset(train_sequences, train_scalar, y_train)
    test_dataset = TensorDataset(test_sequences, test_scalar, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, epochs_no_improve, train_losses, val_losses, checkpoint_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'epochs_no_improve': epochs_no_improve,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        epochs_no_improve = checkpoint['epochs_no_improve']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        print(f"Loaded checkpoint from {checkpoint_path}, resuming training from epoch {start_epoch}")
        return start_epoch, best_val_loss, epochs_no_improve, train_losses, val_losses
    else:
        print("No checkpoint found, starting training from scratch")
        return 0, float('inf'), 0, [], []

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=1000, patience=100, accumulation_steps=1, checkpoint_path='model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Load checkpoint if it exists
    start_epoch, best_val_loss, epochs_no_improve, train_losses, val_losses = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
    
    warm_up_epochs = 100
    total_steps = num_epochs * len(train_loader)
    warm_up_steps = warm_up_epochs * len(train_loader)
    current_step = start_epoch * len(train_loader)
    
    best_model_state = model.state_dict().copy() if start_epoch == 0 else checkpoint['model_state_dict']
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for i, (seq_inputs, scalar_inputs, targets) in enumerate(train_loader):
            seq_inputs, scalar_inputs, targets = seq_inputs.to(device), scalar_inputs.to(device), targets.to(device)
            
            # Apply Mixup
            if random.random() < 0.5:  # Apply Mixup with 50% probability
                seq_inputs, scalar_inputs, targets_mixed = mixup_data(seq_inputs, scalar_inputs, targets, alpha=0.2)
            else:
                targets_mixed = targets
            
            outputs = model(seq_inputs, scalar_inputs)
            loss = criterion(outputs, targets_mixed)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss at epoch {epoch+1}, batch {i+1}")
                return None, None
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps * seq_inputs.size(0)
            
            current_step += 1
            if current_step > warm_up_steps:
                scheduler.step()
            else:
                lr = 5e-6 + (5e-5 - 5e-6) * (current_step / warm_up_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
        
        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
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
                loss = criterion(outputs, targets)
                val_loss += loss.item() * seq_inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save checkpoint after each epoch
        save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, epochs_no_improve, train_losses, val_losses, checkpoint_path)
        
        if val_loss < best_val_loss and not np.isnan(val_loss) and not np.isinf(val_loss):
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            # Save best checkpoint
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, epochs_no_improve, train_losses, val_losses, 'model_best.pth')
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
    plt.savefig('loss.png')
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
        actual = y_test_actual[i][0]
        error_percentage = abs(actual - pred) / actual * 100 if actual > 0 else 0
        error_percentage = min(error_percentage, 300.0)  # Cap extreme errors
        
        results_by_subfolder[subfolder].append({
            'file': file_path,
            'actual': actual,
            'predicted': pred,
            'error_percentage': error_percentage
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
    mask = y_test_actual > 1.0
    mape = np.mean(np.abs((y_test_actual[mask] - y_pred_actual[mask]) / y_test_actual[mask])) * 100 if mask.sum() > 0 else 0.0
    
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
    
    (train_sequences, train_scalar, y_train,
     test_sequences, test_scalar, y_test,
     y_scaler, seq_input_size, scalar_input_size) = prepare_data_for_model(train_features, test_features)
    
    train_loader, test_loader = create_data_loaders(
        train_sequences, train_scalar, y_train,
        test_sequences, test_scalar, y_test,
        batch_size=128
    )
    
    global model
    model = HighlyOptimizedLSTMModel(
        seq_input_size=seq_input_size,
        scalar_input_size=scalar_input_size,
        hidden_sizes=[1024, 768, 512],
        output_size=1,
        dropout_rate=0.15,
        num_heads=8
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    warm_up_epochs = 100
    total_steps = 1000 * len(train_loader)  # num_epochs * len(train_loader)
    warm_up_steps = warm_up_epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warm_up_steps, eta_min=5e-7)
    
    print("Training Highly Optimized LSTM model...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader,
        custom_loss, optimizer, scheduler,
        num_epochs=1000, patience=100, accumulation_steps=1, checkpoint_path='model.pth'
    )
    
    if train_losses is None or val_losses is None:
        print("Training failed due to invalid values")
        return None
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(
        model, test_sequences, test_scalar, y_test,
        y_scaler, test_file_names
    )
    
    print(f"\nSummary for Comparison:")
    print(f"Model: HighlyOptimizedLSTM")
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "synthetic_data"
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    model, y_scaler, y_test_actual, y_pred_actual = main(main_dir)
