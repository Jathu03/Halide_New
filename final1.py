import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader as GeometricDataLoader
from torch_geometric.nn import GCNConv
from torch.optim.lr_scheduler import OneCycleLR
import random
import matplotlib.pyplot as plt
import pickle
import time
import psutil
import logging
import shutil
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

set_seed(42)

def get_execution_time(file_path, max_retries=2, timeout=5):
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            with open(file_path, 'rb') as f:
                raw_content = f.read()
                content = raw_content.decode('utf-8', errors='replace').replace('\0', '')
                data = json.loads(content)
            
            if time.time() - start_time > timeout:
                logging.warning(f"Timeout reading {file_path}")
                return None
            
            if 'programming_details' not in data:
                logging.error(f"'programming_details' key not found in {file_path}")
                return None
            
            schedules = data.get("scheduling_data", [])
            for item in schedules:
                if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
                    execution_time = item.get('value')
                    if execution_time is not None:
                        logging.info(f"Extracted execution time for {file_path}: {execution_time} ms")
                        return float(execution_time)
            
            if schedules and isinstance(schedules[-1], dict) and "value" in schedules[-1]:
                execution_time = schedules[-1]["value"]
                logging.warning(f"'total_execution_time_ms' not found in {file_path}, using last schedule value: {execution_time} ms")
                return float(execution_time)
            
            logging.error(f"No valid execution time found in {file_path}")
            return None
        
        except FileNotFoundError:
            logging.error(f"File {file_path} not found")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON format in {file_path}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(0.1)
                continue
            return None
        except UnicodeDecodeError as e:
            logging.error(f"Encoding issue in {file_path}: {str(e)}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error in {file_path}: {str(e)}")
            return None

def extract_features_from_file(file_path, cache_dir='feature_cache', cache_version='v1'):
    cache_path = os.path.join(cache_dir, file_path.replace('/', '_').replace('.json', f'_v{cache_version}.pkl'))
    
    # Check cache
    if os.path.exists(cache_path) and os.path.getmtime(file_path) <= os.path.getmtime(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            if isinstance(cached_data, tuple) and len(cached_data) == 2:
                features, graph_data = cached_data
                if graph_data.x.size(0) > 0 and graph_data.y is not None:
                    logging.info(f"Loaded cached data for {file_path}: {len(features)} features, {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")
                    return features, graph_data
                else:
                    logging.warning(f"Invalid graph data in cache for {file_path}, reprocessing")
            else:
                logging.warning(f"Invalid cache format for {cache_path}, reprocessing")
        except Exception as e:
            logging.error(f"Error loading cache for {file_path}: {str(e)}, reprocessing")
    
    # Process JSON file
    start_time = time.time()
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"Error reading {file_path}: {str(e)}")
        return None
    
    execution_time = get_execution_time(file_path)
    if execution_time is None:
        return None
    
    nodes_features = []
    edges = []
    node_id_map = {}
    programming_details = data.get("programming_details", {})
    
    # Process nodes
    if 'Nodes' in programming_details:
        for idx, node in enumerate(programming_details['Nodes']):
            node_feature = {}
            if 'Details' in node and 'Op histogram' in node['Details']:
                op_hist = node['Details']['Op histogram']
                for op_line in op_hist:
                    parts = op_line.strip().split(':')
                    if len(parts) == 2:
                        op_name = parts[0].strip()
                        op_count = int(parts[1].strip())
                        node_feature[f'op_{op_name.lower()}'] = op_count
            nodes_features.append(node_feature)
            node_id_map[node.get('Name', f'node_{idx}')] = idx
    
    # Process edges
    if 'Edges' in programming_details:
        for edge in programming_details['Edges']:
            from_node = edge.get('From', '')
            to_node = edge.get('To', '')
            if from_node in node_id_map and to_node in node_id_map:
                edges.append([node_id_map[from_node], node_id_map[to_node]])
    
    # Process scheduling features
    scheduling_features = []
    scheduling_data = data.get("scheduling_data", programming_details.get('Schedules', []))
    for sched in scheduling_data:
        sched_feature = {}
        if 'Details' in sched and 'scheduling_feature' in sched['Details']:
            sf = sched['Details']['scheduling_feature']
            for key, value in sf.items():
                sched_feature[key] = value
        scheduling_features.append(sched_feature)
    
    # Aggregate features
    features = {
        'execution_time': execution_time,
        'nodes_count': len(nodes_features),
        'edges_count': len(edges),
        'scheduling_count': len(scheduling_features),
        'node_edge_ratio': len(nodes_features) / len(edges) if len(edges) > 0 else 0
    }
    
    op_counts = {}
    for node in nodes_features:
        for key, value in node.items():
            if key.startswith('op_'):
                op_counts[key] = op_counts.get(key, 0) + value
    features.update(op_counts)
    
    if scheduling_features and scheduling_features[0]:
        important_metrics = [
            'bytes_at_production', 'bytes_at_realization', 'bytes_at_root', 'bytes_at_task',
            'inner_parallelism', 'outer_parallelism', 'num_productions', 'num_realizations',
            'num_scalars', 'num_vectors', 'points_computed_total', 'working_set'
        ]
        for metric in important_metrics:
            if metric in scheduling_features[0]:
                features[f'sched_{metric}'] = scheduling_features[0][metric]
        
        total_bytes_at_production = sum(sf.get('bytes_at_production', 0) for sf in scheduling_features if isinstance(sf, dict))
        total_vectors = sum(sf.get('num_vectors', 0) for sf in scheduling_features if isinstance(sf, dict))
        total_parallelism = sum(sf.get('inner_parallelism', 0) * sf.get('outer_parallelism', 1) for sf in scheduling_features if isinstance(sf, dict))
        
        features['total_bytes_at_production'] = total_bytes_at_production
        features['total_vectors'] = total_vectors
        features['total_parallelism'] = total_parallelism
        features['bytes_per_vector'] = total_bytes_at_production / total_vectors if total_vectors > 0 else 0
        
        if 'working_set' in scheduling_features[0] and 'bytes_at_production' in scheduling_features[0]:
            features['memory_pressure'] = scheduling_features[0]['working_set'] / scheduling_features[0]['bytes_at_production'] if scheduling_features[0]['bytes_at_production'] > 0 else 0
    
    if len(nodes_features) > 0:
        op_types = sum(1 for k in op_counts.keys())
        features['avg_ops_per_node'] = sum(op_counts.values()) / len(nodes_features)
        features['op_diversity'] = op_types / len(nodes_features)
    
    # Optimize node feature extraction
    op_keys = sorted(set(k[3:] for k in op_counts.keys() if k.startswith('op_')))
    node_features_list = []
    for node in nodes_features:
        node_vec = [node.get(f'op_{op}', 0) for op in op_keys]
        if scheduling_features and scheduling_features[0]:
            for metric in important_metrics:
                if metric in scheduling_features[0]:
                    node_vec.append(scheduling_features[0][metric])
        node_features_list.append(node_vec)
    
    x = torch.tensor(node_features_list, dtype=torch.float) if node_features_list else torch.zeros((1, 1))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    y = torch.tensor([np.log1p(execution_time)], dtype=torch.float)
    
    graph_data = Data(x=x, edge_index=edge_index, y=y)
    
    # Cache features and graph data
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump((features, graph_data), f)
    
    logging.info(f"Processed {file_path}: {len(features)} features, {graph_data.num_nodes} nodes, {graph_data.num_edges} edges, Time: {time.time() - start_time:.3f}s")
    return features, graph_data

def process_directory(directory_path, cache_dir='feature_cache'):
    all_features = []
    all_graphs = []
    file_names = []
    
    json_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.json')])
    
    try:
        for filename in tqdm(json_files, desc=f"Processing {directory_path}", leave=False):
            file_path = os.path.join(directory_path, filename)
            result = extract_features_from_file(file_path, cache_dir)
            if result is not None:
                features, graph_data = result
                all_features.append(features)
                all_graphs.append(graph_data)
                file_names.append(filename)
            else:
                logging.warning(f"Skipping {file_path} due to processing error")
    except KeyboardInterrupt:
        logging.warning(f"Interrupted while processing {directory_path}. Saving partial progress...")
        return all_features, all_graphs, file_names
    return all_features, all_graphs, file_names

def process_main_directory(main_dir, cache_dir='feature_cache', val_size=0.2):
    # Clear cache directory
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logging.info(f"Cleared cache directory: {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)
    
    all_features = []
    all_graphs = []
    all_file_names = []
    
    subdirs = sorted([d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))])
    if len(subdirs) < 1:
        raise ValueError(f"Expected at least 1 subdirectory in {main_dir}, found {len(subdirs)}")
    
    try:
        for subdir in tqdm(subdirs, desc="Processing subdirectories"):
            subdir_path = os.path.join(main_dir, subdir)
            features, graphs, file_names = process_directory(subdir_path, cache_dir)
            if not features:
                logging.warning(f"Skipping {subdir} due to no valid data")
                continue
            all_features.extend(features)
            all_graphs.extend(graphs)
            all_file_names.extend([os.path.join(subdir, fname) for fname in file_names])
            logging.info(f"Processed subdir {subdir}: {len(features)} files, {len(graphs)} graphs")
    except KeyboardInterrupt:
        logging.warning("Interrupted during subdirectory processing. Saving partial progress...")
    
    total_files = len(all_features)
    if total_files < 50:
        raise ValueError(f"Expected at least 50 files total, found {total_files}")
    
    combined = list(zip(all_features, all_graphs, all_file_names))
    random.shuffle(combined)
    all_features, all_graphs, all_file_names = zip(*combined)
    
    test_size = 50
    train_val_features = all_features[:-test_size]
    test_features = all_features[-test_size:]
    train_val_graphs = all_graphs[:-test_size]
    test_graphs = all_graphs[-test_size:]
    train_val_file_names = all_file_names[:-test_size]
    test_file_names = all_file_names[-test_size:]
    
    train_size = int((1 - val_size) * len(train_val_features))
    train_features = train_val_features[:train_size]
    val_features = train_val_features[train_size:]
    train_graphs = train_val_graphs[:train_size]
    val_graphs = train_val_graphs[train_size:]
    train_file_names = train_val_file_names[:train_size]
    val_file_names = train_val_file_names[train_size:]
    
    logging.info(f"Total files: {total_files}")
    logging.info(f"Training files: {len(train_features)}, graphs: {len(train_graphs)}")
    logging.info(f"Validation files: {len(val_features)}, graphs: {len(val_graphs)}")
    logging.info(f"Testing files: {len(test_features)}, graphs: {len(test_graphs)}")
    
    return (train_features, val_features, test_features, 
            train_graphs, val_graphs, test_graphs, 
            list(train_file_names), list(val_file_names), list(test_file_names))

def clean_and_transform_features(train_features, val_features, test_features, train_graphs, val_graphs, test_graphs):
    all_features_df = pd.DataFrame(train_features + val_features + test_features)
    all_features_df = all_features_df.fillna(0)
    
    constant_columns = [col for col in all_features_df.columns 
                       if col != 'execution_time' and all_features_df[col].nunique() == 1]
    all_features_df = all_features_df.drop(columns=constant_columns)
    logging.info(f"Dropped {len(constant_columns)} constant columns")
    
    all_features_df['execution_time_log'] = np.log1p(all_features_df['execution_time'])
    if 'total_vectors' in all_features_df.columns and all_features_df['total_vectors'].max() > 0:
        all_features_df['bytes_per_vector'] = all_features_df['total_bytes_at_production'] / (all_features_df['total_vectors'] + 1e-8)
    
    # Feature selection based on correlation
    numeric_cols = all_features_df.select_dtypes(include=['number']).columns
    correlations = all_features_df[numeric_cols].corr()['execution_time_log'].abs()
    selected_features = correlations[correlations > 0.1].index.tolist()
    if 'execution_time' in selected_features:
        selected_features.remove('execution_time')
    if 'execution_time_log' in selected_features:
        selected_features.remove('execution_time_log')
    all_features_df = all_features_df[selected_features + ['execution_time', 'execution_time_log']]
    logging.info(f"Selected {len(selected_features)} features based on correlation > 0.1: {selected_features}")
    
    train_size = len(train_features)
    val_size = len(val_features)
    train_df = all_features_df.iloc[:train_size]
    val_df = all_features_df.iloc[train_size:train_size + val_size]
    test_df = all_features_df.iloc[train_size + val_size:]
    
    # Update graph node features with selected features
    for graph in train_graphs + val_graphs + test_graphs:
        if graph.x.size(0) > 0:
            node_features = graph.x.numpy()
            node_df = pd.DataFrame(node_features, columns=[f'feature_{i}' for i in range(node_features.shape[1])])
            selected_node_features = node_df.iloc[:, :len(selected_features)].values
            graph.x = torch.tensor(selected_node_features, dtype=torch.float)
    
    return train_df, val_df, test_df, train_graphs, val_graphs, test_graphs

def prepare_data_for_model(train_features, val_features, test_features, train_graphs, val_graphs, test_graphs):
    train_df, val_df, test_df, train_graphs, val_graphs, test_graphs = clean_and_transform_features(
        train_features, val_features, test_features, train_graphs, val_graphs, test_graphs
    )
    
    y_train = train_df['execution_time_log'].values.reshape(-1, 1)
    y_val = val_df['execution_time_log'].values.reshape(-1, 1)
    y_test = test_df['execution_time_log'].values.reshape(-1, 1)
    
    logging.info("\nDebugging target values in prepare_data_for_model:")
    logging.info(f"First 5 y_train raw: {y_train[:5].flatten()}")
    logging.info(f"First 5 y_val raw: {y_val[:5].flatten()}")
    logging.info(f"First 5 y_test raw: {y_test[:5].flatten()}")
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)
    
    logging.info(f"First 5 y_train scaled: {y_train_scaled[:5].flatten()}")
    logging.info(f"First 5 y_val scaled: {y_val_scaled[:5].flatten()}")
    logging.info(f"First 5 y_test scaled: {y_test_scaled[:5].flatten()}")
    
    for i, graph in enumerate(train_graphs):
        graph.y = torch.tensor(y_train_scaled[i], dtype=torch.float)
    for i, graph in enumerate(val_graphs):
        graph.y = torch.tensor(y_val_scaled[i], dtype=torch.float)
    for i, graph in enumerate(test_graphs):
        graph.y = torch.tensor(y_test_scaled[i], dtype=torch.float)
    
    logging.info(f"Graph node feature dimension: {train_graphs[0].x.shape[1] if train_graphs[0].x.size(0) > 0 else 0}")
    logging.info(f"Sample graph: nodes={train_graphs[0].num_nodes}, edges={train_graphs[0].num_edges}")
    
    return train_graphs, val_graphs, test_graphs, scaler_y

class GraphLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], output_size=1, dropout_rate=0.5, num_heads=4):
        super(GraphLSTMModel, self).__init__()
        
        self.hidden_sizes = hidden_sizes
        self.gcn_layers = nn.ModuleList()
        self.lstm_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        in_size = input_size
        for hidden_size in hidden_sizes:
            self.gcn_layers.append(GCNConv(in_size, hidden_size))
            self.lstm_layers.append(nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True))
            self.ln_layers.append(nn.LayerNorm(hidden_size * 2))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            in_size = hidden_size
        
        self.attention = MultiHeadAttention(hidden_sizes[-1] * 2, num_heads)
        
        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        fc_sizes = [hidden_sizes[-1] * 2, 128, 64, 32]
        for i in range(len(fc_sizes) - 1):
            self.fc_layers.append(nn.Linear(fc_sizes[i], fc_sizes[i+1]))
            self.bn_layers.append(nn.BatchNorm1d(fc_sizes[i+1]))
        
        self.output_layer = nn.Linear(fc_sizes[-1], output_size)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        device = x.device
        
        for gcn, lstm, ln, dropout in zip(self.gcn_layers, self.lstm_layers, self.ln_layers, self.dropout_layers):
            x = gcn(x, edge_index)
            x = self.relu(x)
            x = x.unsqueeze(1)  # Add time dimension for LSTM
            x, _ = lstm(x)
            x = ln(x)
            x = x.squeeze(1)
            x = dropout(x)
        
        # Global pooling
        from torch_geometric.nn import global_mean_pool
        x = global_mean_pool(x, batch)
        
        # Attention
        x = x.unsqueeze(1)
        x = self.attention(x, device)
        x = x.squeeze(1)
        
        # Fully connected layers
        for fc, bn in zip(self.fc_layers, self.bn_layers):
            x = fc(x)
            x = bn(x)
            x = self.leaky_relu(x)
        
        x = self.output_layer(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
    
    def forward(self, x, device):
        batch_size = x.shape[0]
        
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        self.scale = self.scale.to(device)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        attention = torch.softmax(energy, dim=-1)
        out = torch.matmul(attention, V).permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hidden_size)
        out = self.fc_out(out)
        return out

def create_data_loaders(train_graphs, val_graphs, test_graphs, batch_size=8):
    try:
        train_loader = GeometricDataLoader(
            train_graphs, batch_size=batch_size, shuffle=True, 
            num_workers=2, pin_memory=True
        )
        val_loader = GeometricDataLoader(
            val_graphs, batch_size=batch_size, shuffle=False, 
            num_workers=2, pin_memory=True
        )
        test_loader = GeometricDataLoader(
            test_graphs, batch_size=batch_size, shuffle=False, 
            num_workers=2, pin_memory=True
        )
    except RuntimeError as e:
        logging.warning(f"DataLoader failed with batch_size={batch_size}, trying batch_size=4: {str(e)}")
        batch_size = 4
        train_loader = GeometricDataLoader(
            train_graphs, batch_size=batch_size, shuffle=True, 
            num_workers=2, pin_memory=True
        )
        val_loader = GeometricDataLoader(
            val_graphs, batch_size=batch_size, shuffle=False, 
            num_workers=2, pin_memory=True
        )
        test_loader = GeometricDataLoader(
            test_graphs, batch_size=batch_size, shuffle=False, 
            num_workers=2, pin_memory=True
        )
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=400, patience=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    model.to(device)
    
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-4,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1000
    )
    
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    val_maes = []
    val_mapes = []
    train_mapes = []
    
    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        train_mape = 0.0
        train_count = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(data)
                loss = criterion(outputs, data.y.view(-1, 1))
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
            scheduler.step()
            running_loss += loss.item() * data.num_graphs
            train_mape += torch.abs((outputs - data.y.view(-1, 1)) / (data.y.view(-1, 1).abs() + 1e-8)).sum().item()
            train_count += data.num_graphs
        
        train_loss = running_loss / len(train_loader.dataset)
        train_mape = (train_mape / len(train_loader.dataset)) * 100
        train_losses.append(train_loss)
        train_mapes.append(train_mape)
        
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_mape = 0.0
        val_count = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    outputs = model(data)
                    loss = criterion(outputs, data.y.view(-1, 1))
                val_loss += loss.item() * data.num_graphs
                val_mae += torch.abs(outputs - data.y.view(-1, 1)).sum().item()
                val_mape += torch.abs((outputs - data.y.view(-1, 1)) / (data.y.view(-1, 1).abs() + 1e-8)).sum().item()
                val_count += data.num_graphs
        
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        val_mape = (val_mape / len(val_loader.dataset)) * 100
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        val_mapes.append(val_mape)
        
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        mem_usage = psutil.Process().memory_info().rss / (1024 ** 2)  # MB
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train MAPE: {train_mape:.2f}%, Val MAE: {val_mae:.4f}, Val MAPE: {val_mape:.2f}%, Grad Norm: {grad_norm:.4f}, Memory: {mem_usage:.2f} MB, Time: {time.time() - epoch_start_time:.2f}s')
        
        if epoch % 50 == 0:
            checkpoint_path = f'graph_checkpoint_epoch_{epoch}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Saved checkpoint: {checkpoint_path}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            logging.info(f'Early stopping after {epoch+1} epochs')
            model.load_state_dict(best_model_state)
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses, val_maes, val_mapes, train_mapes

def evaluate_model(model, test_loader, scaler_y, file_names_test, is_log_transformed=True, original_execution_times=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    y_pred_scaled = []
    y_test_scaled = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(data)
            y_pred_scaled.append(outputs.cpu().numpy())
            y_test_scaled.append(data.y.view(-1, 1).cpu().numpy())
    
    y_pred_scaled = np.concatenate(y_pred_scaled)
    y_test_scaled = np.concatenate(y_test_scaled)
    
    y_test_transformed = scaler_y.inverse_transform(y_test_scaled)
    y_pred_transformed = scaler_y.inverse_transform(y_pred_scaled)
    
    logging.info("\nDebugging transformed values before inverse log:")
    for i in range(min(5, len(y_test_transformed))):
        logging.info(f"Sample {i}: y_test_transformed={y_test_transformed[i][0]:.4f}, y_pred_transformed={y_pred_transformed[i][0]:.4f}")
    
    y_test_actual = np.expm1(y_test_transformed)
    y_pred_actual = np.expm1(y_pred_transformed)
    
    logging.info("\nDebugging final values after all transformations:")
    for i in range(min(5, len(y_test_actual))):
        logging.info(f"Sample {i}: y_test_actual={y_test_actual[i][0]:.4f}, y_pred_actual={y_pred_actual[i][0]:.4f}")
        if original_execution_times:
            logging.info(f"  Original execution time from JSON: {original_execution_times[file_names_test[i]]:.4f}")
    
    results_by_subfolder = {}
    for i, file_path in enumerate(file_names_test):
        subfolder = file_path.split('/')[0]
        if subfolder not in results_by_subfolder:
            results_by_subfolder[subfolder] = []
        
        actual_val = y_test_actual[i][0]
        pred_val = y_pred_actual[i][0]
        error_percentage = abs(actual_val - pred_val) / actual_val * 100 if actual_val > 0 else 0
        
        results_by_subfolder[subfolder].append({
            'file': file_path,
            'actual': actual_val,
            'predicted': pred_val,
            'error_percentage': error_percentage
        })
    
    for subfolder, results in results_by_subfolder.items():
        logging.info(f"\nResults for {subfolder}:")
        for result in results:
            logging.info(f"File: {result['file']}")
            logging.info(f"  Actual execution time: {result['actual']:.4f} ms")
            logging.info(f"  Predicted execution time: {result['predicted']:.4f} ms")
            logging.info(f"  Error percentage: {result['error_percentage']:.2f}%")
    
    mse = np.mean((y_test_actual - y_pred_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / (y_test_actual + 1e-8))) * 100
    
    logging.info("\nOverall Model Performance:")
    logging.info(f"MSE: {mse:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"MAPE: {mape:.2f}%")
    
    # Plot error histogram
    errors = np.abs(y_test_actual - y_pred_actual).flatten()
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, density=True, alpha=0.75)
    plt.title('Histogram of Absolute Prediction Errors')
    plt.xlabel('Absolute Error (ms)')
    plt.ylabel('Density')
    plt.grid(True)
    plt.savefig('error_histogram_graph_lstm.png')
    plt.close()
    logging.info("Error histogram saved as 'error_histogram_graph_lstm.png'")
    
    return y_test_actual, y_pred_actual

def plot_metrics(train_losses, val_losses, val_maes, val_mapes, train_mapes):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(range(1, len(val_maes) + 1), val_maes, label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Validation MAE')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(range(1, len(train_mapes) + 1), train_mapes, label='Training MAPE')
    plt.plot(range(1, len(val_mapes) + 1), val_mapes, label='Validation MAPE')
    plt.xlabel('Epoch')
    plt.ylabel('MAPE (%)')
    plt.title('Training and Validation MAPE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('metrics_graph_lstm.png')
    plt.close()
    logging.info("Metrics plot saved as 'metrics_graph_lstm.png'")

def main(main_dir, cache_dir='feature_cache'):
    logging.info(f"Processing main directory: {main_dir}")
    try:
        (train_features, val_features, test_features, 
         train_graphs, val_graphs, test_graphs, 
         train_file_names, val_file_names, test_file_names) = process_main_directory(main_dir, cache_dir)
        
        logging.info(f"Total training samples: {len(train_features)}, graphs: {len(train_graphs)}")
        logging.info(f"Total validation samples: {len(val_features)}, graphs: {len(val_graphs)}")
        logging.info(f"Total test samples: {len(test_features)}, graphs: {len(test_graphs)}")
        
        if len(train_features) == 0 or len(test_features) == 0:
            logging.error("No valid training or test data found")
            return None
        
        original_execution_times = {fname: f['execution_time'] for f, fname in zip(test_features, test_file_names)}
        
        train_graphs, val_graphs, test_graphs, scaler_y = prepare_data_for_model(
            train_features, val_features, test_features, train_graphs, val_graphs, test_graphs
        )
        
        train_loader, val_loader, test_loader = create_data_loaders(
            train_graphs, val_graphs, test_graphs, batch_size=8
        )
        
        input_size = train_graphs[0].x.shape[1] if train_graphs[0].x.size(0) > 0 else 1
        model = GraphLSTMModel(
            input_size=input_size,
            hidden_sizes=[256, 128, 64],
            output_size=1,
            dropout_rate=0.5,
            num_heads=4
        )
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
        
        logging.info("Building and training Graph LSTM model...")
        train_losses, val_losses, val_maes, val_mapes, train_mapes = train_model(
            model, train_loader, val_loader, criterion, optimizer, num_epochs=400, patience=50
        )
        
        plot_metrics(train_losses, val_losses, val_maes, val_mapes, train_mapes)
        
        logging.info("\nEvaluating model:")
        y_test_actual, y_pred_actual = evaluate_model(
            model, test_loader, scaler_y, test_file_names, is_log_transformed=True, original_execution_times=original_execution_times
        )
        
        logging.info("\nSaving the trained model as 'graph_lstm_model.pt'...")
        model.eval()
        device = next(model.parameters()).device
        logging.info(f"Model is on device: {device}")
        
        try:
            sample_data = train_graphs[0].to(device)
            traced_model = torch.jit.trace(model, sample_data)
            traced_model.save("graph_lstm_model.pt")
            logging.info("Model successfully saved as 'graph_lstm_model.pt'")
        except Exception as e:
            logging.error(f"Error saving the model: {str(e)}")
            torch.save(model.state_dict(), "graph_lstm_model_state.pt")
            logging.info("Saved model state dict as 'graph_lstm_model_state.pt'")
        
        # Save scaler
        with open('scaler_y_graph.pkl', 'wb') as f:
            pickle.dump(scaler_y, f)
        
        return model, scaler_y, y_test_actual, y_pred_actual
    
    except KeyboardInterrupt:
        logging.warning("Main function interrupted. Saving partial progress...")
        return None

if __name__ == "__main__":
    main_dir = "synthetic_data"
    result = main(main_dir)
    if result is not None:
        model, scaler_y, y_test_actual, y_pred_actual = result
