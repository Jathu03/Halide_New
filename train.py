import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, QuantileTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import OneCycleLR
import random
from sklearn.ensemble import IsolationForest
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# Define important metrics for scheduling sequence with additional metrics
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
        
        schedules = data["scheduling_data"]
        for item in schedules:
            if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
                execution_time = item.get('value')
                if execution_time is not None:
                    return float(execution_time)
        
        print(f"Warning: 'total_execution_time_ms' not found in 'Schedules' of {file_path}")
        return schedules[len(schedules)-1]["value"]
    
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
    
    # Enhanced extraction of node features
    if programming_details:
        if 'Nodes' in programming_details:
            for node in programming_details['Nodes']:
                node_feature = {'Name': node.get('Name', '')}
                
                # Extract all details for richer feature set
                if 'Details' in node:
                    details = node['Details']
                    # Process Op histogram
                    if 'Op histogram' in details:
                        op_hist = details['Op histogram']
                        for op_line in op_hist:
                            parts = op_line.strip().split(':')
                            if len(parts) == 2:
                                op_name = parts[0].strip()
                                op_count = int(parts[1].strip())
                                node_feature[f'op_{op_name.lower()}'] = op_count
                    
                    # Extract other nested details if available
                    for key, value in details.items():
                        if key != 'Op histogram' and isinstance(value, (int, float, str)):
                            node_feature[f'detail_{key.lower().replace(" ", "_")}'] = value
                
                nodes_features.append(node_feature)
        
        if 'Edges' in programming_details:
            for edge in programming_details['Edges']:
                edge_feature = {
                    'From': edge.get('From', ''),
                    'To': edge.get('To', ''),
                    'Name': edge.get('Name', '')
                }
                # Extract additional edge properties if available
                if 'Details' in edge:
                    for key, value in edge['Details'].items():
                        if isinstance(value, (int, float, str)):
                            edge_feature[f'edge_{key.lower().replace(" ", "_")}'] = value
                
                edges_features.append(edge_feature)
    
    # Enhanced scheduling feature extraction
    scheduling_features = []
    scheduling_data = data.get("scheduling_data", None)
    if not scheduling_data and programming_details and 'Schedules' in programming_details:
        scheduling_data = programming_details['Schedules']
    
    if scheduling_data:
        for sched in scheduling_data:
            sched_feature = {'Name': sched.get('Name', '')}
            if 'Details' in sched:
                if 'scheduling_feature' in sched['Details']:
                    sf = sched['Details']['scheduling_feature']
                    sched_feature.update(sf)
                # Extract all other details
                for key, value in sched['Details'].items():
                    if key != 'scheduling_feature' and isinstance(value, (int, float, str)):
                        sched_feature[f'sched_{key.lower().replace(" ", "_")}'] = value
            scheduling_features.append(sched_feature)
    
    # Enhanced scheduling sequence with expanded derived features
    scheduling_sequence = []
    for sf in scheduling_features:
        # Base features from important metrics
        seq_vector = [float(sf.get(metric, 0.0)) for metric in important_metrics]
        
        # Extract values for derived features
        bytes_prod = sf.get('bytes_at_production', 0.0)
        bytes_real = sf.get('bytes_at_realization', 0.0)
        bytes_root = sf.get('bytes_at_root', 0.0)
        bytes_task = sf.get('bytes_at_task', 0.0)
        num_vec = sf.get('num_vectors', 0.0)
        num_scalars = sf.get('num_scalars', 0.0)
        points_total = sf.get('points_computed_total', 0.0)
        working_set = sf.get('working_set', 0.0)
        inner_para = sf.get('inner_parallelism', 0.0)
        outer_para = sf.get('outer_parallelism', 0.0)
        num_prod = sf.get('num_productions', 0.0)
        num_real = sf.get('num_realizations', 0.0)
        
        # Advanced derived features with safe division
        safe_div = lambda x, y: np.clip(x / max(abs(y), 1e-6), -1e4, 1e4) if y != 0 else 0.0
        
        # Original derived features
        seq_vector.append(safe_div(bytes_prod, bytes_real))
        seq_vector.append(safe_div(bytes_prod, num_vec))
        seq_vector.append(safe_div(points_total, num_vec))
        seq_vector.append(safe_div(working_set, bytes_prod))
        
        # New derived features
        seq_vector.append(safe_div(bytes_prod, bytes_task))
        seq_vector.append(safe_div(bytes_root, bytes_real))
        seq_vector.append(safe_div(num_vec, num_scalars))
        seq_vector.append(safe_div(num_prod, num_real))
        seq_vector.append(safe_div(working_set, points_total))
        seq_vector.append(inner_para * outer_para)  # Total parallelism
        seq_vector.append(safe_div(bytes_prod, inner_para))
        seq_vector.append(safe_div(points_total, outer_para))
        
        # Log transformations for key metrics (adding 1 to handle zeros)
        seq_vector.append(np.log1p(bytes_prod))
        seq_vector.append(np.log1p(points_total))
        seq_vector.append(np.log1p(working_set))
        
        scheduling_sequence.append(seq_vector)
    
    if not scheduling_sequence:
        # Create dummy sequence if none exists, matching the expanded feature vector size
        scheduling_sequence = [[0.0] * (len(important_metrics) + 15)]
    
    # Per-sample normalization with robust statistics
    seq_array = np.array(scheduling_sequence)
    seq_median = np.median(seq_array, axis=0, keepdims=True)
    seq_iqr = np.percentile(seq_array, 75, axis=0, keepdims=True) - np.percentile(seq_array, 25, axis=0, keepdims=True)
    seq_iqr = np.maximum(seq_iqr, 1e-6)  # Avoid division by zero
    scheduling_sequence = (seq_array - seq_median) / seq_iqr
    scheduling_sequence = np.nan_to_num(scheduling_sequence, nan=0.0).tolist()
    
    # Enhanced scalar features with graph topology metrics
    op_counts = {}
    for node in nodes_features:
        for key, value in node.items():
            if key.startswith('op_'):
                op_counts[key] = op_counts.get(key, 0) + value
    
    # Graph structural features
    total_ops = sum(op_counts.values())
    num_nodes = max(len(nodes_features), 1)
    num_edges = len(edges_features)
    
    # Create node connectivity graph
    node_connections = {}
    for edge in edges_features:
        from_node = edge.get('From', '')
        to_node = edge.get('To', '')
        if from_node not in node_connections:
            node_connections[from_node] = []
        node_connections[from_node].append(to_node)
    
    # Calculate graph metrics
    max_degree = 0
    avg_degree = 0
    if node_connections:
        degrees = [len(connections) for node, connections in node_connections.items()]
        max_degree = max(degrees) if degrees else 0
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
    
    # Advanced scalar features
    scalar_features = {
        'nodes_count': num_nodes,
        'edges_count': num_edges,
        'node_edge_ratio': num_nodes / max(num_edges, 1),
        'total_ops': total_ops,
        'op_diversity': len(op_counts) / max(num_nodes, 1),
        'avg_ops_per_node': total_ops / max(num_nodes, 1),
        'edge_density': num_edges / max(num_nodes * max(num_nodes - 1, 1), 1),
        'max_node_degree': max_degree,
        'avg_node_degree': avg_degree,
        'graph_complexity': num_edges * total_ops / max(num_nodes, 1),
        'total_parallelism': sum(sf.get('inner_parallelism', 0) * sf.get('outer_parallelism', 1) for sf in scheduling_features),
        'max_working_set': max([sf.get('working_set', 0.0) for sf in scheduling_features], default=0.0),
        'avg_points_computed': np.mean([sf.get('points_computed_total', 0.0) for sf in scheduling_features]) if scheduling_features else 0.0,
        'max_bytes_production': max([sf.get('bytes_at_production', 0.0) for sf in scheduling_features], default=0.0),
        'scheduling_sequence_length': len(scheduling_sequence)
    }
    
    # Add op counts to scalar features
    scalar_features.update(op_counts)
    
    # Add log transformations of key metrics
    for key in ['nodes_count', 'edges_count', 'total_ops', 'max_working_set', 'max_bytes_production']:
        if key in scalar_features and scalar_features[key] > 0:
            scalar_features[f'log_{key}'] = np.log(scalar_features[key])
        else:
            scalar_features[f'log_{key}'] = 0.0
    
    # Replace NaN in scalar features
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
    
    # Outlier detection using IsolationForest
    print("Performing outlier detection...")
    exec_times = np.array([f['execution_time'] for f in all_features]).reshape(-1, 1)
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    outlier_labels = iso_forest.fit_predict(np.log1p(exec_times))
    
    # Filter out outliers
    filtered_features = []
    filtered_file_names = []
    for i, (feat, fname, label) in enumerate(zip(all_features, all_file_names, outlier_labels)):
        if label == 1:  # Not an outlier
            filtered_features.append(feat)
            filtered_file_names.append(fname)
        else:
            print(f"Outlier detected: {fname} with execution time {feat['execution_time']}")
    
    print(f"Removed {len(all_features) - len(filtered_features)} outliers")
    
    # Shuffle data
    combined = list(zip(filtered_features, filtered_file_names))
    random.shuffle(combined)
    all_features, all_file_names = zip(*combined)
    
    test_size = 50
    train_features = all_features[:-test_size]
    test_features = all_features[-test_size:]
    train_file_names = all_file_names[:-test_size]
    test_file_names = all_file_names[-test_size:]
    
    print(f"Total files after outlier removal: {len(filtered_features)}")
    print(f"Training files: {len(train_features)}")
    print(f"Testing files: {len(test_features)}")
    
    return train_features, test_features, list(test_file_names)

def prepare_data_for_model(train_features, test_features):
    train_sequences = [torch.FloatTensor(f['scheduling_sequence']) for f in train_features]
    test_sequences = [torch.FloatTensor(f['scheduling_sequence']) for f in test_features]
    
    # Padding with attention mask for variable length sequences
    train_sequences_padded = pad_sequence(train_sequences, batch_first=True)
    test_sequences_padded = pad_sequence(test_sequences, batch_first=True)
    
    # Create attention masks for padded sequences
    train_mask = torch.ones((train_sequences_padded.size(0), train_sequences_padded.size(1)))
    test_mask = torch.ones((test_sequences_padded.size(0), test_sequences_padded.size(1)))
    
    for i, seq in enumerate(train_sequences):
        train_mask[i, seq.size(0):] = 0
    
    for i, seq in enumerate(test_sequences):
        test_mask[i, seq.size(0):] = 0
    
    # Create scalar feature dataframes
    train_scalar_df = pd.DataFrame([f['scalar_features'] for f in train_features])
    test_scalar_df = pd.DataFrame([f['scalar_features'] for f in test_features])
    
    # Fill NaN values
    train_scalar_df = train_scalar_df.fillna(0)
    test_scalar_df = test_scalar_df.fillna(0)
    
    # Handle missing columns in test set
    for col in train_scalar_df.columns:
        if col not in test_scalar_df.columns:
            test_scalar_df[col] = 0
    
    # Ensure test_scalar_df has the same columns in the same order as train_scalar_df
    test_scalar_df = test_scalar_df[train_scalar_df.columns]
    
    # Transform target variable with log
    y_train = np.log1p(np.array([f['execution_time'] for f in train_features])).reshape(-1, 1)
    y_test = np.log1p(np.array([f['execution_time'] for f in test_features])).reshape(-1, 1)
    
    # Use QuantileTransformer for scalar features to handle skewed distributions
    scaler_X_scalar = QuantileTransformer(output_distribution='normal')
    scaler_y = RobustScaler()
    
    train_scalar_scaled = scaler_X_scalar.fit_transform(train_scalar_df)
    test_scalar_scaled = scaler_X_scalar.transform(test_scalar_df)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Handle NaN values
    train_scalar_scaled = np.nan_to_num(train_scalar_scaled, nan=0.0)
    test_scalar_scaled = np.nan_to_num(test_scalar_scaled, nan=0.0)
    y_train_scaled = np.nan_to_num(y_train_scaled, nan=0.0)
    y_test_scaled = np.nan_to_num(y_test_scaled, nan=0.0)
    
    # Convert to tensors
    train_scalar_tensor = torch.FloatTensor(train_scalar_scaled)
    test_scalar_tensor = torch.FloatTensor(test_scalar_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Sequence input size: {train_sequences_padded.shape[2]}")
    print(f"Scalar input size: {train_scalar_tensor.shape[1]}")
    
    return (train_sequences_padded, train_mask, train_scalar_tensor, y_train_tensor,
            test_sequences_padded, test_mask, test_scalar_tensor, y_test_tensor,
            scaler_y, train_sequences_padded.shape[2], train_scalar_tensor.shape[1])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class EnhancedTransformerModel(nn.Module):
    def __init__(self, seq_input_size, scalar_input_size, hidden_size=256, nhead=8, 
                num_encoder_layers=4, dim_feedforward=512, dropout=0.2, output_size=1):
        super(EnhancedTransformerModel, self).__init__()
        
        # Sequence processing
        self.seq_embedding = nn.Linear(seq_input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        
        encoder_layers = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, 
                                                dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Scalar processing
        self.scalar_embedding = nn.Sequential(
            nn.Linear(scalar_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism for sequence-scalar integration
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output layers with residual connections
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
    
    def forward(self, seq_input, seq_mask, scalar_input):
        # Process sequence input
        seq_embedded = self.seq_embedding(seq_input)
        seq_embedded = self.pos_encoder(seq_embedded)
        
        # Create padding mask for transformer
        # In PyTorch, the src_key_padding_mask should be of shape [N, S] 
        # where N is batch size and S is sequence length
        # A value of True indicates that the corresponding key value will be ignored
        pad_mask = (seq_mask == 0)
        
        # Debug information (uncomment if needed)
        # print(f"seq_input shape: {seq_input.shape}")
        # print(f"seq_mask shape: {seq_mask.shape}")
        # print(f"pad_mask shape: {pad_mask.shape}")
        
        # Apply transformer
        # For PyTorch versions where key_padding_mask expects shape [S, N], transpose the mask
        # Check your PyTorch version and adjust accordingly
        try:
            # First try with the mask as is (newer PyTorch versions)
            transformer_out = self.transformer_encoder(seq_embedded, src_key_padding_mask=pad_mask)
        except AssertionError:
            # If that fails, try with transposed mask (older PyTorch versions)
            transformer_out = self.transformer_encoder(seq_embedded, src_key_padding_mask=pad_mask.transpose(0, 1))
        
        # Global sequence representation (averaging non-padded elements)
        masked_transformer_out = transformer_out * seq_mask.unsqueeze(-1)
        seq_lengths = torch.sum(seq_mask, dim=1, keepdim=True)
        seq_lengths = torch.clamp(seq_lengths, min=1.0)  # Avoid division by zero
        seq_repr = torch.sum(masked_transformer_out, dim=1) / seq_lengths
        
        # Process scalar input
        scalar_repr = self.scalar_embedding(scalar_input)
        
        # Attention between sequence and scalar representations
        query = self.query_proj(scalar_repr).unsqueeze(1)
        key = self.key_proj(transformer_out)
        value = self.value_proj(transformer_out)
        
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        # Apply mask to attention scores
        attention_scores = attention_scores.masked_fill(pad_mask.unsqueeze(1), -1e9)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context_vector = torch.matmul(attention_weights, value).squeeze(1)
        
        # Concatenate context and scalar representations
        combined = torch.cat([context_vector, scalar_repr], dim=1)
        
        # Output layers with residual connections and layer normalization
        x = self.fc1(combined)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.selu(x)
        x = self.dropout2(x)
        
        output = self.fc3(x)
        
        return output

class MultiTaskLoss(nn.Module):
    def __init__(self, mse_weight=1.0, huber_weight=0.5, l1_weight=0.3, quantile_weight=0.2, delta=1.0):
        super(MultiTaskLoss, self).__init__()
        self.mse_weight = mse_weight
        self.huber_weight = huber_weight
        self.l1_weight = l1_weight
        self.quantile_weight = quantile_weight
        self.delta = delta
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=delta)
        self.l1 = nn.L1Loss()
    
    def forward(self, outputs, targets):
        mse_loss = self.mse(outputs, targets)
        huber_loss = self.huber(outputs, targets)
        l1_loss = self.l1(outputs, targets)
        
        # Quantile loss (penalizes underestimation more than overestimation)
        errors = targets - outputs
        quantile_loss = torch.mean(torch.max(0.8 * errors, 0.2 * errors))
        
        return (self.mse_weight * mse_loss + 
                self.huber_weight * huber_loss + 
                self.l1_weight * l1_loss + 
                self.quantile_weight * quantile_loss)

def create_data_loaders(train_sequences, train_mask, train_scalar, y_train, 
                       test_sequences, test_mask, test_scalar, y_test, batch_size=16):
    train_dataset = TensorDataset(train_sequences, train_mask, train_scalar, y_train)
    test_dataset = TensorDataset(test_sequences, test_mask, test_scalar, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=300, patience=40, accumulation_steps=4, scheduler=None):
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
        
        for i, (seq_inputs, seq_mask, scalar_inputs, targets) in enumerate(train_loader):
            seq_inputs = seq_inputs.to(device)
            seq_mask = seq_mask.to(device)
            scalar_inputs = scalar_inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(seq_inputs, seq_mask, scalar_inputs)
            loss = criterion(outputs, targets)
            
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}, batch {i+1}")
                continue
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps * seq_inputs.size(0)
        
        # Step optimizer if remaining batches
        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq_inputs, seq_mask, scalar_inputs, targets in test_loader:
                seq_inputs = seq_inputs.to(device)
                seq_mask = seq_mask.to(device)
                scalar_inputs = scalar_inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(seq_inputs, seq_mask, scalar_inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * seq_inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss and not np.isnan(val_loss):
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            print(f"New best model saved with validation loss: {val_loss:.4f}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model state")
    
    return train_losses, val_losses

def evaluate_model(model, X_test_seq, X_test_mask, X_test_scalar, y_test, y_scaler, file_names_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    test_dataset = TensorDataset(X_test_seq, X_test_mask, X_test_scalar, y_test)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for seq_inputs, seq_mask, scalar_inputs, targets in test_loader:
            seq_inputs = seq_inputs.to(device)
            seq_mask = seq_mask.to(device)
            scalar_inputs = scalar_inputs.to(device)
            
            outputs = model(seq_inputs, seq_mask, scalar_inputs)
            
            # Move predictions back to CPU for numpy processing
            pred_cpu = outputs.cpu().numpy()
            targets_cpu = targets.cpu().numpy()
            
            predictions.extend(pred_cpu)
            actuals.extend(targets_cpu)
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Inverse transform the scaled values
    predictions_orig = np.expm1(y_scaler.inverse_transform(predictions))
    actuals_orig = np.expm1(y_scaler.inverse_transform(actuals))
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions_orig - actuals_orig))
    mape = 100 * np.mean(np.abs((predictions_orig - actuals_orig) / actuals_orig))
    rmse = np.sqrt(np.mean((predictions_orig - actuals_orig) ** 2))
    
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    
    # Create detailed results for each file
    results = []
    for i in range(len(predictions_orig)):
        results.append({
            'file_name': file_names_test[i],
            'actual': float(actuals_orig[i][0]),
            'predicted': float(predictions_orig[i][0]),
            'error': float(predictions_orig[i][0] - actuals_orig[i][0]),
            'error_percentage': float(100 * (predictions_orig[i][0] - actuals_orig[i][0]) / actuals_orig[i][0])
        })
    
    return {
        'metrics': {
            'mae': mae,
            'mape': mape,
            'rmse': rmse
        },
        'predictions': results
    }

def save_model_and_results(model, train_history, evaluation_results, output_dir="model_output"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    
    # Save training history
    with open(os.path.join(output_dir, "training_history.json"), 'w') as f:
        json.dump(train_history, f)
    
    # Save evaluation results
    with open(os.path.join(output_dir, "evaluation_results.json"), 'w') as f:
        json.dump(evaluation_results, f)
    
    print(f"Model and results saved to {output_dir}")

def main(data_dir, output_dir="model_output"):
    # Process data
    print("Processing data...")
    train_features, test_features, test_file_names = process_main_directory(data_dir)
    
    # Prepare data for model
    print("Preparing data for model...")
    (train_sequences, train_mask, train_scalar, y_train,
     test_sequences, test_mask, test_scalar, y_test,
     y_scaler, seq_input_size, scalar_input_size) = prepare_data_for_model(train_features, test_features)
    
    # Create model
    print("Creating model...")
    model = EnhancedTransformerModel(
        seq_input_size=seq_input_size,
        scalar_input_size=scalar_input_size,
        hidden_size=256,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=512,
        dropout=0.2
    )
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, test_loader = create_data_loaders(
        train_sequences, train_mask, train_scalar, y_train,
        test_sequences, test_mask, test_scalar, y_test,
        batch_size=16
    )
    
    # Set up loss, optimizer, and scheduler
    criterion = MultiTaskLoss(mse_weight=1.0, huber_weight=0.5, l1_weight=0.3, quantile_weight=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # One cycle learning rate scheduler
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=0.005,
        steps_per_epoch=len(train_loader),
        epochs=300,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    # Train model
    print("Training model...")
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        test_loader, 
        criterion, 
        optimizer, 
        num_epochs=300, 
        patience=40,
        scheduler=scheduler
    )
    
    # Evaluate model
    print("Evaluating model...")
    evaluation_results = evaluate_model(
        model,
        test_sequences,
        test_mask,
        test_scalar,
        y_test,
        y_scaler,
        test_file_names
    )
    
    # Save model and results
    train_history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    save_model_and_results(model, train_history, evaluation_results, output_dir)
    
    print("Done!")
    return evaluation_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate execution time prediction model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing data subdirectories')
    parser.add_argument('--output_dir', type=str, default='model_output', help='Directory to save model and results')
    
    args = parser.parse_args()
    
    main(args.data_dir, args.output_dir)
