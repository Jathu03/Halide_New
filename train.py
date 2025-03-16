import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

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
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path}: {str(e)}")
        return None
    except UnicodeDecodeError as e:
        print(f"Error: Encoding issue in {file_path}: {str(e)}")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred while processing {file_path}: {str(e)}")
        return None

def extract_features_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    execution_time = get_execution_time(file_path)
    if execution_time is None:
        print(f"Warning: No execution time found in {file_path}")
        return None
    
    # Extract programming details (nodes and edges)
    programming_details = data.get("programming_details", {})
    nodes = programming_details.get('Nodes', [])
    edges = programming_details.get('Edges', [])
    
    # Build an adjacency list representation for the tree/graph
    node_dict = {node.get('Name', f'node_{i}'): node for i, node in enumerate(nodes)}
    adj_list = {node_name: [] for node_name in node_dict}
    node_indices = {name: idx for idx, name in enumerate(node_dict.keys())}
    
    for edge in edges:
        from_node = edge.get('From', '')
        to_node = edge.get('To', '')
        if from_node in adj_list and to_node in node_dict:
            adj_list[from_node].append(to_node)
    
    # Find root nodes (nodes with no incoming edges)
    incoming_edges = set(edge['To'] for edge in edges if edge.get('To'))
    roots = [name for name in node_dict if name not in incoming_edges]
    
    # Extract schedule features
    scheduling_data = data.get("scheduling_data", programming_details.get('Schedules', []))
    schedule_features = []
    for sched in scheduling_data:
        sched_feature = {}
        sched_feature['Name'] = sched.get('name', '')  # Adjusted for 'name' key
        if isinstance(sched, dict) and 'Details' in sched and 'scheduling_feature' in sched['Details']:
            sf = sched['Details']['scheduling_feature']
            for key, value in sf.items():
                sched_feature[key] = value
        schedule_features.append(sched_feature)
    
    # Map schedule features to nodes (heuristic: match by name or use first schedule)
    node_schedule_map = {}
    important_metrics = [
        'bytes_at_production', 'bytes_at_realization', 'bytes_at_root', 'bytes_at_task',
        'inner_parallelism', 'outer_parallelism', 'num_productions', 'num_realizations',
        'num_scalars', 'num_vectors', 'points_computed_total', 'working_set'
    ]
    for node_name in node_dict:
        # Try to match schedule by node name, otherwise use the first schedule
        matched_sched = None
        for sched in schedule_features:
            if sched['Name'] and node_name in sched['Name']:
                matched_sched = sched
                break
        if not matched_sched and schedule_features:
            matched_sched = schedule_features[0]
        
        node_schedule_map[node_name] = {metric: matched_sched.get(metric, 0) for metric in important_metrics} if matched_sched else {metric: 0 for metric in important_metrics}
    
    # DFS pre-order traversal to create a node sequence
    node_sequence = []
    visited = set()
    
    def dfs(node_name, depth):
        if node_name in visited:
            return
        visited.add(node_name)
        
        node = node_dict[node_name]
        # Node features: operation counts, depth, schedule metrics
        node_features = [0] * (15 + len(important_metrics))  # Adjust size based on features
        # Structural feature: depth
        node_features[0] = depth
        # Operation counts from Op histogram
        if 'Details' in node and 'Op histogram' in node['Details']:
            op_hist = node['Details']['Op histogram']
            op_counts = {}
            for op_line in op_hist:
                parts = op_line.strip().split(':')
                if len(parts) == 2:
                    op_name = parts[0].strip().lower()
                    op_count = int(parts[1].strip())
                    op_counts[op_name] = op_count
            # Map common operations to fixed indices
            op_types = ['add', 'sub', 'mul', 'div', 'load', 'store', 'call', 'shift', 'and', 'or', 'xor', 'cmp', 'mov', 'cast', 'other']
            for i, op in enumerate(op_types, 1):  # Start after depth
                node_features[i] = op_counts.get(op, 0)
        
        # Schedule metrics
        sched_metrics = node_schedule_map[node_name]
        for i, metric in enumerate(important_metrics, len(op_types) + 1):
            node_features[i] = sched_metrics[metric]
        
        node_sequence.append(node_features)
        
        # Traverse children
        for child in adj_list[node_name]:
            dfs(child, depth + 1)
    
    # Start DFS from each root
    for root in roots:
        dfs(root, 0)
    
    # Global features (for context)
    global_features = {
        'execution_time': execution_time,
        'nodes_count': len(nodes),
        'edges_count': len(edges),
        'scheduling_count': len(schedule_features)
    }
    if len(nodes) > 0 and len(edges) > 0:
        global_features['node_edge_ratio'] = len(nodes) / len(edges)
    else:
        global_features['node_edge_ratio'] = 0
    
    # Add aggregated schedule metrics
    total_bytes_at_production = sum(sf.get('bytes_at_production', 0) for sf in schedule_features if isinstance(sf, dict))
    total_vectors = sum(sf.get('num_vectors', 0) for sf in schedule_features if isinstance(sf, dict))
    total_parallelism = sum(sf.get('inner_parallelism', 0) * sf.get('outer_parallelism', 1) for sf in schedule_features if isinstance(sf, dict))
    
    global_features['total_bytes_at_production'] = total_bytes_at_production
    global_features['total_vectors'] = total_vectors
    global_features['total_parallelism'] = total_parallelism
    if total_vectors > 0:
        global_features['bytes_per_vector'] = total_bytes_at_production / total_vectors
    
    return {'node_sequence': node_sequence, 'global_features': global_features}

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

def clean_and_transform_features(train_features, test_features, max_seq_len=50):
    all_node_sequences = []
    all_global_features = []
    all_execution_times = []
    
    # Separate node sequences, global features, and execution times
    for feat in train_features + test_features:
        all_node_sequences.append(feat['node_sequence'])
        all_global_features.append(feat['global_features'])
        all_execution_times.append(feat['global_features']['execution_time'])
    
    # Pad or truncate node sequences to max_seq_len
    padded_sequences = []
    for seq in all_node_sequences:
        if len(seq) < max_seq_len:
            padded_seq = seq + [[0] * len(seq[0])] * (max_seq_len - len(seq))
        else:
            padded_seq = seq[:max_seq_len]
        padded_sequences.append(padded_seq)
    
    # Convert global features to DataFrame
    global_df = pd.DataFrame(all_global_features)
    global_df = global_df.fillna(0)
    
    constant_columns = [col for col in global_df.columns 
                       if col != 'execution_time' and global_df[col].nunique() == 1]
    global_df = global_df.drop(columns=constant_columns)
    print(f"Dropped {len(constant_columns)} constant columns")
    
    global_df['execution_time_log'] = np.log1p(global_df['execution_time'])
    
    # Feature selection - keep only numeric columns
    numeric_cols = global_df.select_dtypes(include=['number']).columns
    global_df = global_df[numeric_cols]
    
    train_size = len(train_features)
    train_global_df = global_df.iloc[:train_size]
    test_global_df = global_df.iloc[train_size:]
    train_sequences = padded_sequences[:train_size]
    test_sequences = padded_sequences[train_size:]
    
    return train_sequences, test_sequences, train_global_df, test_global_df

def prepare_data_for_model(train_features, test_features, max_seq_len=50):
    train_sequences, test_sequences, train_global_df, test_global_df = clean_and_transform_features(train_features, test_features, max_seq_len)
    
    y_train = train_global_df['execution_time_log'].values.reshape(-1, 1)
    y_test = test_global_df['execution_time_log'].values.reshape(-1, 1)
    X_train_global = train_global_df.drop(['execution_time', 'execution_time_log'], axis=1)
    X_test_global = test_global_df.drop(['execution_time', 'execution_time_log'], axis=1)
    
    scaler_X_global = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_global_scaled = scaler_X_global.fit_transform(X_train_global)
    X_test_global_scaled = scaler_X_global.transform(X_test_global)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    # Convert node sequences to numpy arrays
    X_train_seq = np.array(train_sequences)  # [n_samples, max_seq_len, node_feature_dim]
    X_test_seq = np.array(test_sequences)
    
    # Normalize node sequence features (excluding depth, which is at index 0)
    node_feature_dim = X_train_seq.shape[-1]
    X_train_seq_flat = X_train_seq.reshape(-1, node_feature_dim)
    X_test_seq_flat = X_test_seq.reshape(-1, node_feature_dim)
    scaler_X_seq = StandardScaler()
    X_train_seq_flat[:, 1:] = scaler_X_seq.fit_transform(X_train_seq_flat[:, 1:])
    X_test_seq_flat[:, 1:] = scaler_X_seq.transform(X_test_seq_flat[:, 1:])
    X_train_seq = X_train_seq_flat.reshape(X_train_seq.shape)
    X_test_seq = X_test_seq_flat.reshape(X_test_seq.shape)
    
    # Combine node sequence with global features by repeating global features across timesteps
    X_train_global_expanded = np.repeat(X_train_global_scaled[:, np.newaxis, :], max_seq_len, axis=1)
    X_test_global_expanded = np.repeat(X_test_global_scaled[:, np.newaxis, :], max_seq_len, axis=1)
    
    X_train_combined = np.concatenate([X_train_seq, X_train_global_expanded], axis=-1)
    X_test_combined = np.concatenate([X_test_seq, X_test_global_expanded], axis=-1)
    
    X_train_tensor = torch.FloatTensor(X_train_combined)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_combined)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Input feature dimension: {X_train_combined.shape[-1]}")
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_y, X_train_combined.shape[-1]

class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=1, dropout_rate=0.3):
        super(EnhancedLSTMModel, self).__init__()
        
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        self.lstm_layers.append(nn.LSTM(input_size, hidden_sizes[0], batch_first=True))
        self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(nn.LSTM(hidden_sizes[i-1], hidden_sizes[i], batch_first=True))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        self.attention = nn.Linear(hidden_sizes[-1], 1)
        
        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        
        self.fc_layers.append(nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2))
        self.bn_layers.append(nn.BatchNorm1d(hidden_sizes[-1] // 2))
        
        self.fc_layers.append(nn.Linear(hidden_sizes[-1] // 2, hidden_sizes[-1] // 4))
        self.bn_layers.append(nn.BatchNorm1d(hidden_sizes[-1] // 4))
        
        self.output_layer = nn.Linear(hidden_sizes[-1] // 4, output_size)
        
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        self.has_residual = (hidden_sizes[-1] // 4 == hidden_sizes[-1] // 2)
        if not self.has_residual:
            self.residual_adapter = nn.Linear(hidden_sizes[-1] // 2, hidden_sizes[-1] // 4)
        
    def attention_net(self, lstm_output):
        attn_weights = self.attention(lstm_output).squeeze(2)
        soft_attn_weights = torch.softmax(attn_weights, 1)
        context = torch.bmm(soft_attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context
        
    def forward(self, x):
        lstm_out = x
        for i, (lstm, dropout) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            lstm_out, _ = lstm(lstm_out)
            if i < len(self.lstm_layers) - 1:
                lstm_out = dropout(lstm_out)
        
        attn_output = self.attention_net(lstm_out)
        
        fc_out = self.fc_layers[0](attn_output)
        fc_out = self.bn_layers[0](fc_out)
        fc_out = self.leaky_relu(fc_out)
        
        residual = fc_out
        if not self.has_residual:
            residual = self.residual_adapter(residual)
        
        fc_out = self.fc_layers[1](fc_out)
        fc_out = self.bn_layers[1](fc_out)
        fc_out = self.leaky_relu(fc_out)
        
        fc_out = fc_out + residual
        
        output = self.output_layer(fc_out)
        return output

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=150, patience=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
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
    
    return train_losses, val_losses

def evaluate_model(model, X_test, y_test, y_scaler, file_names_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    X_test = X_test.to(device)
    with torch.no_grad():
        y_pred_scaled = model(X_test)
    
    y_pred_scaled = y_pred_scaled.cpu().numpy()
    y_test = y_test.cpu().numpy()
    
    y_test_transformed = y_scaler.inverse_transform(y_test)
    y_pred_transformed = y_scaler.inverse_transform(y_pred_scaled)
    
    y_test_actual = np.expm1(y_test_transformed)  # Since execution_time_log was used
    y_pred_actual = np.expm1(np.clip(y_pred_transformed, 0, None))
    
    results_by_subfolder = {}
    for i, file_path in enumerate(file_names_test):
        subfolder = file_path.split('/')[0]
        if subfolder not in results_by_subfolder:
            results_by_subfolder[subfolder] = []
        
        results_by_subfolder[subfolder].append({
            'file': file_path,
            'actual': y_test_actual[i][0],
            'predicted': y_pred_actual[i][0],
            'error_percentage': abs(y_test_actual[i][0] - y_pred_actual[i][0]) / y_test_actual[i][0] * 100 if y_test_actual[i][0] > 0 else 0
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
    print(f"Processing main directory: {main_dir}")
    train_features, test_features, test_file_names = process_main_directory(main_dir)
    
    print(f"Total training samples: {len(train_features)} (randomly selected)")
    print(f"Total test samples: {len(test_features)} (50 randomly selected)")
    
    if len(train_features) == 0 or len(test_features) == 0:
        print("Error: No valid training or test data found")
        return None
    
    X_train, y_train, X_test, y_test, y_scaler, input_size = prepare_data_for_model(train_features, test_features)
    
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16)
    
    model = EnhancedLSTMModel(
        input_size=input_size,
        hidden_sizes=[128, 64, 32],
        output_size=1,
        dropout_rate=0.3
    )
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    print("Building and training Enhanced LSTM model...")
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        test_loader, 
        criterion, 
        optimizer, 
        num_epochs=150,
        patience=20
    )
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(model, X_test, y_test, y_scaler, test_file_names)
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "Output_Programs"
    random.seed(42)
    model, y_scaler, y_test_actual, y_pred_actual = main(main_dir)
