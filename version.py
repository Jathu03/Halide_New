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
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import matplotlib.pyplot as plt

# Define important metrics for scheduling sequence
important_metrics = [
    'bytes_at_production', 'bytes_at_realization', 'inner_parallelism', 'outer_parallelism',
    'num_vectors', 'points_computed_total', 'working_set'
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
    
    # Extract node and edge details
    op_counts_per_node = []
    all_op_types = set()
    for node in programming_details['Nodes']:
        node_feature = {'Name': node.get('Name', '')}
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
    
    for edge in programming_details['Edges']:
        edge_feature = {'From': edge.get('From', ''), 'To': edge.get('To', ''), 'Name': edge.get('Name', '')}
        edges_features.append(edge_feature)
    
    # Build tree structure
    num_nodes = max(len(nodes_features), 1)
    node_map = {node['Name']: i for i, node in enumerate(nodes_features)}
    children = [[] for _ in range(num_nodes)]
    for edge in edges_features:
        from_idx = node_map.get(edge['From'], -1)
        to_idx = node_map.get(edge['To'], -1)
        if from_idx != -1 and to_idx != -1:
            children[to_idx].append(from_idx)
    
    # Node features for TreeLSTM
    fixed_op_size = 10
    op_types = sorted(list(all_op_types))[:fixed_op_size]
    node_features = np.zeros((num_nodes, fixed_op_size))
    total_ops = sum(sum(node.get(f'op_{op}', 0) for op in all_op_types) for node in op_counts_per_node)
    for i, op_counts in enumerate(op_counts_per_node):
        for j, op in enumerate(op_types):
            node_features[i, j] = op_counts.get(op, 0) / max(total_ops, 1)
    
    scaler_tree = RobustScaler()
    node_features = scaler_tree.fit_transform(node_features)
    tree_features = {
        'node_features': torch.FloatTensor(node_features),
        'children': children
    }
    
    # Schedule-Specific Features
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
    
    # Scheduling Sequence with Data Augmentation
    scheduling_sequence = []
    for i, sf in enumerate(scheduling_features):
        sched_vector = [float(sf.get(metric, 0.0)) for metric in important_metrics]
        bytes_prod = sf.get('bytes_at_production', 0.0)
        points_total = sf.get('points_computed_total', 0.0)
        inner_p = sf.get('inner_parallelism', 0.0)
        outer_p = sf.get('outer_parallelism', 0.0)
        sched_vector.extend([
            np.log1p(inner_p * outer_p),
            bytes_prod / max(points_total, 1e-4)
        ])
        noise = np.random.normal(0, 0.05, len(sched_vector))
        augmented_vector = np.array(sched_vector, dtype=np.float32) + noise
        scheduling_sequence.append(augmented_vector)
    
    if not scheduling_sequence:
        scheduling_sequence = [np.zeros(len(important_metrics) + 2, dtype=np.float32)]
    
    seq_array = np.array(scheduling_sequence)
    scaler_seq = RobustScaler()
    scheduling_sequence = scaler_seq.fit_transform(seq_array)
    scheduling_sequence = np.nan_to_num(scheduling_sequence, nan=0.0).tolist()
    
    return {
        'tree_features': tree_features,
        'scheduling_sequence': scheduling_sequence,
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
    train_trees = [f['tree_features'] for f in train_features]
    test_trees = [f['tree_features'] for f in test_features]
    
    train_sequences = [torch.FloatTensor(f['scheduling_sequence']) for f in train_features]
    test_sequences = [torch.FloatTensor(f['scheduling_sequence']) for f in test_features]
    
    train_sequences_padded = pad_sequence(train_sequences, batch_first=True)
    test_sequences_padded = pad_sequence(test_sequences, batch_first=True)
    
    y_train_raw = np.array([f['execution_time'] for f in train_features])
    y_test_raw = np.array([f['execution_time'] for f in test_features])
    y_train_raw = np.clip(y_train_raw, 0, np.percentile(y_train_raw, 95))  # Reduced clipping
    y_test_raw = np.clip(y_test_raw, 0, np.percentile(y_test_raw, 95))
    
    y_train = np.log1p(y_train_raw).reshape(-1, 1)
    y_test = np.log1p(y_test_raw).reshape(-1, 1)
    
    scaler_y = RobustScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    y_train_scaled = np.nan_to_num(y_train_scaled, nan=0.0)
    y_test_scaled = np.nan_to_num(y_test_scaled, nan=0.0)
    
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Sequence input size: {train_sequences_padded.shape[2]}")
    
    return (train_trees, train_sequences_padded, y_train_tensor,
            test_trees, test_sequences_padded, y_test_tensor,
            scaler_y, train_sequences_padded.shape[2])

class TreeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)
        self.W_u = nn.Linear(input_size, hidden_size)
        self.U_u = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x, h_children, c_children):
        batch_size = x.size(0)
        
        h_sum = sum(h_children) if h_children else torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_sum = sum(c_children) if c_children else torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        i = torch.sigmoid(self.W_i(x) + self.U_i(h_sum))
        f = torch.sigmoid(self.W_f(x) + self.U_f(h_sum))
        o = torch.sigmoid(self.W_o(x) + self.U_o(h_sum))
        u = torch.tanh(self.W_u(x) + self.U_u(h_sum))
        
        c = i * u + f * c_sum
        h = o * torch.tanh(c)
        
        return h, c

class TreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeLSTM, self).__init__()
        self.cell1 = TreeLSTMCell(input_size, hidden_size)
        self.cell2 = TreeLSTMCell(hidden_size, hidden_size)  # Second layer
        self.hidden_size = hidden_size
    
    def forward(self, trees):
        batch_size = len(trees)
        device = trees[0]['node_features'].device
        
        h_roots = []
        
        for tree in trees:
            node_features = tree['node_features']
            children = tree['children']
            num_nodes = node_features.size(0)
            
            h1 = torch.zeros(num_nodes, self.hidden_size, device=device)
            c1 = torch.zeros(num_nodes, self.hidden_size, device=device)
            h2 = torch.zeros(num_nodes, self.hidden_size, device=device)
            c2 = torch.zeros(num_nodes, self.hidden_size, device=device)
            
            def compute_node(idx):
                if h1[idx].sum() != 0:  # Already computed
                    return h2[idx], c2[idx]
                
                h1_children = []
                c1_children = []
                for child_idx in children[idx]:
                    h_child, c_child = compute_node(child_idx)
                    h1_children.append(h_child)
                    c1_children.append(c_child)
                
                h1[idx], c1[idx] = self.cell1(node_features[idx].unsqueeze(0), [], [])  # First layer
                h2[idx], c2[idx] = self.cell2(h1[idx].unsqueeze(0), h1_children, c1_children)  # Second layer with children
                return h2[idx], c2[idx]
            
            root_idx = 0
            for i in range(num_nodes):
                if not any(i in child_list for child_list in children):
                    root_idx = i
                    break
            
            h_root, _ = compute_node(root_idx)
            h_roots.append(h_root)
        
        return torch.stack(h_roots)  # Shape: (batch_size, hidden_size)

class TreeLSTMModel(nn.Module):
    def __init__(self, tree_input_size, seq_input_size, hidden_size=384, output_size=1, dropout_rate=0.3):
        super(TreeLSTMModel, self).__init__()
        self.tree_lstm = TreeLSTM(tree_input_size, hidden_size)
        self.seq_lstm = nn.LSTM(seq_input_size, hidden_size, batch_first=True, bidirectional=True)
        
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8, dropout=dropout_rate, batch_first=True)
        self.attn_pool = nn.Linear(hidden_size * 2, 1)
        
        self.fc1 = nn.Linear(hidden_size * 3, 192)  # 384 * 3 = 1152
        self.bn1 = nn.BatchNorm1d(192)
        self.ln1 = nn.LayerNorm(192)
        self.fc2 = nn.Linear(192, 96)
        self.bn2 = nn.BatchNorm1d(96)
        self.ln2 = nn.LayerNorm(96)
        self.output_layer = nn.Linear(96, output_size)
        
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.residual_proj = nn.Linear(hidden_size * 3, 96) if hidden_size * 3 != 96 else None
    
    def forward(self, trees, seq_input):
        tree_h = self.tree_lstm(trees)  # (batch_size, hidden_size)
        
        seq_out, _ = self.seq_lstm(seq_input)  # (batch_size, seq_len, hidden_size * 2)
        attn_out, _ = self.attention(seq_out, seq_out, seq_out)
        seq_h = torch.softmax(self.attn_pool(attn_out), dim=1).transpose(1, 2).bmm(attn_out).squeeze(1)  # (batch_size, hidden_size * 2)
        
        combined = torch.cat([tree_h, seq_h], dim=-1)  # (batch_size, hidden_size * 3)
        
        x = self.fc1(combined)
        x = self.bn1(x)
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.ln2(x)
        x = self.gelu(x)
        
        residual = combined if self.residual_proj is None else self.residual_proj(combined)
        x = x + residual
        x = self.dropout(x)
        output = self.output_layer(x)
        
        return output

def focal_loss(outputs, targets, alpha=0.25, gamma=3.0):  # Increased gamma
    mse = (outputs - targets) ** 2
    pt = torch.exp(-mse)
    loss = alpha * (1 - pt) ** gamma * mse
    return torch.mean(loss)

def create_data_loaders(train_trees, train_sequences, y_train, test_trees, test_sequences, y_test, batch_size=8):
    train_dataset = TensorDataset(torch.arange(len(train_trees)), train_sequences, y_train)
    test_dataset = TensorDataset(torch.arange(len(test_trees)), test_sequences, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_trees, test_trees

def train_model(model, train_loader, test_loader, train_trees, test_trees, criterion, optimizer, num_epochs=500, patience=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for i, (tree_indices, seq_inputs, targets) in enumerate(train_loader):
            trees = [train_trees[idx.item()] for idx in tree_indices]
            seq_inputs, targets = seq_inputs.to(device), targets.to(device)
            for tree in trees:
                tree['node_features'] = tree['node_features'].to(device)
            
            outputs = model(trees, seq_inputs)
            loss = criterion(outputs, targets)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss detected at epoch {epoch+1}, batch {i+1}")
                return None, None
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss += loss.item() * seq_inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tree_indices, seq_inputs, targets in test_loader:
                trees = [test_trees[idx.item()] for idx in tree_indices]
                seq_inputs, targets = seq_inputs.to(device), targets.to(device)
                for tree in trees:
                    tree['node_features'] = tree['node_features'].to(device)
                
                outputs = model(trees, seq_inputs)
                loss = criterion(outputs, targets)
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
    plt.show()
    
    return train_losses, val_losses

def evaluate_model(model, test_trees, X_test_seq, y_test, y_scaler, file_names_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    X_test_seq = X_test_seq.to(device)
    for tree in test_trees:
        tree['node_features'] = tree['node_features'].to(device)
    
    with torch.no_grad():
        y_pred_scaled = model(test_trees, X_test_seq)
    
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
    
    (train_trees, train_sequences, y_train,
     test_trees, test_sequences, y_test,
     y_scaler, seq_input_size) = prepare_data_for_model(train_features, test_features)
    
    train_loader, test_loader, train_trees, test_trees = create_data_loaders(
        train_trees, train_sequences, y_train,
        test_trees, test_sequences, y_test,
        batch_size=8
    )
    
    global model
    model = TreeLSTMModel(
        tree_input_size=10,
        seq_input_size=seq_input_size,
        hidden_size=384,
        output_size=1,
        dropout_rate=0.3
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    
    print("Building and training Enhanced TreeLSTM model...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader, train_trees, test_trees,
        focal_loss, optimizer,
        num_epochs=500, patience=50
    )
    
    if train_losses is None or val_losses is None:
        print("Training failed due to invalid values")
        return None
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(
        model, test_trees, test_sequences, y_test,
        y_scaler, test_file_names
    )
    
    print(f"\nSummary for Comparison:")
    print(f"Model: Enhanced TreeLSTM")
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "synthetic_data"
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    model, y_scaler, y_test_actual, y_pred_actual = main(main_dir)
