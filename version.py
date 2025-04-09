import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, PowerTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random
import matplotlib.pyplot as plt
from functools import lru_cache

# Define important metrics for scheduling sequence
important_metrics = [
    'bytes_at_production', 'bytes_at_realization', 'inner_parallelism', 'outer_parallelism',
    'num_vectors', 'points_computed_total', 'working_set', 'memory_bandwidth', 'compute_intensity'
]

@lru_cache(maxsize=1000)
def get_execution_time(file_path):
    """Cached function to retrieve execution time from JSON files."""
    try:
        with open(file_path, 'rb') as f:
            raw_content = f.read()
            content = raw_content.decode('utf-8', errors='ignore').replace('\0', '')
            data = json.loads(content)
        
        if 'programming_details' not in data:
            return None
        
        schedules = data.get("scheduling_data", [])
        for item in schedules:
            if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
                execution_time = item.get('value')
                if execution_time is not None and execution_time > 0:
                    return float(execution_time)
        
        last_value = schedules[-1]["value"] if schedules else None
        return float(last_value) if last_value and last_value > 0 else None
    
    except (json.JSONDecodeError, IndexError, KeyError, ValueError) as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def extract_features_from_file(file_path):
    """Extract features from a JSON file with optimized processing."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        execution_time = get_execution_time(file_path)
        if execution_time is None or not np.isfinite(execution_time):
            print(f"Warning: Invalid execution time in {file_path}")
            return None
        
        programming_details = data.get("programming_details", {})
        nodes = programming_details.get('Nodes', [])
        edges = programming_details.get('Edges', [])
        if not nodes or not edges:
            print(f"Warning: Incomplete programming_details in {file_path}")
            return None
        
        # Pre-allocate arrays for efficiency
        num_nodes = len(nodes)
        op_counts_per_node = [{} for _ in range(num_nodes)]
        all_op_types = set()
        node_memory_footprints = np.zeros(num_nodes)
        node_compute_intensities = np.zeros(num_nodes)
        
        # Extract node features
        for i, node in enumerate(nodes):
            if 'Details' in node and 'Op histogram' in node['Details']:
                for op_line in node['Details']['Op histogram']:
                    if ':' in op_line:
                        op_name, count = op_line.strip().split(':')
                        op_counts_per_node[i][f'op_{op_name.lower()}'] = int(count)
                        all_op_types.add(f'op_{op_name.lower()}')
            
            if 'Details' in node and 'Memory usage' in node['Details']:
                for mem_line in node['Details']['Memory usage']:
                    if isinstance(mem_line, str) and ':' in mem_line and 'bytes' in mem_line.lower():
                        try:
                            node_memory_footprints[i] = float(mem_line.split(':')[1].strip())
                        except ValueError:
                            pass
            
            total_ops = sum(op_counts_per_node[i].values())
            node_compute_intensities[i] = total_ops / max(node_memory_footprints[i], 1)
        
        # Edge features
        edge_data_sizes = np.zeros(len(edges))
        node_map = {node.get('Name', f'node_{i}'): i for i, node in enumerate(nodes)}
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        for i, edge in enumerate(edges):
            from_idx = node_map.get(edge.get('From', ''), -1)
            to_idx = node_map.get(edge.get('To', ''), -1)
            if from_idx != -1 and to_idx != -1:
                adj_matrix[from_idx, to_idx] = 1
            edge_data_sizes[i] = float(edge.get('Details', {}).get('Size', 0.0)) or 0.0
        
        # Graph-level metrics
        total_ops = sum(sum(counts.values()) for counts in op_counts_per_node)
        total_memory = node_memory_footprints.sum()
        graph_density = adj_matrix.sum() / max(num_nodes * (num_nodes - 1), 1)
        
        fixed_op_size = min(15, len(all_op_types))
        op_types = sorted(list(all_op_types))[:fixed_op_size]
        node_features = np.zeros((num_nodes, fixed_op_size), dtype=np.float32)
        for i, counts in enumerate(op_counts_per_node):
            for j, op in enumerate(op_types):
                node_features[i, j] = counts.get(op, 0) / max(total_ops, 1)
        
        graph_embedding = np.mean(node_features, axis=0) if num_nodes == 1 else np.mean(node_features * adj_matrix.sum(axis=1, keepdims=True), axis=0)
        
        template_features = np.concatenate([
            [num_nodes, len(edges), total_ops, len(all_op_types) / max(num_nodes, 1), graph_density, total_memory],
            graph_embedding
        ])
        scaler_template = PowerTransformer(method='yeo-johnson')
        template_features = scaler_template.fit_transform(template_features.reshape(1, -1)).flatten()
        template_features = np.nan_to_num(template_features, nan=0.0)
        
        # Schedule-specific features
        scheduling_data = data.get("scheduling_data", programming_details.get('Schedules', []))
        if not scheduling_data:
            print(f"Warning: No scheduling data in {file_path}")
            return None
        
        scheduling_sequence = []
        seq_length = len(scheduling_data)
        
        for i, sched in enumerate(scheduling_data):
            sf = sched.get('Details', {}).get('scheduling_feature', {})
            sched_vector = np.array([float(sf.get(metric, 0.0)) for metric in important_metrics], dtype=np.float32)
            
            bytes_prod, points_total = sf.get('bytes_at_production', 0.0), sf.get('points_computed_total', 1e-4)
            inner_p, outer_p = sf.get('inner_parallelism', 0.0), sf.get('outer_parallelism', 1e-4)
            working_set = sf.get('working_set', 1e-4)
            
            derived = np.array([
                np.log1p(inner_p * outer_p + 1e-6),
                np.log1p(bytes_prod / points_total + 1e-6),
                inner_p / outer_p,
                i / max(seq_length, 1)
            ], dtype=np.float32)
            
            combined_vector = np.concatenate([sched_vector, derived])
            # Fix: Remove dtype from np.random.normal and cast afterward
            noise = np.random.normal(0, 0.01, combined_vector.shape).astype(np.float32)
            scheduling_sequence.append(np.concatenate([template_features, combined_vector + noise]))
        
        if not scheduling_sequence:
            scheduling_sequence = [np.concatenate([template_features, np.zeros(len(important_metrics) + 4, dtype=np.float32)])]
        
        seq_array = np.array(scheduling_sequence, dtype=np.float32)
        scaler_seq = PowerTransformer(method='yeo-johnson')
        scheduling_sequence = scaler_seq.fit_transform(seq_array)
        scheduling_sequence = np.nan_to_num(scheduling_sequence, nan=0.0)
        
        return {
            'scheduling_sequence': scheduling_sequence.tolist(),
            'execution_time': execution_time,
            'sequence_length': len(scheduling_sequence)
        }
    except Exception as e:
        print(f"Error in extract_features_from_file: {str(e)} for {file_path}")
        return None

def process_directory(directory_path):
    """Process all JSON files in a directory efficiently."""
    all_features, file_names = [], []
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    
    for filename in sorted(json_files):
        file_path = os.path.join(directory_path, filename)
        features = extract_features_from_file(file_path)
        if features:
            all_features.append(features)
            file_names.append(filename)
    
    return all_features, file_names

def process_main_directory(main_dir):
    """Process main directory with stratified sampling."""
    all_features, all_file_names = [], []
    subdirs = [d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]
    
    if not subdirs:
        raise ValueError(f"No subdirectories found in {main_dir}")
    
    for subdir in sorted(subdirs):
        subdir_path = os.path.join(main_dir, subdir)
        features, file_names = process_directory(subdir_path)
        all_features.extend(features)
        all_file_names.extend(os.path.join(subdir, fname) for fname in file_names)
        print(f"Processed subdir {subdir}: {len(features)} files")
    
    total_files = len(all_features)
    if total_files < 50:
        raise ValueError(f"Expected at least 50 files, found {total_files}")
    
    execution_times = np.array([f['execution_time'] for f in all_features])
    quantiles = pd.qcut(execution_times, 5, labels=False, duplicates='drop')
    combined = list(zip(all_features, all_file_names, quantiles))
    random.shuffle(combined)
    
    quantile_groups = {q: [] for q in set(quantiles)}
    for feature, file_name, q in combined:
        quantile_groups[q].append((feature, file_name))
    
    test_size, train_features, test_features, test_file_names = 50, [], [], []
    samples_per_quantile = test_size // len(quantile_groups)
    remaining = test_size % len(quantile_groups)
    
    for q in sorted(quantile_groups.keys()):
        quota = samples_per_quantile + (1 if remaining > 0 else 0)
        remaining -= 1 if remaining > 0 else 0
        group = quantile_groups[q]
        test_subset = group[:quota]
        test_features.extend(f for f, _ in test_subset)
        test_file_names.extend(fn for _, fn in test_subset)
        train_features.extend(f for f, _ in group[quota:])
    
    train_file_names = [fn for f, fn in combined if fn not in test_file_names]
    print(f"\nTotal files: {total_files}, Training files: {len(train_features)}, Testing files: {len(test_features)}")
    
    return train_features, test_features, test_file_names

class SchedulingDataset(Dataset):
    def __init__(self, features):
        self.sequences = [torch.tensor(f['scheduling_sequence'], dtype=torch.float32) for f in features]
        self.targets = torch.tensor([f['execution_time'] for f in features], dtype=torch.float32)
        self.lengths = torch.tensor([f['sequence_length'] for f in features], dtype=torch.long)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.lengths[idx]

def prepare_data_for_model(train_features, test_features):
    """Prepare data with efficient tensor handling."""
    train_dataset = SchedulingDataset(train_features)
    test_dataset = SchedulingDataset(test_features)
    
    train_sequences = pad_sequence([d[0] for d in train_dataset], batch_first=True)
    y_train = train_dataset.targets
    train_lengths = train_dataset.lengths
    
    test_sequences = pad_sequence([d[0] for d in test_dataset], batch_first=True)
    y_test = test_dataset.targets
    test_lengths = test_dataset.lengths
    
    pt = PowerTransformer(method='yeo-johnson')
    y_train_transformed = pt.fit_transform(np.log1p(y_train.numpy()).reshape(-1, 1))
    y_test_transformed = pt.transform(np.log1p(y_test.numpy()).reshape(-1, 1))
    
    y_train_tensor = torch.tensor(y_train_transformed, dtype=torch.float32).squeeze()
    y_test_tensor = torch.tensor(y_test_transformed, dtype=torch.float32).squeeze()
    
    print(f"Sequence input size: {train_sequences.shape[2]}")
    return train_sequences, y_train_tensor, train_lengths, test_sequences, y_test_tensor, test_lengths, pt

class HybridTemporalNet(nn.Module):
    def __init__(self, seq_input_size, hidden_size=256, output_size=1, dropout_rate=0.2):
        super(HybridTemporalNet, self).__init__()
        self.input_proj = nn.Linear(seq_input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(hidden_size * 2, num_heads=4, dropout=dropout_rate, batch_first=True)
        self.pool = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()
    
    def forward(self, x, lengths=None):
        x = self.input_proj(x)
        if lengths is not None:
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
            lstm_out, _ = self.lstm(packed)
            x, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=x.size(1))
        else:
            x, _ = self.lstm(x)
        
        mask = torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None] if lengths is not None else None
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        x = self.norm(attn_out)
        
        weights = torch.softmax(self.pool(x), dim=1)
        context = (x * weights).sum(dim=1)
        return self.fc(self.dropout(self.gelu(context)))

def combined_loss(outputs, targets):
    """Simplified loss function for efficiency."""
    mse = nn.MSELoss()(outputs, targets)
    return mse

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=200, patience=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    best_val_loss, epochs_no_improve, train_losses, val_losses = float('inf'), 0, [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for seq_inputs, targets, lengths in train_loader:
            seq_inputs, targets, lengths = seq_inputs.to(device), targets.to(device), lengths.to(device)
            optimizer.zero_grad()
            outputs = model(seq_inputs, lengths)
            loss = criterion(outputs.squeeze(), targets)
            
            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch+1}")
                return None, None
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * seq_inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq_inputs, targets, lengths in test_loader:
                seq_inputs, targets, lengths = seq_inputs.to(device), targets.to(device), lengths.to(device)
                outputs = model(seq_inputs, lengths)
                val_loss += criterion(outputs.squeeze(), targets).item() * seq_inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            model.load_state_dict(torch.load('best_model.pt'))
            break
    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, y_transform, file_names_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    y_pred, y_true = [], []
    with torch.no_grad():
        for seq_inputs, targets, lengths in test_loader:
            seq_inputs, lengths = seq_inputs.to(device), lengths.to(device)
            outputs = model(seq_inputs, lengths)
            y_pred.extend(outputs.cpu().numpy().flatten())
            y_true.extend(targets.numpy().flatten())
    
    y_pred = np.array(y_pred).reshape(-1, 1)
    y_true = np.array(y_true).reshape(-1, 1)
    
    y_test_actual = np.expm1(y_transform.inverse_transform(y_true))
    y_pred_actual = np.expm1(y_transform.inverse_transform(y_pred))
    y_pred_actual = np.clip(y_pred_actual, 0, np.percentile(y_test_actual, 99))
    
    results_by_subfolder = {}
    for i, file_path in enumerate(file_names_test):
        subfolder = file_path.split('/')[0]
        results_by_subfolder.setdefault(subfolder, []).append({
            'file': file_path,
            'actual': y_test_actual[i, 0],
            'predicted': y_pred_actual[i, 0],
            'error_percentage': abs(y_test_actual[i, 0] - y_pred_actual[i, 0]) / max(y_test_actual[i, 0], 1e-6) * 100
        })
    
    for subfolder, results in results_by_subfolder.items():
        print(f"\nResults for {subfolder}:")
        for result in results:
            print(f"File: {result['file']}, Actual: {result['actual']:.2f} ms, Predicted: {result['predicted']:.2f} ms, Error: {result['error_percentage']:.2f}%")
    
    mse = np.mean((y_test_actual - y_pred_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / (y_test_actual + 1e-8))) * 100
    
    print(f"\nOverall Model Performance: MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
    return y_test_actual, y_pred_actual

def main(main_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_features, test_features, test_file_names = process_main_directory(main_dir)
    if not train_features or not test_features:
        print("Error: Insufficient data")
        return
    
    train_seq, y_train, train_lengths, test_seq, y_test, test_lengths, y_transform = prepare_data_for_model(train_features, test_features)
    
    train_loader = DataLoader(
        SchedulingDataset(train_features), batch_size=32, shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        SchedulingDataset(test_features), batch_size=32, shuffle=False, num_workers=2, pin_memory=True
    )
    
    model = HybridTemporalNet(seq_input_size=train_seq.shape[2])
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    train_losses, val_losses = train_model(model, train_loader, test_loader, combined_loss, optimizer, scheduler)
    if train_losses is None:
        print("Training failed")
        return
    
    evaluate_model(model, test_loader, y_transform, test_file_names)

if __name__ == "__main__":
    main_dir = "synthetic_data"
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    main(main_dir)
