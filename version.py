import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random
import matplotlib.pyplot as plt
from functools import lru_cache
from torch.cuda.amp import GradScaler, autocast

# Define important metrics
important_metrics = [
    'bytes_at_production', 'bytes_at_realization', 'inner_parallelism', 'outer_parallelism',
    'num_vectors', 'points_computed_total', 'working_set', 'memory_bandwidth', 'compute_intensity'
]

@lru_cache(maxsize=1024)
def get_execution_time(file_path):
    """Cached function to retrieve execution time from JSON files."""
    try:
        with open(file_path, 'rb') as f:
            data = json.loads(f.read().decode('utf-8', errors='replace').replace('\0', ''))
        schedules = data.get("scheduling_data", [])
        for item in schedules:
            if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
                value = item.get('value')
                return float(value) if value and value > 0 else None
        last_value = schedules[-1]["value"] if schedules else None
        return float(last_value) if last_value and last_value > 0 else None
    except Exception:
        return None

def extract_features_from_file(file_path, scaler_template=None, scaler_seq=None):
    """Optimized feature extraction with precomputed scalers."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        execution_time = get_execution_time(file_path)
        if not execution_time or not np.isfinite(execution_time):
            print(f"Warning: Invalid execution time in {file_path}")
            return None, scaler_template, scaler_seq
        
        programming_details = data.get("programming_details", {})
        nodes = programming_details.get('Nodes', [])
        edges = programming_details.get('Edges', [])
        if not nodes or not edges:
            print(f"Warning: Missing nodes or edges in {file_path}")
            return None, scaler_template, scaler_seq
        
        num_nodes = len(nodes)
        num_edges = len(edges)
        fixed_op_size = 15
        op_types = set()
        node_features = np.zeros((num_nodes, fixed_op_size), dtype=np.float32)
        total_ops = 0
        
        for i, node in enumerate(nodes):
            if 'Details' in node and 'Op histogram' in node['Details']:
                for op_line in node['Details']['Op histogram']:
                    parts = op_line.strip().split(':')
                    if len(parts) == 2:
                        op_name = f'op_{parts[0].strip().lower()}'
                        op_count = int(parts[1].strip())
                        if op_name not in op_types and len(op_types) < fixed_op_size:
                            op_types.add(op_name)
                        total_ops += op_count
            node_features[i] = [node['Details']['Op histogram'].count(op) for op in sorted(op_types)][:fixed_op_size] / max(total_ops, 1)
        
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.int8)
        node_map = {node.get('Name', str(i)): i for i, node in enumerate(nodes)}
        for edge in edges:
            from_idx = node_map.get(edge.get('From', ''), -1)
            to_idx = node_map.get(edge.get('To', ''), -1)
            if from_idx != -1 and to_idx != -1:
                adj_matrix[from_idx, to_idx] = 1
        
        graph_embedding = np.mean(node_features @ adj_matrix, axis=0) if num_nodes > 1 else np.mean(node_features, axis=0)
        template_features = np.concatenate([[num_nodes, num_edges, total_ops], graph_embedding])
        
        if scaler_template:
            template_features = scaler_template.transform(template_features.reshape(1, -1)).flatten()
        else:
            scaler_template = PowerTransformer(method='yeo-johnson')
            template_features = scaler_template.fit_transform(template_features.reshape(1, -1)).flatten()
        
        scheduling_data = data.get("scheduling_data", programming_details.get('Schedules', []))
        if not scheduling_data:
            print(f"Warning: No scheduling data in {file_path}")
            return None, scaler_template, scaler_seq
        
        scheduling_sequence = []
        seq_length = len(scheduling_data)
        for i, sched in enumerate(scheduling_data):
            sf = sched.get('Details', {}).get('scheduling_feature', {})
            sched_vector = np.array([float(sf.get(m, 0)) for m in important_metrics], dtype=np.float32)
            derived = [
                np.log1p(sched_vector[2] * sched_vector[3] + 1e-6),
                sched_vector[0] / max(sched_vector[5], 1e-4),
                i / max(seq_length, 1)
            ]
            combined = np.concatenate([sched_vector, derived])
            scheduling_sequence.append(np.concatenate([template_features, combined]))
        
        seq_array = np.array(scheduling_sequence, dtype=np.float32)
        if scaler_seq:
            seq_array = scaler_seq.transform(seq_array)
        else:
            scaler_seq = PowerTransformer(method='yeo-johnson')
            seq_array = scaler_seq.fit_transform(seq_array)
        
        return {
            'scheduling_sequence': seq_array,
            'execution_time': execution_time,
            'sequence_length': seq_length
        }, scaler_template, scaler_seq
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None, scaler_template, scaler_seq

class SchedulingDataset(Dataset):
    """Custom Dataset with error handling."""
    def __init__(self, file_paths, scaler_template=None, scaler_seq=None):
        self.file_paths = file_paths
        self.scaler_template = scaler_template
        self.scaler_seq = scaler_seq
        self.features = [None] * len(file_paths)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        if self.features[idx] is None:
            feat, _, _ = extract_features_from_file(self.file_paths[idx], self.scaler_template, self.scaler_seq)
            self.features[idx] = feat
        feat = self.features[idx]
        if feat is None:
            # Return a dummy item for invalid files to avoid crashing; filtered later
            dummy_seq = torch.zeros(1, 27, dtype=torch.float32)  # Adjust size based on expected feature length
            return dummy_seq, torch.tensor(0.0, dtype=torch.float32), 1
        return (torch.from_numpy(feat['scheduling_sequence']),
                torch.tensor(feat['execution_time'], dtype=torch.float32),
                feat['sequence_length'])

def process_main_directory(main_dir):
    """Optimized directory processing with validation."""
    all_file_paths = []
    for subdir in sorted(os.listdir(main_dir)):
        subdir_path = os.path.join(main_dir, subdir)
        if os.path.isdir(subdir_path):
            json_files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.json')]
            all_file_paths.extend(json_files)
    
    if len(all_file_paths) < 50:
        raise ValueError(f"Expected at least 50 files, found {len(all_file_paths)}")
    
    random.shuffle(all_file_paths)
    test_size = 50
    train_paths = all_file_paths[:-test_size]
    test_paths = all_file_paths[-test_size:]
    
    # Precompute scalers on first valid file
    scaler_template, scaler_seq = None, None
    for path in train_paths:
        feat, scaler_template, scaler_seq = extract_features_from_file(path)
        if feat is not None:
            break
    if scaler_template is None:
        raise ValueError("No valid files found to initialize scalers")
    
    train_dataset = SchedulingDataset(train_paths, scaler_template, scaler_seq)
    test_dataset = SchedulingDataset(test_paths, scaler_template, scaler_seq)
    
    # Filter out invalid entries
    valid_train_indices = [i for i in range(len(train_dataset)) if train_dataset[i][1].item() > 0]
    valid_test_indices = [i for i in range(len(test_dataset)) if test_dataset[i][1].item() > 0]
    
    train_dataset = torch.utils.data.Subset(train_dataset, valid_train_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, valid_test_indices)
    
    print(f"Total files: {len(all_file_paths)}, Training: {len(train_dataset)}, Testing: {len(test_dataset)}")
    return train_dataset, test_dataset, [os.path.relpath(p, main_dir) for p in test_paths]

def collate_fn(batch):
    """Custom collate function for efficient batching."""
    valid_batch = [(s, t, l) for s, t, l in batch if t > 0]
    if not valid_batch:
        return None  # Skip empty batches
    sequences, targets, lengths = zip(*valid_batch)
    sequences_padded = pad_sequence(sequences, batch_first=True)
    targets = torch.stack(targets)
    return sequences_padded, targets, lengths

class HybridTemporalNet(nn.Module):
    def __init__(self, seq_input_size, hidden_size=256, output_size=1, dropout_rate=0.2, num_heads=4):
        super(HybridTemporalNet, self).__init__()
        self.input_proj = nn.Linear(seq_input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size // 2, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout_rate, batch_first=True)
        self.pool = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()
    
    def forward(self, seq_input, lengths=None):
        x = self.gelu(self.input_proj(seq_input))
        if lengths:
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True)
            x, _ = self.lstm(packed)
            x, _ = pad_packed_sequence(x, batch_first=True, total_length=seq_input.size(1))
        else:
            x, _ = self.lstm(x)
        
        x = self.norm(x)
        mask = torch.arange(seq_input.size(1), device=x.device)[None, :] >= torch.tensor(lengths, device=x.device)[:, None] if lengths else None
        x, _ = self.attn(x, x, x, key_padding_mask=mask)
        
        weights = torch.softmax(self.pool(x), dim=1)
        context = (x * weights).sum(dim=1)
        return self.fc(self.dropout(context))

def combined_loss(outputs, targets):
    """Simplified loss function for efficiency."""
    mse = nn.MSELoss()(outputs, targets)
    rel_error = torch.mean(torch.abs(outputs - targets) / (targets.abs() + 1e-6))
    return mse + 0.1 * rel_error

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=200, patience=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            if batch is None:
                continue
            seq_inputs, targets, lengths = batch
            seq_inputs, targets = seq_inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(seq_inputs, lengths)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * seq_inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                if batch is None:
                    continue
                seq_inputs, targets, lengths = batch
                seq_inputs, targets = seq_inputs.to(device), targets.to(device)
                with autocast():
                    outputs = model(seq_inputs, lengths)
                    loss = criterion(outputs, targets)
                val_loss += loss.item() * seq_inputs.size(0)
        
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

def evaluate_model(model, test_loader, test_file_names):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    y_test_actual = []
    y_pred_actual = []
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            seq_inputs, targets, lengths = batch
            seq_inputs, targets = seq_inputs.to(device), targets.to(device)
            with autocast():
                outputs = model(seq_inputs, lengths)
            y_test_actual.extend(targets.cpu().numpy())
            y_pred_actual.extend(outputs.cpu().numpy())
    
    y_test_actual = np.array(y_test_actual)
    y_pred_actual = np.clip(y_pred_actual, 0, np.percentile(y_test_actual, 99))
    
    results_by_subfolder = {}
    for i, file_path in enumerate(test_file_names):
        subfolder = file_path.split('/')[0]
        results_by_subfolder.setdefault(subfolder, []).append({
            'file': file_path,
            'actual': y_test_actual[i],
            'predicted': y_pred_actual[i],
            'error_percentage': abs(y_test_actual[i] - y_pred_actual[i]) / y_test_actual[i] * 100 if y_test_actual[i] > 0 else 0
        })
    
    for subfolder, results in results_by_subfolder.items():
        print(f"\nResults for {subfolder}:")
        for r in results:
            print(f"File: {r['file']}, Actual: {r['actual']:.2f} ms, Predicted: {r['predicted']:.2f} ms, Error: {r['error_percentage']:.2f}%")
    
    mse = np.mean((y_test_actual - y_pred_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / (y_test_actual + 1e-8))) * 100
    
    print(f"\nMSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%")
    return y_test_actual, y_pred_actual

def main(main_dir):
    torch.backends.cudnn.benchmark = True
    train_dataset, test_dataset, test_file_names = process_main_directory(main_dir)
    
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("Error: No valid data after filtering")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    
    model = HybridTemporalNet(seq_input_size=train_dataset[0][0].shape[1])
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    train_model(model, train_loader, test_loader, combined_loss, optimizer, scheduler)
    evaluate_model(model, test_loader, test_file_names)

if __name__ == "__main__":
    main_dir = "synthetic_data"
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    main(main_dir)
