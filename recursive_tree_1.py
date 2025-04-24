import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# Define fixed set of features based on feature importance
FIXED_FEATURES = [
    'cache_hits', 'cache_misses', 'sched_num_realizations', 'sched_num_productions',
    'sched_points_computed_total', 'sched_innermost_loop_extent', 'sched_inner_parallelism',
    'sched_outer_parallelism', 'sched_bytes_at_realization', 'sched_bytes_at_production',
    'sched_bytes_at_root', 'sched_unique_bytes_read_per_realization', 'sched_working_set',
    'sched_vector_size', 'sched_num_vectors', 'sched_num_scalars', 'sched_bytes_at_task',
    'sched_working_set_at_task', 'sched_working_set_at_production', 'sched_working_set_at_realization',
    'sched_working_set_at_root', 'total_parallelism', 'scheduling_count', 'total_bytes_at_production',
    'total_vectors', 'computation_efficiency', 'memory_pressure', 'memory_utilization_ratio',
    'bytes_processing_rate', 'bytes_per_parallelism', 'bytes_per_vector', 'nodes_count',
    'edges_count', 'node_edge_ratio', 'nodes_per_schedule', 'op_diversity',
    'op_add', 'op_sub', 'op_mul', 'op_div', 'op_eq', 'op_lt', 'op_le', 'op_min', 'op_max',
    'op_constant', 'op_variable', 'op_funccall',
    'memory_transpose_0', 'memory_transpose_1', 'memory_slice_0', 'memory_slice_1',
    'memory_broadcast_0', 'memory_broadcast_1', 'memory_pointwise_0', 'memory_pointwise_1'
]

# Node class for tree representation
class TreeNode:
    def __init__(self, features, children=None):
        self.features = features  # Dictionary of features
        self.children = children if children is not None else []  # List of TreeNode objects

# Feature extraction function
def extract_features(json_data, execution_time_ms=None):
    features = {}
    
    # Extract global features
    features['cache_hits'] = json_data.get('cache_hits', 0)
    features['cache_misses'] = json_data.get('cache_misses', 0)
    
    # Extract op_histogram
    op_histogram = defaultdict(int)
    if 'op_histogram' in json_data:
        for op, count in json_data['op_histogram'].items():
            op_histogram[op.lower()] += count
    for op, count in op_histogram.items():
        features[f'op_{op.lower()}'] = count
    
    # Extract memory patterns
    memory_patterns = defaultdict(lambda: [0, 0, 0, 0])
    if 'memory_patterns' in json_data:
        for pattern, values in json_data['memory_patterns'].items():
            memory_patterns[pattern] = [sum(x) for x in zip(memory_patterns[pattern], values)]
    for pattern, values in memory_patterns.items():
        for i, val in enumerate(values):
            features[f'memory_{pattern.lower()}_{i}'] = val
    
    # Extract scheduling features
    scheduling_keys = [
        'num_realizations', 'num_productions', 'points_computed_total', 'innermost_loop_extent',
        'inner_parallelism', 'outer_parallelism', 'bytes_at_realization', 'bytes_at_production',
        'bytes_at_root', 'unique_bytes_read_per_realization', 'working_set', 'vector_size',
        'num_vectors', 'num_scalars', 'bytes_at_task', 'working_set_at_task', 'working_set_at_production',
        'working_set_at_realization', 'working_set_at_root'
    ]
    if 'scheduling' in json_data:
        for key in scheduling_keys:
            features[f'sched_{key}'] = json_data['scheduling'].get(key, 0)
    
    # Derived features
    features['total_parallelism'] = features.get('sched_inner_parallelism', 0) + features.get('sched_outer_parallelism', 0)
    features['scheduling_count'] = features.get('sched_num_realizations', 0) + features.get('sched_num_productions', 0)
    features['total_bytes_at_production'] = features.get('sched_bytes_at_production', 0)
    features['total_vectors'] = features.get('sched_num_vectors', 0)
    features['computation_efficiency'] = (features.get('sched_points_computed_total', 0) /
                                         features.get('sched_bytes_at_realization', 1)) if features.get('sched_bytes_at_realization', 0) != 0 else 0
    features['memory_pressure'] = (features.get('sched_working_set', 0) /
                                  features.get('sched_bytes_at_root', 1)) if features.get('sched_bytes_at_root', 0) != 0 else 0
    features['memory_utilization_ratio'] = (features.get('sched_unique_bytes_read_per_realization', 0) /
                                           features.get('sched_bytes_at_task', 1)) if features.get('sched_bytes_at_task', 0) != 0 else 0
    features['bytes_processing_rate'] = (features.get('sched_bytes_at_realization', 0) /
                                        (execution_time_ms or 1)) if execution_time_ms and execution_time_ms > 0 else 0
    features['bytes_per_parallelism'] = (features.get('sched_bytes_at_task', 0) /
                                        features.get('total_parallelism', 1)) if features.get('total_parallelism', 0) != 0 else 0
    features['bytes_per_vector'] = (features.get('sched_bytes_at_realization', 0) /
                                   features.get('sched_num_vectors', 1)) if features.get('sched_num_vectors', 0) != 0 else 0
    
    # Structural features
    nodes_count = 1  # Count current node
    edges_count = len(json_data.get('children', []))
    features['nodes_count'] = nodes_count
    features['edges_count'] = edges_count
    features['node_edge_ratio'] = nodes_count / (edges_count + 1)
    features['nodes_per_schedule'] = nodes_count / (features.get('scheduling_count', 1)) if features.get('scheduling_count', 0) != 0 else 0
    features['op_diversity'] = len([k for k, v in features.items() if k.startswith('op_') and v > 0])
    
    # Filter to fixed features
    fixed_features = {key: features.get(key, 0.0) for key in FIXED_FEATURES}
    return fixed_features

# Build tree from JSON
def build_tree(json_data, execution_time_ms=None):
    features = extract_features(json_data, execution_time_ms)
    children = []
    for child in json_data.get('children', []):
        if child['name'] != 'Global Features':  # Skip global features node
            child_tree = build_tree(child, execution_time_ms)
            children.append(child_tree)
    return TreeNode(features, children)

# Process Tree_Output directory
def process_tree_output_directory(main_dir):
    trees = []
    file_names = []
    invalid_files = []
    valid_execution_times = []
    
    # First pass: collect valid execution times
    for root, dirs, files in os.walk(main_dir):
        if 'tree_representation.json' in files:
            file_path = os.path.join(root, 'tree_representation.json')
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                global_node = next((child for child in json_data['children'] if child['name'] == 'Global Features'), None)
                if global_node:
                    exec_time = global_node.get('execution_time_ms', 0)
                    if exec_time > 0 and np.isfinite(exec_time):
                        valid_execution_times.append(exec_time)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Compute median execution time
    median_exec_time = np.median(valid_execution_times) if valid_execution_times else 1.0
    print(f"Median execution time for imputation: {median_exec_time:.2f} ms")
    
    # Second pass: build trees
    for root, dirs, files in os.walk(main_dir):
        if 'tree_representation.json' in files:
            file_path = os.path.join(root, 'tree_representation.json')
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                global_node = next((child for child in json_data['children'] if child['name'] == 'Global Features'), None)
                exec_time = global_node.get('execution_time_ms', 0) if global_node else 0
                if exec_time <= 0 or not np.isfinite(exec_time):
                    invalid_files.append(file_path)
                    exec_time = median_exec_time
                    print(f"Imputed execution time {median_exec_time:.2f} ms for {file_path}")
                tree = build_tree(json_data, exec_time)
                tree.features['execution_time_ms'] = exec_time
                trees.append(tree)
                file_names.append(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    if not trees:
        raise ValueError("No valid JSON files found in Tree_Output directory.")
    
    # Save invalid files log
    log_path = os.path.join(main_dir, 'invalid_files_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("Files with invalid execution times (imputed with median):\n")
        for file_path in invalid_files:
            f.write(f"{file_path}\n")
    
    total_files = len(trees)
    print(f"Total files found: {total_files}")
    print(f"Files with invalid execution times (imputed): {len(invalid_files)}")
    if total_files < 50:
        raise ValueError(f"Expected at least 50 files total, found {total_files}")
    
    combined = list(zip(trees, file_names))
    random.shuffle(combined)
    trees, file_names = zip(*combined)
    
    test_size = min(50, total_files)
    train_trees = trees[:-test_size]
    test_trees = trees[-test_size:]
    train_file_names = file_names[:-test_size]
    test_file_names = file_names[-test_size:]
    
    print(f"Training files: {len(train_trees)}")
    print(f"Testing files: {len(test_trees)}")
    
    return train_trees, test_trees, list(test_file_names)

# Dataset for trees
class TreeDataset(Dataset):
    def __init__(self, trees, scaler_y=None):
        self.trees = trees
        self.scaler_y = scaler_y
        self.y = np.array([tree.features['execution_time_ms'] for tree in trees])
        self.y = np.clip(self.y, 0, np.percentile(self.y, 99))
        self.y = np.log1p(self.y).reshape(-1, 1)
        if scaler_y is None:
            self.scaler_y = RobustScaler()
            self.y_scaled = self.scaler_y.fit_transform(self.y)
        else:
            self.y_scaled = scaler_y.transform(self.y)
        self.y_scaled = np.nan_to_num(self.y_scaled, nan=0.0)
    
    def __len__(self):
        return len(self.trees)
    
    def __getitem__(self, idx):
        return self.trees[idx], torch.FloatTensor(self.y_scaled[idx])

# Recursive LSTM model
class RecursiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=1, num_layers=2, dropout_rate=0.2):
        super(RecursiveLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()
        self.ln = nn.LayerNorm(hidden_size * 2)
    
    def forward(self, node, device):
        # Process node features
        features = torch.FloatTensor([[node.features.get(key, 0.0) for key in FIXED_FEATURES]]).to(device)
        lstm_out, _ = self.lstm(features)  # Shape: (1, 1, hidden_size * 2)
        lstm_out = self.ln(lstm_out)
        node_out = lstm_out[:, -1, :]  # Take last time step
        
        # Process children recursively
        if node.children:
            child_outputs = []
            for child in node.children:
                child_out = self.forward(child, device)
                child_outputs.append(child_out)
            child_outputs = torch.stack(child_outputs)  # Shape: (num_children, hidden_size * 2)
            child_agg = child_outputs.mean(dim=0, keepdim=True)  # Aggregate by averaging
            node_out = node_out + child_agg  # Combine with node output
        
        # Final processing
        node_out = self.dropout(self.gelu(self.fc(node_out)))
        if not node.children:  # Only leaf nodes contribute to output
            output = self.output_layer(node_out)
        else:
            output = self.output_layer(node_out)
        return node_out

# Custom loss function
def custom_loss(outputs, targets, feature_importances, huber_delta=0.5, mae_weight=0.3, l1_lambda=1e-5):
    huber = nn.HuberLoss(delta=huber_delta)(outputs, targets)
    mae = torch.mean(torch.abs(outputs - targets))
    l1_reg = sum(param.abs().sum() for param in model.parameters()) * l1_lambda
    
    weights = torch.ones_like(targets)
    for feature, importance in feature_importances.items():
        weights = weights * (1.0 + importance * 2.0)
    
    weighted_huber = (huber * weights).mean()
    weighted_mae = (mae * weights).mean()
    return weighted_huber + mae_weight * weighted_mae + l1_reg

# Create data loaders
def create_data_loaders(train_trees, test_trees, batch_size=32):
    train_dataset = TreeDataset(train_trees)
    test_dataset = TreeDataset(test_trees, scaler_y=train_dataset.scaler_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_dataset.scaler_y

# Train the model with checkpointing
def train_model(model, train_loader, test_loader, criterion, optimizer, feature_importances, checkpoint_path='check.pth', num_epochs=700, patience=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model.to(device)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    start_epoch = 0
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        print(f"Resumed training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        
        for trees, targets in train_loader:
            optimizer.zero_grad()
            targets = targets.to(device)
            outputs = []
            for tree in trees:
                out = model(tree, device)
                outputs.append(out)
            outputs = torch.stack(outputs).squeeze()
            
            loss = criterion(outputs, targets, feature_importances)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss detected at epoch {epoch+1}")
                return None, None
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * len(trees)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for trees, targets in test_loader:
                targets = targets.to(device)
                outputs = []
                for tree in trees:
                    out = model(tree, device)
                    outputs.append(out)
                outputs = torch.stack(outputs).squeeze()
                loss = criterion(outputs, targets, feature_importances)
                val_loss += loss.item() * len(trees)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save checkpoint
        if val_loss < best_val_loss and not np.isnan(val_loss) and not np.isinf(val_loss):
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            if best_model_state is not None:
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
    plt.close()
    
    return train_losses, val_losses

# Evaluate the model
def evaluate_model(model, test_trees, test_file_names, scaler_y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    test_dataset = TreeDataset(test_trees, scaler_y)
    y_test = torch.FloatTensor(test_dataset.y_scaled).to(device)
    y_test_actual = np.expm1(scaler_y.inverse_transform(test_dataset.y_scaled))
    
    outputs = []
    with torch.no_grad():
        for tree in test_trees:
            out = model(tree, device)
            outputs.append(out)
    y_pred_scaled = torch.stack(outputs).squeeze().cpu().numpy()
    y_pred_actual = np.expm1(scaler_y.inverse_transform(y_pred_scaled))
    
    results_by_subfolder = {}
    for i, file_path in enumerate(test_file_names):
        subfolder = '/'.join(file_path.split('/')[:-1])
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

# Main function
def main(main_dir):
    if torch.cuda.is_available():
        torch.cuda.init()
        print(f"CUDA initialized. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Using CPU.")
    
    print(f"Processing main directory: {main_dir}")
    train_trees, test_trees, test_file_names = process_tree_output_directory(main_dir)
    
    if len(train_trees) == 0 or len(test_trees) == 0:
        print("Error: No valid training or test data found")
        return None
    
    train_loader, test_loader, scaler_y = create_data_loaders(train_trees, test_trees, batch_size=32)
    
    global model
    model = RecursiveLSTM(
        input_size=len(FIXED_FEATURES),
        hidden_size=256,
        output_size=1,
        num_layers=2,
        dropout_rate=0.2
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-4)
    
    feature_importances = {
        'cache_hits': 0.5860,
        'bytes_processing_rate': 0.2893,
        'sched_bytes_at_task': 0.0422,
        'sched_working_set_at_root': 0.0248,
        'sched_bytes_at_realization': 0.0055,
        'sched_unique_bytes_read_per_realization': 0.0049
    }
    
    print("Building and training Recursive LSTM model...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader,
        custom_loss, optimizer, feature_importances,
        checkpoint_path='check.pth',
        num_epochs=700, patience=50
    )
    
    if train_losses is None or val_losses is None:
        print("Training failed due to invalid values")
        return None
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(
        model, test_trees, test_file_names, scaler_y
    )
    
    print(f"\nSummary for Comparison:")
    print(f"Model: RecursiveLSTM")
    
    return model, scaler_y, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "Tree_Output"
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    model, y_scaler, y_test_actual, y_pred_actual = main(main_dir)
