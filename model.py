import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict

class ChildSumTreeLSTM(nn.Module):
    """
    Child-Sum Tree-LSTM for graph-structured data.
    """
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super(ChildSumTreeLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        # Tree-LSTM parameters
        self.W_iou = nn.Linear(input_dim, 3 * hidden_dim)
        self.U_iou = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, node_features, adj_list, node_order):
        """
        Forward pass for Tree-LSTM over the graph.
        node_features: (num_nodes, input_dim)
        adj_list: dict mapping node_idx -> list of child indices
        node_order: list of node indices in processing order (bottom-up)
        """
        h = torch.zeros(node_features.size(0), self.hidden_dim, device=node_features.device)
        c = torch.zeros(node_features.size(0), self.hidden_dim, device=node_features.device)
        
        for node_idx in node_order:
            # Gather children hidden states
            child_h = []
            child_c = []
            for child_idx in adj_list.get(node_idx, []):
                child_h.append(h[child_idx])
                child_c.append(c[child_idx])
            
            child_h_sum = sum(child_h) if child_h else torch.zeros(self.hidden_dim, device=h.device)
            child_h = torch.stack(child_h) if child_h else torch.zeros(0, self.hidden_dim, device=h.device)
            
            # Input features for current node
            x = node_features[node_idx]
            
            # Tree-LSTM computations
            iou = self.W_iou(x) + self.U_iou(child_h_sum)
            i, o, u = torch.split(iou, self.hidden_dim, dim=-1)
            i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
            
            f = torch.sigmoid(self.W_f(x).unsqueeze(0).repeat(len(child_h), 1) + self.U_f(child_h)) if child_h.size(0) > 0 else torch.zeros(0, self.hidden_dim, device=h.device)
            c_tilde = sum(f * c_child for f, c_child in zip(f, child_c)) if child_c else torch.zeros(self.hidden_dim, device=c.device)
            
            c[node_idx] = i * u + c_tilde
            h[node_idx] = o * torch.tanh(c[node_idx])
        
        # Aggregate root node’s hidden state
        root_h = h[0]  # Assume node 0 is a root or use last processed node
        output = self.fc(self.dropout(root_h))
        return output.squeeze()

class GraphDataset(Dataset):
    """
    Dataset for graphs with node features and adjacency lists.
    """
    def __init__(self, graphs, execution_times):
        self.graphs = graphs  # List of (node_features, adj_list, node_order)
        self.execution_times = execution_times

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        node_features, adj_list, node_order = self.graphs[idx]
        return {
            'node_features': torch.tensor(node_features, dtype=torch.float32),
            'adj_list': adj_list,
            'node_order': node_order,
            'execution_time': torch.tensor(self.execution_times[idx], dtype=torch.float32)
        }

def parse_graph_data(json_data):
    """
    Parse JSON to extract node features, adjacency list, and execution time.
    """
    if 'programming_details' not in json_data:
        return None, None, None, None
    
    prog_data = json_data['programming_details']
    nodes = prog_data.get('Nodes', [])
    edges = prog_data.get('Edges', [])
    
    if not nodes or not edges:
        return None, None, None, None
    
    # Extract execution time from scheduling_data
    execution_time = None
    if 'scheduling_data' in json_data and isinstance(json_data['scheduling_data'], list):
        for item in json_data['scheduling_data']:
            if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
                try:
                    execution_time = float(item.get('value'))
                    break
                except (ValueError, TypeError):
                    continue
    if execution_time is None:
        execution_time = 0.0
    
    # Build node features
    node_features = []
    node_index = {}
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict) or 'Name' not in node:
            continue
        name = node['Name']
        node_index[name] = idx
        details = node.get('Details', {})
        
        feature_vector = []
        if 'Memory access patterns' in details:
            mem_patterns = details['Memory access patterns']
            for pattern in mem_patterns:
                if isinstance(pattern, str):
                    values = [float(v) for v in pattern.split() if v.replace('.', '').replace('-', '').isdigit()]
                    feature_vector.extend(values)
        
        if 'Op histogram' in details:
            op_hist = details['Op histogram']
            for op in op_hist:
                if isinstance(op, str):
                    try:
                        value = float(op.split(':')[-1].strip())
                        feature_vector.append(value)
                    except (ValueError, IndexError):
                        continue
        
        if 'scheduling_feature' in details:
            sched = details['scheduling_feature']
            sched_features = [
                sched.get('allocation_bytes_read_per_realization', 0.0),
                sched.get('bytes_at_production', 0.0),
                sched.get('bytes_at_realization', 0.0),
                sched.get('bytes_at_root', 0.0),
                sched.get('bytes_at_task', 0.0),
                sched.get('inlined_calls', 0.0),
                sched.get('inner_parallelism', 0.0),
                sched.get('innermost_bytes_at_production', 0.0),
                sched.get('innermost_bytes_at_realization', 0.0),
                sched.get('innermost_bytes_at_root', 0.0),
                sched.get('innermost_bytes_at_task', 0.0),
                sched.get('innermost_loop_extent', 0.0),
                sched.get('innermost_pure_loop_extent', 0.0),
                sched.get('native_vector_size', 0.0),
                sched.get('num_productions', 0.0),
                sched.get('num_realizations', 0.0),
                sched.get('num_scalars', 0.0),
                sched.get('num_vectors', 0.0),
                sched.get('outer_parallelism', 0.0),
                sched.get('points_computed_minimum', 0.0),
                sched.get('points_computed_per_production', 0.0),
                sched.get('points_computed_per_realization', 0.0),
                sched.get('points_computed_total', 0.0),
                sched.get('scalar_loads_per_scalar', 0.0),
                sched.get('scalar_loads_per_vector', 0.0),
                sched.get('unique_bytes_read_per_realization', 0.0),
                sched.get('unique_lines_read_per_realization', 0.0),
                sched.get('unrolled_loop_extent', 0.0),
                sched.get('vector_loads_per_vector', 0.0),
                sched.get('vector_size', 0.0),
                sched.get('working_set', 0.0),
                sched.get('working_set_at_production', 0.0),
                sched.get('working_set_at_realization', 0.0),
                sched.get('working_set_at_root', 0.0),
                sched.get('working_set_at_task', 0.0),
            ]
            feature_vector.extend(sched_features)
        
        node_features.append(feature_vector)
    
    if not node_features:
        return None, None, None, None
    
    # Pad node features to max length
    max_feature_len = max(len(f) for f in node_features)
    node_features = [f + [0.0] * (max_feature_len - len(f)) for f in node_features]
    node_features = np.array(node_features, dtype=np.float32)
    
    # Build adjacency list (children of each node)
    adj_list = defaultdict(list)
    for edge in edges:
        if not isinstance(edge, dict) or 'From' not in edge or 'To' not in edge:
            continue
        from_node = edge['From']
        to_node = edge['To']
        if from_node in node_index and to_node in node_index:
            adj_list[node_index[to_node]].append(node_index[from_node])
    
    # Compute bottom-up processing order (leaf to root)
    import networkx as nx
    G = nx.DiGraph()
    for node in node_index.values():
        G.add_node(node)
    for edge in edges:
        if edge['From'] in node_index and edge['To'] in node_index:
            G.add_edge(node_index[edge['From']], node_index[edge['To']])
    
    try:
        node_order = list(nx.topological_sort(G))[::-1]  # Reverse for bottom-up
    except nx.NetworkXUnfeasible:
        return None, None, None, None
    
    return node_features, adj_list, node_order, execution_time

def prepare_graph_dataset(synthetic_data_dir):
    """
    Process JSON files to create graph dataset.
    """
    graphs = []
    execution_times = []
    
    json_files = glob.glob(os.path.join(synthetic_data_dir, '**/*.json'), recursive=True)
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            try:
                json_data = json.load(f)
            except json.JSONDecodeError:
                continue
                
        node_features, adj_list, node_order, execution_time = parse_graph_data(json_data)
        if node_features is None:
            continue
        
        graphs.append((node_features, adj_list, node_order))
        execution_times.append(execution_time)
    
    return graphs, np.array(execution_times, dtype=np.float32)

def split_data(graphs, execution_times, test_size=20, val_split=0.2, random_state=42):
    """
    Split data into train, validation, and test sets.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        graphs, execution_times, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=1):
    """
    Create DataLoaders. Batch size=1 due to variable graph sizes.
    """
    train_dataset = GraphDataset(X_train, y_train)
    val_dataset = GraphDataset(X_val, y_val)
    test_dataset = GraphDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, device, epochs=100, patience=10):
    """
    Train the model with early stopping and cosine annealing.
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = 'best_tree_lstm_model.pth'
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in train_loader:
            node_features = batch['node_features'].to(device)
            adj_list = batch['adj_list']
            node_order = batch['node_order']
            y_batch = batch['execution_time'].to(device)
            
            optimizer.zero_grad()
            outputs = model(node_features[0], adj_list[0], node_order[0])
            loss = criterion(outputs, y_batch[0])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_count += 1
        train_loss /= train_count
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                node_features = batch['node_features'].to(device)
                adj_list = batch['adj_list']
                node_order = batch['node_order']
                y_batch = batch['execution_time'].to(device)
                
                outputs = model(node_features[0], adj_list[0], node_order[0])
                loss = criterion(outputs, y_batch[0])
                val_loss += loss.item()
                val_mae += torch.abs(outputs - y_batch[0]).item()
                val_count += 1
        val_loss /= val_count
        val_mae /= val_count
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    model.load_state_dict(torch.load(best_model_path))
    os.remove(best_model_path)
    return train_losses, val_losses

def evaluate_model(model, test_loader, device, scaler):
    """
    Evaluate model and compute error percentages for test set.
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            node_features = batch['node_features'].to(device)
            adj_list = batch['adj_list']
            node_order = batch['node_order']
            y_batch = batch['execution_time'].to(device)
            
            outputs = model(node_features[0], adj_list[0], node_order[0])
            predictions.append(outputs.cpu().numpy())
            targets.append(y_batch.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Inverse transform to original scale
    predictions_orig = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    targets_orig = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()
    
    # Calculate error percentages
    error_percentages = np.abs(predictions_orig - targets_orig) / np.abs(targets_orig + 1e-10) * 100
    mean_error_percentage = np.mean(error_percentages)
    
    # Print predictions and errors
    print("\nTest Set Predictions:")
    print("Sample | Actual Time (ms) | Predicted Time (ms) | Error Percentage (%)")
    print("-" * 60)
    for i, (actual, pred, err) in enumerate(zip(targets_orig, predictions_orig, error_percentages)):
        print(f"{i+1:6d} | {actual:15.4f} | {pred:18.4f} | {err:20.4f}")
    
    return predictions_orig, targets_orig, error_percentages, mean_error_percentage

def plot_loss(train_losses, val_losses, output_file='loss_plot_tree_lstm.png'):
    """
    Plot training and validation loss and save to file.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss (Tree-LSTM)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    plt.close()

def main():
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare dataset
    synthetic_data_dir = "synthetic_data"
    graphs, execution_times = prepare_graph_dataset(synthetic_data_dir)
    print(f"Loaded dataset with {len(graphs)} samples")
    
    # Standardize execution times
    time_scaler = StandardScaler()
    execution_times = time_scaler.fit_transform(execution_times.reshape(-1, 1)).flatten()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(graphs, execution_times)
    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # Initialize model
    input_dim = X_train[0][0].shape[1] if X_train else 1  # Feature dim from first graph
    model = ChildSumTreeLSTM(input_dim=input_dim).to(device)
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, device)
    
    # Evaluate model
    predictions, targets, error_percentages, mean_error_percentage = evaluate_model(
        model, test_loader, device, time_scaler
    )
    print(f"\nMean Error Percentage: {mean_error_percentage:.4f}%")
    
    # Plot loss
    plot_loss(train_losses, val_losses)
    print("Loss plot saved as 'loss_plot_tree_lstm.png'")

if __name__ == "__main__":
    main()
