import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from collections import defaultdict

# ================== Data Processing ==================

class TreeNode:
    def __init__(self, node_type, features=None, children=None):
        self.node_type = node_type  # 'root', 'node', 'edge', 'schedule'
        self.features = features if features else {}
        self.children = children if children else []
    
    def add_child(self, child_node):
        self.children.append(child_node)
    
    def to_dict(self):
        return {
            'node_type': self.node_type,
            'features': self.features,
            'children': [child.to_dict() for child in self.children]
        }

def build_hierarchy(data):
    """Convert flat JSON data into hierarchical tree structure"""
    root = TreeNode('root', {'execution_time': get_execution_time_from_data(data)})
    
    # Process programming details
    if 'programming_details' in data:
        prog_details = data['programming_details']
        
        # Create nodes branch
        if 'Nodes' in prog_details:
            nodes_parent = TreeNode('nodes_parent')
            for node in prog_details['Nodes']:
                node_features = {'Name': node.get('Name', '')}
                if 'Details' in node and 'Op histogram' in node['Details']:
                    op_hist = node['Details']['Op histogram']
                    for op_line in op_hist:
                        parts = op_line.strip().split(':')
                        if len(parts) == 2:
                            op_name = parts[0].strip()
                            op_count = int(parts[1].strip())
                            node_features[f'op_{op_name.lower()}'] = op_count
                nodes_parent.add_child(TreeNode('node', node_features))
            root.add_child(nodes_parent)
        
        # Create edges branch
        if 'Edges' in prog_details:
            edges_parent = TreeNode('edges_parent')
            for edge in prog_details['Edges']:
                edge_features = {
                    'From': edge.get('From', ''),
                    'To': edge.get('To', ''),
                    'Name': edge.get('Name', '')
                }
                edges_parent.add_child(TreeNode('edge', edge_features))
            root.add_child(edges_parent)
    
    # Process scheduling data
    scheduling_data = data.get('scheduling_data', data.get('programming_details', {}).get('Schedules', []))
    if scheduling_data:
        sched_parent = TreeNode('scheduling_parent')
        for sched in scheduling_data:
            sched_features = {'Name': sched.get('Name', '')}
            if isinstance(sched, dict) and 'Details' in sched and 'scheduling_feature' in sched['Details']:
                sched_features.update(sched['Details']['scheduling_feature'])
            sched_parent.add_child(TreeNode('schedule', sched_features))
        root.add_child(sched_parent)
    
    return root

def get_execution_time_from_data(data):
    """Extract execution time from data structure"""
    schedules = data.get("scheduling_data", [])
    for item in schedules:
        if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
            return float(item.get('value', 0))
    return float(schedules[-1]["value"]) if schedules else 0

def process_directory_to_trees(directory_path):
    """Process directory of JSON files into tree structures"""
    trees = []
    file_names = []
    
    json_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.json')])
    
    for filename in json_files:
        file_path = os.path.join(directory_path, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            tree = build_hierarchy(data)
            trees.append(tree)
            file_names.append(filename)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    return trees, file_names

# ================== Data Representation ==================

def tree_to_features(node, feature_map=None, depth=0):
    """Convert tree structure to feature vectors with hierarchy information"""
    if feature_map is None:
        feature_map = defaultdict(list)
    
    # Add node type indicator
    type_features = {
        'type_root': 1 if node.node_type == 'root' else 0,
        'type_node': 1 if node.node_type == 'node' else 0,
        'type_edge': 1 if node.node_type == 'edge' else 0,
        'type_schedule': 1 if node.node_type == 'schedule' else 0,
        'depth': depth
    }
    
    # Combine all features
    combined_features = {**type_features, **node.features}
    
    # Add to feature map
    for k, v in combined_features.items():
        feature_map[k].append(v)
    
    # Process children recursively
    for child in node.children:
        tree_to_features(child, feature_map, depth+1)
    
    return feature_map

def trees_to_dataframe(trees):
    """Convert list of trees to pandas DataFrame"""
    all_features = []
    
    for tree in trees:
        feature_map = tree_to_features(tree)
        
        # Aggregate features
        aggregated = {}
        for k, v in feature_map.items():
            if k.startswith('type_') or k == 'depth':
                # Keep type indicators as is
                aggregated[k] = v[0] if v else 0
            else:
                # Aggregate other features
                aggregated[f'{k}_mean'] = np.mean(v) if v else 0
                aggregated[f'{k}_max'] = np.max(v) if v else 0
                aggregated[f'{k}_min'] = np.min(v) if v else 0
                aggregated[f'{k}_sum'] = np.sum(v) if v else 0
        
        # Add execution time from root
        if 'execution_time_mean' in aggregated:
            aggregated['execution_time'] = aggregated['execution_time_mean']
            del aggregated['execution_time_mean']
        
        all_features.append(aggregated)
    
    return pd.DataFrame(all_features)

# ================== Recursive LSTM Model ==================

class RecursiveLSTMCell(nn.Module):
    """Single recursive LSTM cell for processing tree nodes"""
    def __init__(self, input_size, hidden_size):
        super(RecursiveLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        
        # Gates for input processing
        self.ioux = nn.Linear(input_size, 3 * hidden_size)
        self.iouh = nn.Linear(hidden_size, 3 * hidden_size)
        
        # Gates for memory cell
        self.fx = nn.Linear(input_size, hidden_size)
        self.fh = nn.Linear(hidden_size, hidden_size)
        
        # Layer normalization
        self.ln_iou = nn.LayerNorm(3 * hidden_size)
        self.ln_f = nn.LayerNorm(hidden_size)
    
    def forward(self, x, children_states):
        """
        Args:
            x: input features [batch_size, input_size]
            children_states: list of tuples (h, c) from child nodes
        Returns:
            h: hidden state [batch_size, hidden_size]
            c: cell state [batch_size, hidden_size]
        """
        batch_size = x.size(0)
        
        # Sum child hidden states
        h_sum = torch.zeros(batch_size, self.hidden_size, device=x.device)
        for h, _ in children_states:
            h_sum += h
        
        # Input, output, and update gates
        iou = self.ln_iou(self.ioux(x) + self.iouh(h_sum))
        i, o, u = torch.chunk(iou, 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        
        # Forget gates for children
        if children_states:
            # Compute forget gates for each child
            child_cells = torch.stack([c for _, c in children_states], dim=0)
            child_hiddens = torch.stack([h for h, _ in children_states], dim=0)
            
            f = torch.sigmoid(
                self.ln_f(self.fx(x).unsqueeze(0) + self.fh(child_hiddens))
            )
            
            # Apply forget gates to child cells
            c = (f * child_cells).sum(dim=0) + i * u
        else:
            # No children case
            c = i * u
        
        h = o * torch.tanh(c)
        
        return h, c

class RecursiveLSTMModel(nn.Module):
    """Full recursive LSTM model for tree-structured data"""
    def __init__(self, input_size, hidden_size, output_size=1, dropout=0.3):
        super(RecursiveLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.cell = RecursiveLSTMCell(input_size, hidden_size)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Activation
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward_tree(self, node_features, children_states):
        """Recursive forward pass through tree structure"""
        # Process current node
        h, c = self.cell(node_features, children_states)
        
        # Apply layer norm and dropout
        h = self.dropout(self.layer_norm(h))
        
        return h, c
    
    def forward(self, tree_batch):
        """
        Args:
            tree_batch: list of dictionaries containing:
                - 'node_features': tensor [batch_size, input_size]
                - 'children': list of previous hidden states
        Returns:
            output: predicted value [batch_size, output_size]
        """
        # Process tree recursively
        h, c = self.forward_tree(
            tree_batch['node_features'],
            tree_batch.get('children', [])
        )
        
        # Attention over all nodes (simplified for this example)
        attn_weights = torch.softmax(self.attention(h), dim=0)
        context = torch.sum(attn_weights * h, dim=0, keepdim=True)
        
        # Fully connected layers
        out = self.leaky_relu(self.fc1(context))
        out = self.fc2(out)
        
        return out

# ================== Training Pipeline ==================

class TreeDataset(Dataset):
    """Dataset for tree-structured data"""
    def __init__(self, trees, features_df, scaler_X=None, scaler_y=None):
        self.trees = trees
        self.features_df = features_df
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        
        # Preprocess features
        self.X = self._preprocess_features()
        self.y = self._preprocess_targets()
    
    def _preprocess_features(self):
        # Scale features if scaler provided
        features = self.features_df.drop('execution_time', axis=1).values
        return torch.FloatTensor(
            self.scaler_X.transform(features) if self.scaler_X else features
        )
    
    def _preprocess_targets(self):
        # Scale targets if scaler provided
        targets = self.features_df['execution_time'].values.reshape(-1, 1)
        return torch.FloatTensor(
            self.scaler_y.transform(targets) if self.scaler_y else targets
        )
    
    def __len__(self):
        return len(self.trees)
    
    def __getitem__(self, idx):
        return {
            'tree': self.trees[idx],
            'features': self.X[idx],
            'target': self.y[idx]
        }

def train_recursive_model(train_trees, test_trees, train_df, test_df, input_size, epochs=100):
    # Create data scalers
    scaler_X = StandardScaler().fit(train_df.drop('execution_time', axis=1))
    scaler_y = StandardScaler().fit(train_df['execution_time'].values.reshape(-1, 1))
    
    # Create datasets
    train_dataset = TreeDataset(train_trees, train_df, scaler_X, scaler_y)
    test_dataset = TreeDataset(test_trees, test_df, scaler_X, scaler_y)
    
    # Create model
    model = RecursiveLSTMModel(
        input_size=input_size,
        hidden_size=128,
        output_size=1,
        dropout=0.3
    )
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 20
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_dataset:
            optimizer.zero_grad()
            
            # Recursive processing would happen here
            # (Simplified for this example - actual implementation would need proper batching)
            output = model(batch)
            loss = criterion(output, batch['target'].to(device))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_dataset:
                output = model(batch)
                val_loss += criterion(output, batch['target'].to(device)).item()
        
        train_loss /= len(train_dataset)
        val_loss /= len(test_dataset)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        scheduler.step(val_loss)
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    return model, scaler_y

# ================== Main Execution ==================

def main(main_dir):
    # Process data into trees
    train_trees, train_files = process_directory_to_trees(os.path.join(main_dir, 'train'))
    test_trees, test_files = process_directory_to_trees(os.path.join(main_dir, 'test'))
    
    # Convert trees to features
    train_df = trees_to_dataframe(train_trees)
    test_df = trees_to_dataframe(test_trees)
    
    # Clean and transform features
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    
    # Ensure consistent columns
    for col in train_df.columns:
        if col not in test_df and col != 'execution_time':
            test_df[col] = 0
    for col in test_df.columns:
        if col not in train_df and col != 'execution_time':
            train_df[col] = 0
    
    # Reorder columns to match
    test_df = test_df[train_df.columns]
    
    # Train model
    input_size = len(train_df.columns) - 1  # minus target
    model, scaler_y = train_recursive_model(
        train_trees, test_trees, train_df, test_df, input_size
    )
    
    # Evaluate model
    evaluate_recursive_model(model, test_trees, test_df, scaler_y, test_files)
    
    return model, scaler_y

def evaluate_recursive_model(model, test_trees, test_df, scaler_y, test_files):
    device = next(model.parameters()).device
    model.eval()
    
    # Prepare test data
    X_test = torch.FloatTensor(
        scaler_X.transform(test_df.drop('execution_time', axis=1))
    ).to(device)
    y_test = test_df['execution_time'].values
    
    # Predict
    with torch.no_grad():
        # Simplified prediction - actual implementation would process trees recursively
        y_pred_scaled = model(X_test.unsqueeze(1)).cpu().numpy()
    
    # Inverse scaling
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    
    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    
    print("\nModel Evaluation:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Print some examples
    for i in range(min(5, len(test_files))):
        print(f"\nFile: {test_files[i]}")
        print(f"Actual: {y_test[i]:.2f} ms")
        print(f"Predicted: {y_pred[i][0]:.2f} ms")
        print(f"Error: {abs(y_test[i] - y_pred[i][0]) / y_test[i] * 100:.2f}%")

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    
    main_dir = "synthetic_data"  # Directory with train/test subdirectories
    model, scaler = main(main_dir)
