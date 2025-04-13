import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import re

# --- Feature Extraction ---

def extract_node_features(node):
    """Extract features from a node."""
    details = node.get('Details', {})
    
    # Memory access patterns
    mem_patterns = details.get('Memory access patterns', [])
    mem_features = []
    for pattern in mem_patterns:
        # Convert pattern values to numbers (assuming they are space-separated)
        values = pattern.split(':')[-1].strip().split()
        mem_features.extend([float(v) for v in values])
    
    # Operation histogram
    op_hist = details.get('Op histogram', [])
    op_features = []
    for op in op_hist:
        # Extract number after colon
        value = float(op.split(':')[-1].strip())
        op_features.append(value)
    
    # Scheduling features (if present)
    sched_features = details.get('scheduling_feature', {})
    sched_values = [float(v) for v in sched_features.values() if isinstance(v, (int, float))]
    
    # Combine all features
    return np.array(mem_features + op_features + sched_values, dtype=np.float32)

def extract_edge_features(edge):
    """Extract features from an edge."""
    details = edge.get('Details', {})
    
    # Load Jacobians
    jacobians = details.get('Load Jacobians', [])
    jacobian_features = []
    for row in jacobians:
        # Parse fractions and numbers
        values = row.strip().split()
        parsed = []
        for v in values:
            if '/' in v:
                num, denom = map(float, v.split('/'))
                parsed.append(num / denom)
            else:
                parsed.append(float(v))
        jacobian_features.extend(parsed)
    
    # Footprint (simplified: count elements)
    footprint = details.get('Footprint', [])
    footprint_features = [len(footprint)]
    
    return np.array(jacobian_features + footprint_features, dtype=np.float32)

def build_tree(nodes, edges):
    """Build a tree structure from nodes and edges."""
    node_dict = {node['Name']: idx for idx, node in enumerate(nodes)}
    node_features = [extract_node_features(node) for node in nodes]
    
    # Initialize adjacency list
    children = [[] for _ in nodes]
    parents = [-1] * len(nodes)
    
    for edge in edges:
        from_node = edge['From']
        to_node = edge['To']
        if from_node in node_dict and to_node in node_dict:
            from_idx = node_dict[from_node]
            to_idx = node_dict[to_node]
            children[from_idx].append(to_idx)
            if parents[to_idx] == -1:  # Assign first parent
                parents[to_idx] = from_idx
    
    # Find root (node with no parent)
    root = next((i for i, p in enumerate(parents) if p == -1), 0)
    
    return {
        'node_features': node_features,
        'children': children,
        'root': root
    }

# --- Dataset Creation ---

class HalideDataset(Dataset):
    def __init__(self, data_dir):
        self.trees = []
        self.labels = []
        
        # Iterate through synthetic_data folder
        for subfolder in tqdm(os.listdir(data_dir), desc="Processing folders"):
            subfolder_path = os.path.join(data_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            for filename in os.listdir(subfolder_path):
                if not filename.endswith('.json'):
                    continue
                file_path = os.path.join(subfolder_path, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract programming details
                    prog_details = data.get('programming_details', {})
                    nodes = prog_details.get('Nodes', [])
                    edges = prog_details.get('Edges', [])
                    
                    # Extract execution time
                    exec_time = None
                    for item in prog_details:
                        if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
                            exec_time = float(item['value'])
                            break
                    if exec_time is None:
                        continue
                    
                    # Build tree
                    tree = build_tree(nodes, edges)
                    self.trees.append(tree)
                    self.labels.append(exec_time)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    def __len__(self):
        return len(self.trees)
    
    def __getitem__(self, idx):
        return {
            'tree': self.trees[idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

# --- TreeLSTM Model ---

class TreeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Gates: input, forget, output, cell
        self.W_iou = nn.Linear(input_size, 3 * hidden_size)
        self.U_iou = nn.Linear(hidden_size, 3 * hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x, h_children, c_children):
        # x: (input_size)
        # h_children: list of (hidden_size)
        # c_children: list of (hidden_size)
        
        batch_size = 1  # Single node processing
        iou = self.W_iou(x) + sum(self.U_iou(h) for h in h_children)
        i, o, u = torch.split(iou, self.hidden_size, dim=-1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        
        f = []
        for h in h_children:
            f.append(torch.sigmoid(self.W_f(x) + self.U_f(h)))
        
        c = i * u + sum(f_k * c_k for f_k, c_k in zip(f, c_children))
        h = o * torch.tanh(c)
        
        return h, c

class TreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeLSTM, self).__init__()
        self.cell = TreeLSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)  # Predict execution time
    
    def forward(self, tree):
        node_features = tree['node_features']
        children = tree['children']
        root = tree['root']
        
        # Initialize hidden and cell states
        h = [torch.zeros(1, self.cell.hidden_size) for _ in node_features]
        c = [torch.zeros(1, self.cell.hidden_size) for _ in node_features]
        visited = set()
        
        def process_node(idx):
            if idx in visited:
                return h[idx], c[idx]
            visited.add(idx)
            
            # Process children
            h_children = []
            c_children = []
            for child_idx in children[idx]:
                h_child, c_child = process_node(child_idx)
                h_children.append(h_child)
                c_children.append(c_child)
            
            # Convert node features to tensor
            x = torch.tensor(node_features[idx], dtype=torch.float32).unsqueeze(0)
            
            # Compute node state
            h[idx], c[idx] = self.cell(x, h_children, c_children)
            return h[idx], c[idx]
        
        # Process from root
        h_root, _ = process_node(root)
        
        # Predict execution time
        output = self.fc(h_root)
        return output.squeeze()

# --- Training ---

def train_model(dataset, input_size, hidden_size=128, num_epochs=50, batch_size=1):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = TreeLSTM(input_size, hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            tree = batch['tree']
            label = batch['label']
            
            optimizer.zero_grad()
            output = model(tree[0])  # Process single tree
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                tree = batch['tree']
                label = batch['label']
                output = model(tree[0])
                loss = criterion(output, label)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

# --- Main Execution ---

if __name__ == "__main__":
    data_dir = "synthetic_data"
    
    # Create dataset
    dataset = HalideDataset(data_dir)
    if len(dataset) == 0:
        print("No valid data found.")
        exit()
    
    # Determine input size (based on first node's features)
    input_size = len(dataset[0]['tree']['node_features'][0])
    
    # Train the model
    train_model(dataset, input_size)
