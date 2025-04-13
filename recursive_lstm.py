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
        try:
            values = pattern.split(':')[-1].strip().split()
            mem_features.extend([float(v) for v in values if v.replace('.', '', 1).isdigit()])
        except Exception as e:
            print(f"Error parsing memory pattern {pattern}: {e}")
            mem_features.extend([0.0] * 4)
    
    # Operation histogram
    op_hist = details.get('Op histogram', [])
    op_features = []
    for op in op_hist:
        try:
            value = float(op.split(':')[-1].strip())
            op_features.append(value)
        except Exception as e:
            print(f"Error parsing op histogram {op}: {e}")
            op_features.append(0.0)
    
    # Scheduling features
    sched_features = details.get('scheduling_feature', {})
    sched_values = []
    for v in sched_features.values():
        try:
            sched_values.append(float(v))
        except (TypeError, ValueError):
            sched_values.append(0.0)
    
    # Combine all features
    feature_vector = mem_features + op_features + sched_values
    if not feature_vector:
        feature_vector = [0.0]
    return np.array(feature_vector, dtype=np.float32)

def extract_edge_features(edge):
    """Extract features from an edge."""
    details = edge.get('Details', {})
    
    # Load Jacobians
    jacobians = details.get('Load Jacobians', [])
    jacobian_features = []
    for row in jacobians:
        try:
            values = row.strip().split()
            parsed = []
            for v in values:
                if '/' in v:
                    try:
                        num, denom = map(float, v.split('/'))
                        parsed.append(num / denom)
                    except:
                        parsed.append(0.0)
                else:
                    try:
                        parsed.append(float(v))
                    except:
                        parsed.append(0.0)
            jacobian_features.extend(parsed)
        except Exception as e:
            print(f"Error parsing Jacobian {row}: {e}")
            jacobian_features.extend([0.0] * 3)
    
    # Footprint
    footprint = details.get('Footprint', [])
    footprint_features = [float(len(footprint))]
    
    return np.array(jacobian_features + footprint_features, dtype=np.float32)

def build_tree(nodes, edges):
    """Build a tree structure from nodes and edges."""
    if not nodes:
        return None
    
    node_dict = {node['Name']: idx for idx, node in enumerate(nodes)}
    node_features = [extract_node_features(node) for node in nodes]
    
    # Ensure all feature vectors have the same length
    max_len = max(len(f) for f in node_features)
    node_features = [np.pad(f, (0, max_len - len(f)), mode='constant') for f in node_features]
    
    # Initialize adjacency list
    children = [[] for _ in nodes]
    parents = [-1] * len(nodes)
    
    for edge in edges:
        from_node = edge.get('From')
        to_node = edge.get('To')
        if from_node in node_dict and to_node in node_dict:
            from_idx = node_dict[from_node]
            to_idx = node_dict[to_node]
            children[from_idx].append(to_idx)
            if parents[to_idx] == -1:
                parents[to_idx] = from_idx
    
    # Find root
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
        
        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} does not exist.")
            return
        
        for subfolder in tqdm(os.listdir(data_dir), desc="Processing folders"):
            subfolder_path = os.path.join(data_dir, subfolder)
            if not os.path.isdir(subfolder_path):
                print(f"Skipping {subfolder_path}: not a directory")
                continue
            for filename in os.listdir(subfolder_path):
                if not filename.endswith('.json'):
                    print(f"Skipping {filename}: not a JSON file")
                    continue
                file_path = os.path.join(subfolder_path, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    prog_details = data.get('programming_details', [])
                    if not prog_details:
                        print(f"No programming_details in {file_path}")
                        continue
                    
                    nodes = []
                    edges = []
                    exec_time = None
                    
                    if isinstance(prog_details, dict):
                        nodes = prog_details.get('Nodes', [])
                        edges = prog_details.get('Edges', [])
                        for key, value in prog_details.items():
                            if key == 'total_execution_time_ms' or (isinstance(value, dict) and value.get('name') == 'total_execution_time_ms'):
                                exec_time = float(value.get('value', 0.0)) if isinstance(value, dict) else float(value)
                                break
                    elif isinstance(prog_details, list):
                        for item in prog_details:
                            if isinstance(item, dict):
                                if item.get('Nodes'):
                                    nodes = item['Nodes']
                                if item.get('Edges'):
                                    edges = item['Edges']
                                if item.get('name') == 'total_execution_time_ms'):
                                    exec_time = float(item.get('value', 0.0))
                    
                    if not nodes or not edges:
                        print(f"No nodes or edges in {file_path}")
                        continue
                    if exec_time is None:
                        print(f"No execution time found in {file_path}")
                        continue
                    
                    tree = build_tree(nodes, edges)
                    if tree is None:
                        print(f"Failed to build tree for {file_path}")
                        continue
                    
                    self.trees.append(tree)
                    self.labels.append(exec_time)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        if not self.trees:
            print("No valid trees were created. Check JSON file structure.")
    
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
        
        self.W_iou = nn.Linear(input_size, 3 * hidden_size)
        self.U_iou = nn.Linear(hidden_size, 3 * hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x, h_children, c_children):
        batch_size = 1
        iou = self.W_iou(x) + sum(self.U_iou(h) for h in h_children)
        i, o, u = torch.split(iou, self.hidden_size, dim=-1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        
        f = [torch.sigmoid(self.W_f(x) + self.U_f(h)) for h in h_children]
        c = i * u + sum(f_k * c_k for f_k, c_k in zip(f, c_children))
        h = o * torch.tanh(c)
        
        return h, c

class TreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeLSTM, self).__init__()
        self.cell = TreeLSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, tree):
        node_features = tree['node_features']
        children = tree['children']
        root = tree['root']
        
        h = [torch.zeros(1, self.cell.hidden_size) for _ in node_features]
        c = [torch.zeros(1, self.cell.hidden_size) for _ in node_features]
        visited = set()
        
        def process_node(idx):
            if idx in visited:
                return h[idx], c[idx]
            visited.add(idx)
            
            h_children = []
            c_children = []
            for child_idx in children[idx]:
                h_child, c_child = process_node(child_idx)
                h_children.append(h_child)
                c_children.append(c_child)
            
            x = torch.tensor(node_features[idx], dtype=torch.float32).unsqueeze(0)
            h[idx], c[idx] = self.cell(x, h_children, c_children)
            return h[idx], c[idx]
        
        h_root, _ = process_node(root)
        output = self.fc(h_root)
        return output.squeeze()

# --- Training ---

def train_model(dataset, input_size, hidden_size=128, num_epochs=50, batch_size=1):
    if len(dataset) == 0:
        print("Empty dataset. Cannot train model.")
        return
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = TreeLSTM(input_size, hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            tree = batch['tree']
            label = batch['label']
            
            optimizer.zero_grad()
            output = model(tree[0])
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
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
        print("No valid data found. Exiting.")
        exit()
    
    # Determine input size
    input_size = len(dataset[0]['tree']['node_features'][0])
    
    # Train the model
    train_model(dataset, input_size)
