import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class HalideTreeDataset(Dataset):
    def __init__(self, root_dir, scaler=None):
        self.root_dir = root_dir
        self.samples = []
        self.scaler = scaler or StandardScaler()
        self._preprocess_data()
        
    def _load_tree(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract execution time (skip invalid entries)
        exec_time = data.get("Global Features", {}).get("execution_time_ms", -1)
        if exec_time <= 0 or exec_time > 1e6:  # Filter invalid times
            return None, None
            
        # Recursive tree parsing
        def parse_node(node):
            features = []
            # Extract important features from different sections
            features += list(node.get("op_histogram", {}).values())
            features += list(node.get("memory_patterns", {}).values())
            features += list(node.get("scheduling", {}).values())
            
            # Convert to floats and handle missing values
            features = [float(x) for x in features if x is not None]
            
            # Process children recursively
            children = [parse_node(child) for child in node.get("children", [])]
            return {"features": features, "children": children}
        
        tree = parse_node(data)
        return tree, exec_time

    def _preprocess_data(self):
        all_features = []
        valid_samples = []
        
        for program_dir in os.listdir(self.root_dir):
            program_path = os.path.join(self.root_dir, program_dir)
            if not os.path.isdir(program_path):
                continue
                
            for schedule_dir in os.listdir(program_path):
                json_path = os.path.join(program_path, schedule_dir, "tree_representation.json")
                if not os.path.exists(json_path):
                    continue
                
                tree, exec_time = self._load_tree(json_path)
                if tree is not None:
                    self._collect_features(tree, all_features)
                    valid_samples.append((tree, exec_time))
        
        # Fit scaler on collected features
        if not self.scaler.fit_:
            self.scaler.fit(np.array(all_features))
        
        self.samples = valid_samples

    def _collect_features(self, node, collector):
        collector.append(node["features"])
        for child in node["children"]:
            self._collect_features(child, collector)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tree, exec_time = self.samples[idx]
        scaled_tree = self._scale_tree(tree)
        return scaled_tree, torch.tensor(exec_time, dtype=torch.float32)

    def _scale_tree(self, node):
        scaled_features = self.scaler.transform([node["features"]])[0]
        scaled_children = [self._scale_tree(child) for child in node["children"]]
        return {
            "features": torch.tensor(scaled_features, dtype=torch.float32),
            "children": scaled_children
        }

class RecursiveLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, tree):
        # Process children recursively
        child_embeddings = [self.forward(child) for child in tree["children"]]
        
        if len(child_embeddings) > 0:
            children = torch.stack(child_embeddings)
            _, (hidden, _) = self.lstm(children)
        else:
            hidden = torch.zeros(1, 1, self.hidden_size)
        
        # Combine node features with children embeddings
        node_features = tree["features"].unsqueeze(0)
        combined = torch.cat([node_features, hidden], dim=-1)
        
        # Final prediction layers
        return self.fc(combined).squeeze()

def train_model():
    # Configuration
    root_dir = "Tree_Output"
    batch_size = 32
    hidden_size = 128
    epochs = 100
    
    # Load dataset
    dataset = HalideTreeDataset(root_dir)
    train_data, val_data = train_test_split(dataset, test_size=0.2)
    
    # Create model
    input_size = len(dataset.scaler.scale_)
    model = RecursiveLSTM(input_size, hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        for tree, target in DataLoader(train_data, batch_size=batch_size):
            optimizer.zero_grad()
            output = model(tree)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for tree, target in DataLoader(val_data, batch_size=batch_size):
                output = model(tree)
                val_loss += criterion(output, target).item()
            print(f"Epoch {epoch+1}, Val Loss: {val_loss/len(val_data):.4f}")

    # Save artifacts
    torch.save(model.state_dict(), "halide_lstm.pth")
    torch.save(dataset.scaler, "scaler.pth")

if __name__ == "__main__":
    train_model()
