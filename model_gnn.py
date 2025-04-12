import os
import json
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_absolute_error

# Define the HalideDataset class
class HalideDataset(Dataset):
    def __init__(self, data_list=None, root='data_g'):
        self.data_list = data_list if data_list is not None else []
        super(HalideDataset, self).__init__(root)
        os.makedirs(self.processed_dir, exist_ok=True)
        # Filter data_list if provided
        if self.data_list:
            self.data_list = [data for data in self.data_list if data.x.shape[0] > 0]
        # Initialize valid file cache
        self.valid_files_cache = None
        self._load_valid_files()
    
    def _load_valid_files(self):
        """Load or generate a list of valid .pt files."""
        cache_path = os.path.join(self.processed_dir, 'valid_files.pkl')
        # Check if cache exists and matches current files
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_files, cached_mtime = pickle.load(f)
                # Verify cache is up-to-date
                current_files = set(f for f in os.listdir(self.processed_dir) if f.endswith('.pt'))
                if cached_files and all(os.path.getmtime(os.path.join(self.processed_dir, f)) <= cached_mtime for f in cached_files):
                    self.valid_files_cache = sorted(cached_files)
                    return
            except:
                pass
        
        # Generate valid files list
        valid_files = []
        for f in sorted(os.listdir(self.processed_dir)):
            if f.endswith('.pt'):
                try:
                    data = torch.load(os.path.join(self.processed_dir, f))
                    if data.x.shape[0] > 0:
                        valid_files.append(f)
                except:
                    continue
        
        self.valid_files_cache = valid_files
        # Save cache with current timestamp
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump((valid_files, os.path.getmtime(self.processed_dir)), f)
        except:
            pass
    
    @property
    def processed_file_names(self):
        """Return cached list of valid .pt files."""
        if self.data_list:
            return [f'data_{i}.pt' for i in range(len(self.data_list))]
        return self.valid_files_cache
    
    @property
    def num_graphs(self):
        if self.data_list:
            return len(self.data_list)
        return len(self.processed_file_names)
    
    def len(self):
        return self.num_graphs
    
    def get(self, idx):
        if self.data_list:
            data = self.data_list[idx]
        else:
            valid_files = self.processed_file_names
            if idx >= len(valid_files):
                raise IndexError(f"Index {idx} out of range for {len(valid_files)} valid graphs")
            data = torch.load(os.path.join(self.processed_dir, valid_files[idx]))
        # Validate (should be redundant due to filtering)
        if data.x.shape[0] == 0:
            raise ValueError(f"Graph {idx} has empty node features (shape: {data.x.shape})")
        return data
    
    def process(self):
        if not self.data_list:
            return
        # Clear existing files
        for f in os.listdir(self.processed_dir):
            if f.endswith('.pt') or f == 'valid_files.pkl':
                os.remove(os.path.join(self.processed_dir, f))
        # Save valid graphs
        for i, data in enumerate(self.data_list):
            if data.x.shape[0] > 0:
                torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))
        # Update cache
        self.valid_files_cache = [f'data_{i}.pt' for i in range(len(self.data_list)) if self.data_list[i].x.shape[0] > 0]
        try:
            with open(os.path.join(self.processed_dir, 'valid_files.pkl'), 'wb') as f:
                pickle.dump((self.valid_files_cache, os.path.getmtime(self.processed_dir)), f)
        except:
            pass
    
    def _process(self):
        if not self.data_list:
            return
        self.process()

# Define the GNN+LSTM model
class GNNLSTMModel(nn.Module):
    def __init__(self, node_dim, edge_dim, seq_dim, hidden_dim=64, lstm_layers=2):
        super(GNNLSTMModel, self).__init__()
        self.gnn1 = torch_geometric.nn.GCNConv(node_dim, hidden_dim, node_dim=0)
        self.gnn2 = torch_geometric.nn.GCNConv(hidden_dim, hidden_dim, node_dim=0)
        self.lstm = nn.LSTM(seq_dim, hidden_dim // 2, lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim + hidden_dim // 2, 1)
        self.relu = nn.ReLU()
    
    def forward(self, data):
        x, edge_index, edge_attr, node_sequences = data.x, data.edge_index, data.edge_attr, data.node_sequences
        
        if len(x.shape) != 2:
            raise ValueError(f"Expected x to be 2D [num_nodes, node_dim], got shape {x.shape}")
        if len(edge_index.shape) != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Expected edge_index to be [2, num_edges], got shape {edge_index.shape}")
        
        x = self.gnn1(x, edge_index)
        x = self.relu(x)
        x = self.gnn2(x, edge_index)
        gnn_out = self.relu(x)
        
        if len(node_sequences.shape) == 2:
            node_sequences = node_sequences.unsqueeze(0)
        if node_sequences.shape[-1] != self.lstm.input_size:
            node_sequences = node_sequences.transpose(1, 2)
        
        lstm_out, _ = self.lstm(node_sequences)
        lstm_out = lstm_out.squeeze(0)
        
        combined = torch.cat([gnn_out, lstm_out], dim=1)
        out = combined.mean(dim=0)
        out = self.fc(out)
        return out

# Function to split dataset
def split_dataset(dataset, num_test=20, val_ratio=0.1):
    total_samples = len(dataset)
    if total_samples < num_test:
        raise ValueError(f"Dataset has only {total_samples} samples, need at least {num_test} for testing.")
    
    num_test = min(num_test, total_samples)
    num_train_val = total_samples - num_test
    num_val = int(num_train_val * val_ratio)
    num_train = num_train_val - num_val
    
    indices = np.random.permutation(total_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:num_train + num_val + num_test]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset

# Training function
def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.num_graphs
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.y)
                val_loss += loss.item() * data.num_graphs
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return train_losses, val_losses

# Testing function
def test_model(model, test_loader, device):
    model.eval()
    predictions = []
    actuals = []
    percentage_errors = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            pred = out.cpu().numpy().flatten()[0]
            actual = data.y.cpu().numpy().flatten()[0]
            predictions.append(pred)
            actuals.append(actual)
            if actual != 0:
                perc_error = abs(pred - actual) / actual * 100
            else:
                perc_error = float('inf') if pred != 0 else 0.0
            percentage_errors.append(perc_error)
    
    return predictions, actuals, percentage_errors

# Plotting function
def plot_losses(train_losses, val_losses, save_path='loss_gnn.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# Main execution
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = HalideDataset(root='data_g')
    if len(dataset) < 20:
        raise ValueError(f"Dataset has only {len(dataset)} samples, need at least {num_test} for testing.")
    
    metadata_path = os.path.join('data_g', 'metadata.pkl')
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    node_dim = metadata['node_feature_dim']
    edge_dim = metadata['edge_feature_dim']
    seq_dim = metadata['seq_feature_dim']
    
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, num_test=20, val_ratio=0.1)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = GNNLSTMModel(node_dim=node_dim, edge_dim=edge_dim, seq_dim=seq_dim, hidden_dim=64).to(device)
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, device, num_epochs=100)
    
    plot_losses(train_losses, val_losses, save_path='loss_gnn.png')
    print("Loss plot saved as 'loss_gnn.png'")
    
    predictions, actuals, percentage_errors = test_model(model, test_loader, device)
    
    print("\nTest Sample Predictions:")
    print("Sample | Predicted (ms) | Actual (ms) | Percentage Error (%)")
    print("-" * 60)
    for i, (pred, actual, perc_error) in enumerate(zip(predictions, actuals, percentage_errors)):
        print(f"{i+1:5d} | {pred:13.4f} | {actual:11.4f} | {perc_error:19.4f}")
    
    valid_percentage_errors = [pe for pe in percentage_errors if pe != float('inf')]
    if valid_percentage_errors:
        avg_percentage_error = np.mean(valid_percentage_errors)
        print(f"\nAverage Percentage Error (excluding inf): {avg_percentage_error:.4f}%")
    else:
        print("\nNo valid percentage errors to average.")

if __name__ == "__main__":
    main()
