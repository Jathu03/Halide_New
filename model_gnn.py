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

# Define the HalideDataset class (from previous code, repeated for completeness)
class HalideDataset(Dataset):
    def __init__(self, data_list=None, root='data_g'):
        self.data_list = data_list if data_list is not None else []
        super(HalideDataset, self).__init__(root)
        os.makedirs(self.processed_dir, exist_ok=True)
    
    @property
    def processed_file_names(self):
        if self.data_list:
            return [f'data_{i}.pt' for i in range(len(self.data_list))]
        if os.path.exists(self.processed_dir):
            return [f for f in os.listdir(self.processed_dir) if f.endswith('.pt')]
        return []
    
    @property
    def num_graphs(self):
        if self.data_list:
            return len(self.data_list)
        if os.path.exists(self.processed_dir):
            return len([f for f in os.listdir(self.processed_dir) if f.endswith('.pt')])
        return 0
    
    def len(self):
        return self.num_graphs
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    
    def process(self):
        if not self.data_list:
            return
        for i, data in enumerate(self.data_list):
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))
    
    def _process(self):
        if not self.data_list:
            return
        expected_files = set(self.processed_file_names)
        existing_files = set(f for f in os.listdir(self.processed_dir) if f.endswith('.pt'))
        if expected_files == existing_files:
            return
        for f in existing_files:
            os.remove(os.path.join(self.processed_dir, f))
        self.process()

# Define the GNN+LSTM model
class GNNLSTMModel(nn.Module):
    def __init__(self, node_dim, edge_dim, seq_dim, hidden_dim=64, lstm_layers=2):
        super(GNNLSTMModel, self).__init__()
        # GNN layers
        self.gnn1 = torch_geometric.nn.GCNConv(node_dim, hidden_dim)
        self.gnn2 = torch_geometric.nn.GCNConv(hidden_dim, hidden_dim)
        # LSTM layer
        self.lstm = nn.LSTM(seq_dim, hidden_dim // 2, lstm_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim + hidden_dim // 2, 1)
        self.relu = nn.ReLU()
    
    def forward(self, data):
        x, edge_index, edge_attr, node_sequences = data.x, data.edge_index, data.edge_attr, data.node_sequences
        
        # GNN processing
        x = self.gnn1(x, edge_index)
        x = self.relu(x)
        x = self.gnn2(x, edge_index)
        gnn_out = self.relu(x)  # [num_nodes, hidden_dim]
        
        # LSTM processing
        lstm_out, _ = self.lstm(node_sequences)  # [num_nodes, seq_len, hidden_dim // 2]
        lstm_out = lstm_out[:, -1, :]  # Take last output: [num_nodes, hidden_dim // 2]
        
        # Combine GNN and LSTM outputs
        combined = torch.cat([gnn_out, lstm_out], dim=1)  # [num_nodes, hidden_dim + hidden_dim // 2]
        
        # Global pooling (mean) and prediction
        out = combined.mean(dim=0)  # [hidden_dim + hidden_dim // 2]
        out = self.fc(out)  # [1]
        return out

# Function to split dataset
def split_dataset(dataset, num_test=20, val_ratio=0.1):
    total_samples = len(dataset)
    if total_samples < num_test:
        raise ValueError(f"Dataset has only {total_samples} samples, need at least {num_test} for testing.")
    
    num_test = min(num_test, total_samples)  # Ensure we don't exceed dataset size
    num_train_val = total_samples - num_test
    num_val = int(num_train_val * val_ratio)
    num_train = num_train_val - num_val
    
    # Shuffle indices
    indices = np.random.permutation(total_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:num_train + num_val + num_test]
    
    # Create subsets
    train_dataset = torch_geometric.data.Subset(dataset, train_indices)
    val_dataset = torch_geometric.data.Subset(dataset, val_indices)
    test_dataset = torch_geometric.data.Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset

# Training function
def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
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
        
        # Validation
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
            # Calculate percentage error
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
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = HalideDataset(root='data_g')
    if len(dataset) < 20:
        raise ValueError(f"Dataset has only {len(dataset)} samples, need at least 20 for testing.")
    
    # Load metadata for model dimensions
    metadata_path = os.path.join('data_g', 'metadata.pkl')
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    node_dim = metadata['node_feature_dim']
    edge_dim = metadata['edge_feature_dim']
    seq_dim = metadata['seq_feature_dim']
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, num_test=20, val_ratio=0.1)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Initialize model
    model = GNNLSTMModel(node_dim=node_dim, edge_dim=edge_dim, seq_dim=seq_dim, hidden_dim=64).to(device)
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, device, num_epochs=100)
    
    # Plot losses
    plot_losses(train_losses, val_losses, save_path='loss_gnn.png')
    print("Loss plot saved as 'loss_gnn.png'")
    
    # Test model
    predictions, actuals, percentage_errors = test_model(model, test_loader, device)
    
    # Print test results
    print("\nTest Sample Predictions:")
    print("Sample | Predicted (ms) | Actual (ms) | Percentage Error (%)")
    print("-" * 60)
    for i, (pred, actual, perc_error) in enumerate(zip(predictions, actuals, percentage_errors)):
        print(f"{i+1:5d} | {pred:13.4f} | {actual:11.4f} | {perc_error:19.4f}")
    
    # Calculate and print average percentage error
    valid_percentage_errors = [pe for pe in percentage_errors if pe != float('inf')]
    if valid_percentage_errors:
        avg_percentage_error = np.mean(valid_percentage_errors)
        print(f"\nAverage Percentage Error (excluding inf): {avg_percentage_error:.4f}%")
    else:
        print("\nNo valid percentage errors to average.")

if __name__ == "__main__":
    main()
