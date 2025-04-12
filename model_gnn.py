import os
import json
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import scatter
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler

# Define the HalideDataset class with outlier handling
class HalideDataset(Dataset):
    def __init__(self, data_list=None, root='data_g', y_max=1e5):
        self.data_list = data_list if data_list is not None else []
        self.y_max = y_max
        super(HalideDataset, self).__init__(root)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.x_scaler = StandardScaler()
        self.y_scaler = RobustScaler()
        self.node_seq_scaler = StandardScaler()
        self.valid_files_cache = None
        self.scaler_cache_path = os.path.join(self.processed_dir, 'scalers.pkl')
        if self.data_list:
            self._fit_scalers()
            self.data_list = [self._normalize_data(data) for data in self.data_list if data.x.shape[0] > 0 and data.y <= self.y_max]
        else:
            self._load_valid_files()
            self._load_or_fit_scalers()
    
    def _fit_scalers(self):
        x_data = []
        y_data = []
        node_seq_data = []
        data_source = self.data_list if self.data_list else [torch.load(os.path.join(self.processed_dir, f)) for f in self.valid_files_cache]
        for data in data_source:
            if data.x.shape[0] > 0 and data.y <= self.y_max:
                x_data.append(data.x.numpy())
                y_clipped = np.clip(data.y.numpy(), 1e-6, self.y_max)
                y_data.append(np.log1p(y_clipped).reshape(-1, 1))
                node_seq_data.append(data.node_sequences.numpy().reshape(-1, data.node_sequences.shape[-1]))
        if x_data:
            self.x_scaler.fit(np.vstack(x_data))
            self.y_scaler.fit(np.vstack(y_data))
            self.node_seq_scaler.fit(np.vstack(node_seq_data))
            try:
                with open(self.scaler_cache_path, 'wb') as f:
                    pickle.dump({
                        'x_scaler': self.x_scaler,
                        'y_scaler': self.y_scaler,
                        'node_seq_scaler': self.node_seq_scaler
                    }, f)
            except:
                pass
    
    def _load_or_fit_scalers(self):
        if os.path.exists(self.scaler_cache_path):
            try:
                with open(self.scaler_cache_path, 'rb') as f:
                    scalers = pickle.load(f)
                self.x_scaler = scalers['x_scaler']
                self.y_scaler = scalers['y_scaler']
                self.node_seq_scaler = scalers['node_seq_scaler']
                return
            except:
                pass
        self._fit_scalers()
    
    def _normalize_data(self, data):
        x = torch.tensor(self.x_scaler.transform(data.x.numpy()), dtype=torch.float)
        y_clipped = np.clip(data.y.numpy(), 1e-6, self.y_max)
        y = torch.tensor(self.y_scaler.transform(np.log1p(y_clipped).reshape(-1, 1)).flatten(), dtype=torch.float)
        node_seq = torch.tensor(self.node_seq_scaler.transform(data.node_sequences.numpy().reshape(-1, data.node_sequences.shape[-1])).reshape(data.node_sequences.shape), dtype=torch.float)
        return Data(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=y, node_sequences=node_seq)
    
    def _load_valid_files(self):
        cache_path = os.path.join(self.processed_dir, 'valid_files.pkl')
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_files, cached_mtime = pickle.load(f)
                current_files = set(f for f in os.listdir(self.processed_dir) if f.endswith('.pt'))
                if cached_files and all(os.path.getmtime(os.path.join(self.processed_dir, f)) <= cached_mtime for f in cached_files):
                    self.valid_files_cache = sorted(cached_files)
                    return
            except:
                pass
        
        valid_files = []
        for f in sorted(os.listdir(self.processed_dir)):
            if f.endswith('.pt'):
                try:
                    data = torch.load(os.path.join(self.processed_dir, f))
                    if data.x.shape[0] > 0 and data.y <= self.y_max:
                        valid_files.append(f)
                except:
                    continue
        
        self.valid_files_cache = valid_files
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump((valid_files, os.path.getmtime(self.processed_dir)), f)
        except:
            pass
    
    @property
    def processed_file_names(self):
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
            data = self._normalize_data(data)
        if data.x.shape[0] == 0:
            raise ValueError(f"Graph {idx} has empty node features (shape: {data.x.shape})")
        return data
    
    def process(self):
        if not self.data_list:
            return
        for f in os.listdir(self.processed_dir):
            if f.endswith('.pt') or f in ['valid_files.pkl', 'scalers.pkl']:
                os.remove(os.path.join(self.processed_dir, f))
        for i, data in enumerate(self.data_list):
            if data.x.shape[0] > 0 and data.y <= self.y_max:
                torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))
        self.valid_files_cache = [f'data_{i}.pt' for i in range(len(self.data_list)) if self.data_list[i].x.shape[0] > 0 and self.data_list[i].y <= self.y_max]
        try:
            with open(os.path.join(self.processed_dir, 'valid_files.pkl'), 'wb') as f:
                pickle.dump((self.valid_files_cache, os.path.getmtime(self.processed_dir)), f)
        except:
            pass
    
    def _process(self):
        if not self.data_list:
            return
        self.process()

# Define a hybrid loss combining MAE and relative MAE
class HybridMAELoss(nn.Module):
    def __init__(self, alpha=0.7, epsilon=1e-6):
        super(HybridMAELoss, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.mae = nn.L1Loss()
    
    def forward(self, pred, target):
        mae_loss = self.mae(pred, target)
        abs_error = torch.abs(pred - target)
        relative_error = abs_error / (torch.abs(target) + self.epsilon)
        rel_mae = torch.mean(relative_error)
        return self.alpha * mae_loss + (1 - self.alpha) * rel_mae

# Define the GNN+LSTM model with proper batch handling
class GNNLSTMModel(nn.Module):
    def __init__(self, node_dim, edge_dim, seq_dim, hidden_dim=256, lstm_layers=2, dropout=0.3):
        super(GNNLSTMModel, self).__init__()
        self.gnn1 = torch_geometric.nn.GCNConv(node_dim, hidden_dim, node_dim=0)
        self.gnn2 = torch_geometric.nn.GCNConv(hidden_dim, hidden_dim, node_dim=0)
        self.gnn3 = torch_geometric.nn.GCNConv(hidden_dim, hidden_dim, node_dim=0)
        self.lstm = nn.LSTM(seq_dim, hidden_dim // 2, lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim + hidden_dim // 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)
    
    def forward(self, data):
        x, edge_index, edge_attr, node_sequences = data.x, data.edge_index, data.edge_attr, data.node_sequences
        batch = getattr(data, 'batch', None)  # Batch tensor for graph assignment
        
        if len(x.shape) != 2:
            raise ValueError(f"Expected x to be 2D [num_nodes, node_dim], got shape {x.shape}")
        if len(edge_index.shape) != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Expected edge_index to be [2, num_edges], got shape {edge_index.shape}")
        
        x = self.gnn1(x, edge_index)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.gnn2(x, edge_index)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.gnn3(x, edge_index)
        gnn_out = self.relu(x)
        
        if len(node_sequences.shape) == 2:
            node_sequences = node_sequences.unsqueeze(0)
        if node_sequences.shape[-1] != self.lstm.input_size:
            node_sequences = node_sequences.transpose(1, 2)
        
        lstm_out, _ = self.lstm(node_sequences)
        lstm_out = lstm_out.squeeze(0)
        lstm_out = self.bn3(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        combined = torch.cat([gnn_out, lstm_out], dim=1)
        
        # Aggregate per graph using scatter_mean
        if batch is not None:
            combined = scatter(combined, batch, dim=0, reduce='mean')  # [num_graphs, hidden_dim + hidden_dim // 2]
        else:
            combined = combined.mean(dim=0, keepdim=True)  # [1, hidden_dim + hidden_dim // 2]
        
        out = self.fc(combined)  # [num_graphs, 1]
        out = self.relu(out)  # Ensure non-negative
        return out.squeeze(-1)  # [num_graphs]

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
def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=0.0005, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = HybridMAELoss(alpha=0.7, epsilon=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(torch.load('best_model.pt'))
                break
    
    return train_losses, val_losses

# Testing function with inverse scaling
def test_model(model, test_loader, device, y_scaler, y_max=1e5):
    model.eval()
    predictions = []
    actuals = []
    percentage_errors = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            pred_scaled = y_scaler.inverse_transform(out.cpu().numpy().reshape(-1, 1)).flatten()
            pred = np.expm1(pred_scaled)[0]
            pred = np.clip(pred, 0, y_max)
            actual_scaled = y_scaler.inverse_transform(data.y.cpu().numpy().reshape(-1, 1)).flatten()
            actual = np.expm1(actual_scaled)[0]
            actual = np.clip(actual, 0, y_max)
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
    plt.ylabel('Hybrid MAE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# Main execution
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = HalideDataset(root='data_g', y_max=1e5)
    if len(dataset) < 20:
        raise ValueError(f"Dataset has only {len(dataset)} samples, need at least 20 for testing.")
    
    metadata_path = os.path.join('data_g', 'metadata.pkl')
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    node_dim = metadata['node_feature_dim']
    edge_dim = metadata['edge_feature_dim']
    seq_dim = metadata['seq_feature_dim']
    
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, num_test=20, val_ratio=0.1)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = GNNLSTMModel(node_dim=node_dim, edge_dim=edge_dim, seq_dim=seq_dim, hidden_dim=256, dropout=0.3).to(device)
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, device, num_epochs=100, lr=0.0005, patience=20)
    
    plot_losses(train_losses, val_losses, save_path='loss_gnn.png')
    print("Loss plot saved as 'loss_gnn.png'")
    
    predictions, actuals, percentage_errors = test_model(model, test_loader, device, dataset.y_scaler, y_max=1e5)
    
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
