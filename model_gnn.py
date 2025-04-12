import os
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

# Define the HalideDataset class
class HalideDataset(Dataset):
    def __init__(self, data_list=None, root='data_g'):
        self.data_list = data_list if data_list is not None else []
        super(HalideDataset, self).__init__(root)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.x_scaler = StandardScaler()
        self.y_scaler = None
        self.node_seq_scaler = StandardScaler()
        self.valid_files_cache = None
        self.scaler_cache_path = os.path.join(self.processed_dir, 'scalers.pkl')
        self.metadata_path = os.path.join(root, 'metadata.pkl')
        self.expected_node_dim = None
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                self.expected_node_dim = metadata['node_feature_dim']
        if self.data_list:
            self._fit_scalers()
            self.data_list = [self._normalize_data(data) for data in self.data_list if data.x.shape[0] > 0]
        else:
            self._load_valid_files()
            self._load_or_fit_scalers()
    
    def _fit_scalers(self):
        x_data = []
        node_seq_data = []
        data_source = self.data_list if self.data_list else [torch.load(os.path.join(self.processed_dir, f)) for f in self.valid_files_cache]
        for data in data_source:
            if data.x.shape[0] > 0:
                if self.expected_node_dim and data.x.shape[1] != self.expected_node_dim:
                    print(f"Warning: Graph has {data.x.shape[1]} features, expected {self.expected_node_dim}")
                    continue
                num_nodes = data.x.shape[0]
                num_edges = data.edge_index.shape[1]
                edge_density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
                graph_stats = torch.tensor([num_nodes, num_edges, edge_density], dtype=torch.float).repeat(num_nodes, 1)
                x_augmented = np.concatenate([data.x.numpy(), graph_stats.numpy()], axis=1)
                x_data.append(x_augmented)
                node_seq_data.append(data.node_sequences.numpy().reshape(-1, data.node_sequences.shape[-1]))
        if x_data:
            self.x_scaler.fit(np.vstack(x_data))
            self.node_seq_scaler.fit(np.vstack(node_seq_data))
            try:
                with open(self.scaler_cache_path, 'wb') as f:
                    pickle.dump({
                        'x_scaler': self.x_scaler,
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
                self.node_seq_scaler = scalers['node_seq_scaler']
                if self.expected_node_dim:
                    expected_features = self.expected_node_dim + 3
                    if hasattr(self.x_scaler, 'n_features_in_') and self.x_scaler.n_features_in_ != expected_features:
                        print(f"Scaler expects {self.x_scaler.n_features_in_} features, refitting for {expected_features}")
                        self._fit_scalers()
                return
            except:
                pass
        self._fit_scalers()
    
    def _normalize_data(self, data):
        if self.expected_node_dim and data.x.shape[1] != self.expected_node_dim:
            raise ValueError(f"Graph has {data.x.shape[1]} features, expected {self.expected_node_dim}")
        num_nodes = data.x.shape[0]
        num_edges = data.edge_index.shape[1]
        edge_density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        graph_stats = torch.tensor([num_nodes, num_edges, edge_density], dtype=torch.float).repeat(num_nodes, 1)
        x_augmented = np.concatenate([data.x.numpy(), graph_stats.numpy()], axis=1)
        x = torch.tensor(self.x_scaler.transform(x_augmented), dtype=torch.float)
        y_orig = float(data.y.item() if data.y.dim() > 0 else data.y)
        y_weight = 1.0 / (1.0 + np.log1p(np.abs(y_orig) / 500)) if y_orig > 500 else 1.0
        y = torch.tensor(np.log1p(y_orig), dtype=torch.float).squeeze()
        node_seq = torch.tensor(self.node_seq_scaler.transform(data.node_sequences.numpy().reshape(-1, data.node_sequences.shape[-1])).reshape(data.node_sequences.shape), dtype=torch.float)
        return Data(
            x=x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            y=y,
            node_sequences=node_seq,
            y_weight=torch.tensor(y_weight, dtype=torch.float).squeeze()
        )
    
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
                    if data.x.shape[0] > 0:
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
            if data.x.shape[0] > 0:
                torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))
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

# Define the GNN+LSTM model with edge features
class GNNLSTMModel(nn.Module):
    def __init__(self, node_dim, edge_dim, seq_dim, hidden_dim=512, lstm_layers=3, dropout=0.2):
        super(GNNLSTMModel, self).__init__()
        self.gnn1 = GATConv(node_dim, hidden_dim // 2, heads=8, dropout=dropout, edge_dim=edge_dim)
        self.gnn2 = GATConv((hidden_dim // 2) * 8, hidden_dim, heads=4, dropout=dropout, edge_dim=edge_dim)
        self.gnn3 = GATConv(hidden_dim * 4, hidden_dim, heads=1, dropout=dropout, edge_dim=edge_dim)
        self.bn1 = nn.BatchNorm1d((hidden_dim // 2) * 8)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.lstm = nn.LSTM(seq_dim, hidden_dim // 2, lstm_layers, batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim + hidden_dim // 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual_proj = nn.Linear(node_dim, (hidden_dim // 2) * 8) if node_dim != (hidden_dim // 2) * 8 else None
    
    def forward(self, data):
        x, edge_index, edge_attr, node_sequences, batch = data.x, data.edge_index, data.edge_attr, data.node_sequences, data.batch
        
        if len(x.shape) != 2:
            raise ValueError(f"Expected x to be 2D [num_nodes, node_dim], got shape {x.shape}")
        if len(edge_index.shape) != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"Expected edge_index to be [2, num_edges], got shape {edge_index.shape}")
        
        x_residual = x
        if self.residual_proj:
            x_residual = self.residual_proj(x)
        
        x = self.gnn1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = self.relu(x + x_residual)
        x = self.dropout(x)
        x = self.gnn2(x, edge_index, edge_attr=edge_attr)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.gnn3(x, edge_index, edge_attr=edge_attr)
        x = self.bn3(x)
        gnn_out = self.relu(x)
        
        if len(node_sequences.shape) == 2:
            node_sequences = node_sequences.unsqueeze(0)
        if node_sequences.shape[-1] != self.lstm.input_size:
            node_sequences = node_sequences.transpose(1, 2)
        
        lstm_out, _ = self.lstm(node_sequences)
        lstm_out = lstm_out.squeeze(0)
        lstm_out = self.dropout(lstm_out)
        
        combined = torch.cat([gnn_out, lstm_out], dim=1)
        out = torch_geometric.nn.global_mean_pool(combined, batch)
        out = self.fc(out)
        return out.squeeze(-1)

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

# Learning rate scheduler with warmup
class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, base_lr, final_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.final_lr = final_lr
        super(WarmupLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            progress = self.last_epoch / self.warmup_epochs
            return [self.base_lr + (self.final_lr - self.base_lr) * progress for _ in self.optimizer.param_groups]
        return [self.final_lr for _ in self.optimizer.param_groups]

# Training function
def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=0.0002, patience=15, accum_steps=8):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.HuberLoss(delta=0.3, reduction='none')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=lr * 0.1)
    warmup_scheduler = WarmupLR(optimizer, warmup_epochs=10, base_lr=lr * 0.1, final_lr=lr)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        total_weight = 0.0
        optimizer.zero_grad()
        for i, data in enumerate(train_loader):
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data.y)
            weight = data.y_weight if hasattr(data, 'y_weight') else torch.ones_like(loss)
            weighted_loss = (loss * weight).mean() / accum_steps
            weighted_loss.backward()
            
            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += weighted_loss.item() * accum_steps * data.num_graphs
            total_weight += weight.sum().item()
        
        if len(train_loader) % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss /= max(total_weight, 1e-8)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        total_weight = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.y)
                weight = data.y_weight if hasattr(data, 'y_weight') else torch.ones_like(loss)
                weighted_loss = (loss * weight).mean()
                val_loss += weighted_loss.item() * data.num_graphs
                total_weight += weight.sum().item()
        val_loss /= max(total_weight, 1e-8)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if epoch >= 10:  # Apply cosine annealing after warmup
            scheduler.step()
        warmup_scheduler.step()
        
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
            pred = np.expm1(out.cpu().numpy().item())
            actual = np.expm1(data.y.cpu().numpy().item())
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
    plt.ylabel('Weighted Huber Loss')
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
        raise ValueError(f"Dataset has only {len(dataset)} samples, need at least 20 for testing.")
    
    metadata_path = os.path.join('data_g', 'metadata.pkl')
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    node_dim = metadata['node_feature_dim'] + 3
    edge_dim = metadata['edge_feature_dim']
    seq_dim = metadata['seq_feature_dim']
    
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, num_test=20, val_ratio=0.1)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = GNNLSTMModel(node_dim=node_dim, edge_dim=edge_dim, seq_dim=seq_dim, hidden_dim=512, lstm_layers=3, dropout=0.2).to(device)
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, device, num_epochs=100, lr=0.0002, patience=15, accum_steps=8)
    
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
