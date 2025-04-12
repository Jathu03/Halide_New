import os
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool, global_max_pool
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from torch.nn import LayerNorm
import random
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Dataset class with fixed feature dimensions
class HalideDataset(Dataset):
    def __init__(self, data_list=None, root='data_g', fixed_node_dim=None):
        self.data_list = data_list if data_list is not None else []
        self.fixed_node_dim = fixed_node_dim  # Will be set from metadata
        super(HalideDataset, self).__init__(root)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.x_scaler = StandardScaler()
        self.node_seq_scaler = StandardScaler()
        self.valid_files_cache = None
        self.scaler_cache_path = os.path.join(self.processed_dir, 'scalers.pkl')
        # Load metadata to set fixed_node_dim
        metadata_path = os.path.join(root, 'metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            self.fixed_node_dim = metadata.get('node_feature_dim', fixed_node_dim)
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
                x = data.x.numpy()
                # Pad or truncate x to fixed_node_dim
                if self.fixed_node_dim:
                    current_dim = x.shape[1]
                    if current_dim < self.fixed_node_dim:
                        x = np.pad(x, ((0, 0), (0, self.fixed_node_dim - current_dim)), mode='constant')
                    elif current_dim > self.fixed_node_dim:
                        x = x[:, :self.fixed_node_dim]
                x_data.append(x)
                node_seq = data.node_sequences.numpy().reshape(-1, data.node_sequences.shape[-1])
                node_seq_data.append(node_seq)
        if x_data:
            self.x_scaler.fit(np.vstack(x_data))
            self.node_seq_scaler.fit(np.vstack(node_seq_data))
            with open(self.scaler_cache_path, 'wb') as f:
                pickle.dump({'x_scaler': self.x_scaler, 'node_seq_scaler': self.node_seq_scaler}, f)

    def _load_or_fit_scalers(self):
        if os.path.exists(self.scaler_cache_path):
            try:
                with open(self.scaler_cache_path, 'rb') as f:
                    scalers = pickle.load(f)
                self.x_scaler = scalers['x_scaler']
                self.node_seq_scaler = scalers['node_seq_scaler']
                return
            except:
                pass
        self._fit_scalers()

    def _normalize_data(self, data):
        x = data.x.numpy()
        # Pad or truncate x to fixed_node_dim
        if self.fixed_node_dim:
            current_dim = x.shape[1]
            if current_dim < self.fixed_node_dim:
                x = np.pad(x, ((0, 0), (0, self.fixed_node_dim - current_dim)), mode='constant')
            elif current_dim > self.fixed_node_dim:
                x = x[:, :self.fixed_node_dim]
        x = torch.tensor(self.x_scaler.transform(x), dtype=torch.float)
        y_val = data.y.numpy()
        y_val = max(y_val, 1e-6)
        y = torch.tensor(np.log1p(y_val), dtype=torch.float)
        node_seq_np = data.node_sequences.numpy()
        original_shape = node_seq_np.shape
        flattened = node_seq_np.reshape(-1, original_shape[-1])
        transformed = self.node_seq_scaler.transform(flattened)
        node_seq = torch.tensor(transformed.reshape(original_shape), dtype=torch.float)
        return Data(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=y, node_sequences=node_seq, num_nodes=data.x.shape[0])

    def _load_valid_files(self):
        cache_path = os.path.join(self.processed_dir, 'valid_files.pkl')
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_files, _ = pickle.load(f)
                self.valid_files_cache = cached_files
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
        with open(cache_path, 'wb') as f:
            pickle.dump((valid_files, os.path.getmtime(self.processed_dir)), f)

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.data_list))] if self.data_list else self.valid_files_cache

    def len(self):
        return len(self.data_list) if self.data_list else len(self.processed_file_names)

    def get(self, idx):
        if self.data_list:
            return self.data_list[idx]
        data = torch.load(os.path.join(self.processed_dir, self.processed_file_names[idx]))
        return self._normalize_data(data)

# GNN+LSTM model
class EnhancedGNNLSTMModel(nn.Module):
    def __init__(self, node_dim, edge_dim, seq_dim, hidden_dim=512, lstm_layers=3, dropout=0.2, heads=4, num_gnn_layers=3, pooling_type='mean'):
        super(EnhancedGNNLSTMModel, self).__init__()
        self.num_gnn_layers = num_gnn_layers
        self.pooling_type = pooling_type
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            in_channels = node_dim if i == 0 else hidden_dim
            self.gnn_layers.append(GATv2Conv(in_channels, hidden_dim // heads, heads=heads, edge_dim=edge_dim))
        self.norms = nn.ModuleList([LayerNorm(hidden_dim) for _ in range(num_gnn_layers)])
        self.lstm = nn.LSTM(seq_dim, hidden_dim // 2, lstm_layers, batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0)
        self.attention_gnn = nn.Linear(hidden_dim, 1)
        self.attention_lstm = nn.Linear(hidden_dim, 1)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()

    def forward(self, data):
        x, edge_index, edge_attr, node_sequences = data.x, data.edge_index, data.edge_attr, data.node_sequences
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.gnn_layers[0](x, edge_index, edge_attr)
        x = self.norms[0](x)
        x = self.elu(x)
        x = self.dropout(x)
        for i in range(1, self.num_gnn_layers):
            x_res = x
            x = self.gnn_layers[i](x, edge_index, edge_attr)
            x = self.norms[i](x)
            x = self.elu(x)
            x = self.dropout(x)
            x = x + x_res
        pooled_gnn = {'mean': global_mean_pool, 'add': global_add_pool, 'max': global_max_pool}[self.pooling_type](x, batch)
        if len(node_sequences.shape) == 2:
            node_sequences = node_sequences.unsqueeze(0)
        if node_sequences.shape[-1] != self.lstm.input_size:
            node_sequences = node_sequences.transpose(1, 2)
        lstm_out, _ = self.lstm(node_sequences)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        gnn_attn = torch.sigmoid(self.attention_gnn(pooled_gnn))
        lstm_attn = torch.sigmoid(self.attention_lstm(lstm_out))
        attn_sum = gnn_attn + lstm_attn
        gnn_weighted = pooled_gnn * (gnn_attn / attn_sum)
        lstm_weighted = lstm_out * (lstm_attn / attn_sum)
        combined = torch.cat([gnn_weighted, lstm_weighted], dim=1)
        h1 = self.fc1(combined)
        h1 = self.elu(h1)
        h1 = self.dropout(h1)
        h2 = self.fc2(h1)
        h2 = self.elu(h2)
        h2 = self.dropout(h2)
        return self.fc3(h2)

# Dataset splitting
def split_dataset(dataset, num_test=20, val_ratio=0.1):
    total_samples = len(dataset)
    num_test = min(num_test, total_samples)
    num_train_val = total_samples - num_test
    num_val = int(num_train_val * val_ratio)
    num_train = num_train_val - num_val
    indices = np.random.permutation(total_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:num_train + num_val]
    test_indices = indices[num_train + num_val:]
    return (torch.utils.data.Subset(dataset, train_indices),
            torch.utils.data.Subset(dataset, val_indices),
            torch.utils.data.Subset(dataset, test_indices))

# Training function
def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=0.001, patience=25):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = lambda pred, target: 0.7 * nn.HuberLoss(delta=1.0)(pred, target) + 0.3 * nn.MSELoss()(pred, target)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=lr/20)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for i, data in enumerate(train_loader):
            data = data.to(device)
            with torch.cuda.amp.autocast() if scaler else torch.no_grad():
                out = model(data)
                loss = criterion(out, data.y) / 2
            if scaler:
                scaler.scale(loss).backward()
                if (i + 1) % 2 == 0 or i == len(train_loader) - 1:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (i + 1) % 2 == 0 or i == len(train_loader) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            train_loss += loss.item() * 2 * data.num_graphs
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                val_loss += criterion(out, data.y).item() * data.num_graphs
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({'state_dict': model.state_dict(), 'config': model.__dict__.get('config', {})}, f'best_model_config{epoch}.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    return train_losses, val_losses

# Testing function
def test_model(model, test_loader, device):
    model.eval()
    predictions, actuals, percentage_errors = [], [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            pred = np.expm1(out.cpu().numpy().flatten())[0]
            actual = np.expm1(data.y.cpu().numpy().flatten())[0]
            predictions.append(pred)
            actuals.append(actual)
            perc_error = abs(pred - actual) / actual * 100 if actual != 0 else (float('inf') if pred != 0 else 0.0)
            percentage_errors.append(perc_error)
    return predictions, actuals, percentage_errors

# Ensemble prediction
def predict_with_ensemble(models, data, device):
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(data.to(device))
            pred = np.expm1(out.cpu().numpy().flatten())[0]
            predictions.append(pred)
    return np.median(predictions)

# Plotting function
def plot_losses(train_losses, val_losses, save_path='loss_gnn.png'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# Main execution
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = HalideDataset(root='data_g')
    with open(os.path.join('data_g', 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    node_dim, edge_dim, seq_dim = metadata['node_feature_dim'], metadata['edge_feature_dim'], metadata['seq_feature_dim']
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    configs = [
        {'num_gnn_layers': 3, 'hidden_dim': 512, 'pooling_type': 'mean'},
        {'num_gnn_layers': 4, 'hidden_dim': 256, 'pooling_type': 'add'},
        {'num_gnn_layers': 2, 'hidden_dim': 768, 'pooling_type': 'max'}
    ]
    models = []
    for idx, config in enumerate(configs):
        torch.manual_seed(42 + idx)
        np.random.seed(42 + idx)
        random.seed(42 + idx)
        model = EnhancedGNNLSTMModel(
            node_dim=node_dim, edge_dim=edge_dim, seq_dim=seq_dim,
            hidden_dim=config['hidden_dim'], num_gnn_layers=config['num_gnn_layers'],
            pooling_type=config['pooling_type'], heads=4, lstm_layers=3, dropout=0.2
        ).to(device)
        model.config = config
        train_losses, val_losses = train_model(model, train_loader, val_loader, device)
        plot_losses(train_losses, val_losses, f'loss_gnn_config{idx}.png')
        torch.save({'state_dict': model.state_dict(), 'config': config}, f'best_model_config{idx}.pt')
        models.append(model)
    for idx in range(len(configs)):
        checkpoint = torch.load(f'best_model_config{idx}.pt')
        config = checkpoint['config']
        model = EnhancedGNNLSTMModel(
            node_dim=node_dim, edge_dim=edge_dim, seq_dim=seq_dim,
            hidden_dim=config['hidden_dim'], num_gnn_layers=config['num_gnn_layers'],
            pooling_type=config['pooling_type'], heads=4, lstm_layers=3, dropout=0.2
        ).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        models[idx] = model
    predictions, actuals = [], []
    for data in test_loader:
        pred = predict_with_ensemble(models, data, device)
        actual = np.expm1(data.y.cpu().numpy().flatten())[0]
        predictions.append(pred)
        actuals.append(actual)
    percentage_errors = [abs(pred - actual) / actual * 100 if actual != 0 else (float('inf') if pred != 0 else 0.0)
                        for pred, actual in zip(predictions, actuals)]
    print("\nTest Sample Predictions:")
    print("Sample | Predicted (ms) | Actual (ms) | Percentage Error (%)")
    print("-" * 60)
    for i, (pred, actual, perc_error) in enumerate(zip(predictions, actuals, percentage_errors)):
        print(f"{i+1:5d} | {pred:13.4f} | {actual:11.4f} | {perc_error:19.4f}")
    valid_errors = [pe for pe in percentage_errors if pe != float('inf')]
    avg_error = np.mean(valid_errors) if valid_errors else float('nan')
    print(f"\nAverage Percentage Error: {avg_error:.4f}%")

if __name__ == "__main__":
    main()
