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
from sklearn.preprocessing import RobustScaler
from torch_geometric.nn import GATConv

# Custom Dataset with Enhanced Preprocessing
class HalideDataset(Dataset):
    def __init__(self, data_list=None, root='data_g'):
        self.data_list = data_list if data_list is not None else []
        super(HalideDataset, self).__init__(root)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.x_scaler = RobustScaler()
        self.node_seq_scaler = RobustScaler()
        self.scaler_cache_path = os.path.join(self.processed_dir, 'scalers.pkl')
        if self.data_list:
            self._fit_scalers()
            self.data_list = [self._normalize_data(data) for data in self.data_list if data.x.shape[0] > 0]
        else:
            self._load_or_fit_scalers()

    def _fit_scalers(self):
        x_data = [data.x.numpy() for data in self.data_list if data.x.shape[0] > 0]
        node_seq_data = [data.node_sequences.numpy().reshape(-1, data.node_sequences.shape[-1]) 
                         for data in self.data_list if data.x.shape[0] > 0]
        if x_data:
            self.x_scaler.fit(np.vstack(x_data))
            self.node_seq_scaler.fit(np.vstack(node_seq_data))
            with open(self.scaler_cache_path, 'wb') as f:
                pickle.dump({'x_scaler': self.x_scaler, 'node_seq_scaler': self.node_seq_scaler}, f)

    def _load_or_fit_scalers(self):
        if os.path.exists(self.scaler_cache_path):
            with open(self.scaler_cache_path, 'rb') as f:
                scalers = pickle.load(f)
            self.x_scaler = scalers['x_scaler']
            self.node_seq_scaler = scalers['node_seq_scaler']
        else:
            self._fit_scalers()

    def _normalize_data(self, data):
        x = torch.tensor(self.x_scaler.transform(data.x.numpy()), dtype=torch.float)
        y = torch.tensor(np.log1p(data.y.numpy()), dtype=torch.float)  # Log-scaling for y
        node_seq = torch.tensor(self.node_seq_scaler.transform(
            data.node_sequences.numpy().reshape(-1, data.node_sequences.shape[-1])).reshape(data.node_sequences.shape), 
            dtype=torch.float)
        # Noise injection for robustness
        if self.training:
            x += torch.randn_like(x) * 0.01
            node_seq += torch.randn_like(node_seq) * 0.01
        return Data(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr, y=y, node_sequences=node_seq)

    def len(self):
        return len(self.data_list) if self.data_list else len(os.listdir(self.processed_dir))

    def get(self, idx):
        if self.data_list:
            return self.data_list[idx]
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return self._normalize_data(data)

    def process(self):
        if not self.data_list:
            return
        for i, data in enumerate(self.data_list):
            if data.x.shape[0] > 0:
                torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))

# Enhanced GNN+LSTM Model with GAT and Residual Connections
class GNNLSTMModel(nn.Module):
    def __init__(self, node_dim, edge_dim, seq_dim, hidden_dim=256, lstm_layers=3, dropout=0.3):
        super(GNNLSTMModel, self).__init__()
        self.gat1 = GATConv(node_dim, hidden_dim, heads=4, concat=True, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True, dropout=dropout)
        self.gat3 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.lstm = nn.LSTM(seq_dim, hidden_dim // 2, lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim + hidden_dim // 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim * 4)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, data):
        x, edge_index, node_sequences = data.x, data.edge_index, data.node_sequences
        x1 = self.gat1(x, edge_index)
        x1 = self.relu(self.norm1(x1))
        x1 = self.dropout(x1)
        x2 = self.gat2(x1, edge_index) + x1  # Residual connection
        x2 = self.relu(self.norm1(x2))
        x2 = self.dropout(x2)
        gnn_out = self.gat3(x2, edge_index)
        gnn_out = self.relu(self.norm2(gnn_out))

        if len(node_sequences.shape) == 2:
            node_sequences = node_sequences.unsqueeze(0)
        lstm_out, _ = self.lstm(node_sequences)
        lstm_out = lstm_out[:, -1, :]  # Take the last output

        combined = torch.cat([gnn_out.mean(dim=0, keepdim=True), lstm_out], dim=1)
        out = self.fc(combined)
        return out.squeeze()

# Training Function with Cosine Annealing
def train_model(model, train_loader, val_loader, device, num_epochs=100, lr=0.0003, patience=15):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_losses, val_losses = [], []
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
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                val_loss += criterion(out, data.y).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

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

# Testing Function
def test_model(model, test_loader, device):
    model.eval()
    predictions, actuals, percentage_errors = [], [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            pred = np.expm1(out.cpu().numpy())  # Inverse log-scaling
            actual = np.expm1(data.y.cpu().numpy())
            predictions.append(pred.item())
            actuals.append(actual.item())
            perc_error = abs(pred - actual) / actual * 100 if actual != 0 else 0
            percentage_errors.append(perc_error)
    return predictions, actuals, percentage_errors

# Plotting Function
def plot_losses(train_losses, val_losses, save_path='loss_gnn.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Huber Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

# Main Execution
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = HalideDataset(root='data_g')
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with open(os.path.join('data_g', 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    model = GNNLSTMModel(node_dim=metadata['node_feature_dim'], edge_dim=metadata['edge_feature_dim'], 
                         seq_dim=metadata['seq_feature_dim']).to(device)

    train_losses, val_losses = train_model(model, train_loader, val_loader, device)
    plot_losses(train_losses, val_losses)

    predictions, actuals, percentage_errors = test_model(model, test_loader, device)
    print("\nTest Sample Predictions:")
    print("Sample | Predicted (ms) | Actual (ms) | Percentage Error (%)")
    print("-" * 60)
    for i, (pred, act, err) in enumerate(zip(predictions, actuals, percentage_errors)):
        print(f"{i+1:5d} | {pred:13.4f} | {act:11.4f} | {err:19.4f}")
    avg_error = np.mean(percentage_errors)
    print(f"\nAverage Percentage Error: {avg_error:.4f}%")

if __name__ == "__main__":
    main()
