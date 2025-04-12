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
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler

class HalideDataset(Dataset):
    def __init__(self, data_list=None, root='data_g'):
        self.data_list = data_list if data_list is not None else []
        super(HalideDataset, self).__init__(root)
        os.makedirs(self.processed_dir, exist_ok=True)
        # Initialize scalers
        self.x_scaler = RobustScaler(quantile_range=(5, 95))  # More robust to outliers
        self.edge_scaler = RobustScaler(quantile_range=(5, 95))
        self.node_seq_scaler = RobustScaler(quantile_range=(5, 95))
        self.valid_files_cache = None
        self.scaler_cache_path = os.path.join(self.processed_dir, 'scalers.pkl')
        if self.data_list:
            self._fit_scalers()
            self.data_list = [self._normalize_data(data) for data in self.data_list if data.x.shape[0] > 0]
        else:
            self._load_valid_files()
            self._load_or_fit_scalers()
    
    def _fit_scalers(self):
        x_data = []
        edge_data = []
        node_seq_data = []
        data_source = self.data_list if self.data_list else [torch.load(os.path.join(self.processed_dir, f)) for f in self.valid_files_cache]
        for data in data_source:
            if data.x.shape[0] > 0:
                x_data.append(data.x.numpy())
                if data.edge_attr is not None:
                    edge_data.append(data.edge_attr.numpy())
                node_seq_data.append(data.node_sequences.numpy().reshape(-1, data.node_sequences.shape[-1]))
        
        if x_data:
            self.x_scaler.fit(np.vstack(x_data))
            if edge_data:
                self.edge_scaler.fit(np.vstack(edge_data))
            self.node_seq_scaler.fit(np.vstack(node_seq_data))
            try:
                with open(self.scaler_cache_path, 'wb') as f:
                    pickle.dump({
                        'x_scaler': self.x_scaler,
                        'edge_scaler': self.edge_scaler,
                        'node_seq_scaler': self.node_seq_scaler
                    }, f)
            except Exception as e:
                print(f"Error saving scalers: {e}")

    def _normalize_data(self, data):
        x = torch.tensor(self.x_scaler.transform(data.x.numpy()), dtype=torch.float)
        # Log-scale y with small epsilon to avoid log(0)
        y = torch.tensor(np.log1p(data.y.numpy() + 1e-6), dtype=torch.float)
        node_seq = torch.tensor(self.node_seq_scaler.transform(
            data.node_sequences.numpy().reshape(-1, data.node_sequences.shape[-1])
        ).reshape(data.node_sequences.shape), dtype=torch.float)
        
        edge_attr = None
        if data.edge_attr is not None:
            edge_attr = torch.tensor(self.edge_scaler.transform(data.edge_attr.numpy()), dtype=torch.float)
        
        return Data(x=x, edge_index=data.edge_index, edge_attr=edge_attr, y=y, node_sequences=node_seq)

class GNNLSTMModel(nn.Module):
    def __init__(self, node_dim, edge_dim, seq_dim, hidden_dim=512, lstm_layers=3, dropout=0.4):
        super(GNNLSTMModel, self).__init__()
        # Enhanced GNN with edge features
        self.gnn1 = torch_geometric.nn.GATConv(node_dim, hidden_dim, edge_dim=edge_dim)
        self.gnn2 = torch_geometric.nn.GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim)
        self.gnn3 = torch_geometric.nn.GATConv(hidden_dim, hidden_dim, edge_dim=edge_dim)
        
        # Bidirectional LSTM for sequence processing
        self.lstm = nn.LSTM(seq_dim, hidden_dim // 4, lstm_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout if lstm_layers > 1 else 0)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=0)
        )
        
        # Final prediction layers with skip connections
        self.fc1 = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, edge_attr, node_sequences = data.x, data.edge_index, data.edge_attr, data.node_sequences
        
        # GNN processing
        x = self.leaky_relu(self.gnn1(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = self.leaky_relu(self.gnn2(x, edge_index, edge_attr))
        x = self.dropout(x)
        x = self.leaky_relu(self.gnn3(x, edge_index, edge_attr))
        gnn_out = self.norm1(x)
        
        # LSTM processing
        if len(node_sequences.shape) == 2:
            node_sequences = node_sequences.unsqueeze(0)
        if node_sequences.shape[-1] != self.lstm.input_size:
            node_sequences = node_sequences.transpose(1, 2)
        
        lstm_out, _ = self.lstm(node_sequences)
        lstm_out = lstm_out.squeeze(0)
        lstm_out = self.dropout(lstm_out)
        
        # Combine features with attention
        combined = torch.cat([gnn_out, lstm_out], dim=1)
        attention_weights = self.attention(combined)
        attended = (combined * attention_weights).sum(dim=0, keepdim=True)
        
        # Final prediction with skip connections
        out = self.leaky_relu(self.fc1(attended))
        out = self.dropout(out)
        out = self.leaky_relu(self.fc2(out + attended[:, :self.fc2.in_features]))
        out = self.norm2(out)
        out = self.fc3(out)
        
        return out

def train_model(model, train_loader, val_loader, device, num_epochs=200, lr=0.0001, patience=20):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7, verbose=True)
    
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
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
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

def test_model(model, test_loader, device):
    model.eval()
    predictions = []
    actuals = []
    percentage_errors = []
    mae_errors = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            # Inverse transform with clipping to avoid extreme values
            pred = np.clip(np.expm1(out.cpu().numpy().flatten())[0], 0, 1e10)
            actual = np.clip(np.expm1(data.y.cpu().numpy().flatten())[0], 0, 1e10)
            
            predictions.append(pred)
            actuals.append(actual)
            
            if actual != 0:
                perc_error = abs(pred - actual) / actual * 100
                mae_errors.append(abs(pred - actual))
            else:
                perc_error = float('inf') if pred != 0 else 0.0
            percentage_errors.append(perc_error)
    
    # Calculate metrics
    metrics = {
        'mae': np.mean(mae_errors) if mae_errors else float('inf'),
        'r2_score': r2_score(actuals, predictions) if len(actuals) > 1 else 0.0,
        'median_percentage_error': np.median([pe for pe in percentage_errors if pe != float('inf')]) if any(pe != float('inf') for pe in percentage_errors) else float('inf')
    }
    
    return predictions, actuals, percentage_errors, metrics

def plot_results(predictions, actuals, save_path='results_gnn.png'):
    plt.figure(figsize=(12, 6))
    plt.scatter(actuals, predictions, alpha=0.6)
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
    plt.xlabel('Actual Runtime (ms)')
    plt.ylabel('Predicted Runtime (ms)')
    plt.title('Actual vs Predicted Runtimes')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    dataset = HalideDataset(root='data_g')
    print(f"Total samples in dataset: {len(dataset)}")
    
    # Use 80-10-10 split for train-val-test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    
    # Use larger batch sizes if memory allows
    batch_size = 8 if device.type == 'cuda' else 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    metadata_path = os.path.join('data_g', 'metadata.pkl')
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    model = GNNLSTMModel(
        node_dim=metadata['node_feature_dim'],
        edge_dim=metadata['edge_feature_dim'] if metadata['edge_feature_dim'] else 1,
        seq_dim=metadata['seq_feature_dim'],
        hidden_dim=512,
        lstm_layers=3,
        dropout=0.4
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, device,
        num_epochs=200, lr=0.0001, patience=20)
    
    plot_losses(train_losses, val_losses)
    
    predictions, actuals, percentage_errors, metrics = test_model(model, test_loader, device)
    plot_results(predictions, actuals)
    
    print("\nTest Results:")
    print(f"MAE: {metrics['mae']:.4f} ms")
    print(f"R² Score: {metrics['r2_score']:.4f}")
    print(f"Median Percentage Error: {metrics['median_percentage_error']:.4f}%")
    
    print("\nDetailed Predictions:")
    print("Sample | Predicted (ms) | Actual (ms) | Percentage Error (%)")
    print("-" * 60)
    for i, (pred, actual, perc_error) in enumerate(zip(predictions, actuals, percentage_errors)):
        print(f"{i+1:5d} | {pred:13.4f} | {actual:11.4f} | {perc_error:19.4f}")

if __name__ == "__main__":
    main()
