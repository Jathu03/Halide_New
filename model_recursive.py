import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Any

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Custom Dataset class with normalization
class HalideDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], scaler=None):
        self.data = data
        self.scaler = scaler if scaler else MinMaxScaler()
        execution_times = np.array([d['execution_time'] for d in data]).reshape(-1, 1)
        self.scaler.fit(execution_times) if not scaler else None
        self.scaled_times = self.scaler.transform(execution_times)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'adj_list': sample['adj_list'],
            'node_features': torch.tensor(sample['node_features'], dtype=torch.float32),
            'node_sequences': torch.tensor(sample['node_sequences'], dtype=torch.float32),
            'edge_features': torch.tensor(sample['edge_features'], dtype=torch.float32),
            'edge_sequences': torch.tensor(sample['edge_sequences'], dtype=torch.float32),
            'execution_time': torch.tensor(self.scaled_times[idx], dtype=torch.float32)
        }

    def inverse_transform(self, scaled_values):
        return self.scaler.inverse_transform(scaled_values.reshape(-1, 1)).flatten()

# Recursive LSTM Model with improved graph processing
class RecursiveLSTM(nn.Module):
    def __init__(self, node_feature_dim, node_seq_dim, edge_feature_dim, edge_seq_dim, hidden_dim=128, lstm_hidden_dim=64):
        super(RecursiveLSTM, self).__init__()
        self.node_lstm = nn.LSTM(node_seq_dim, lstm_hidden_dim, batch_first=True, num_layers=2)
        self.edge_lstm = nn.LSTM(edge_seq_dim, lstm_hidden_dim, batch_first=True, num_layers=2)
        self.node_fc = nn.Linear(node_feature_dim + lstm_hidden_dim, hidden_dim)
        self.edge_fc = nn.Linear(edge_feature_dim + lstm_hidden_dim, hidden_dim)
        self.graph_fc = nn.Linear(hidden_dim, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, adj_list, node_features, node_sequences, edge_features, edge_sequences):
        batch_size = len(node_features)
        graph_outputs = []

        for i in range(batch_size):
            # Node processing
            node_seq = node_sequences[i].unsqueeze(0)  # (1, nodes, seq_len)
            node_seq_out, _ = self.node_lstm(node_seq)  # (1, nodes, lstm_hidden_dim)
            node_feat = node_features[i]  # (nodes, node_feature_dim)
            node_combined = torch.cat([node_feat, node_seq_out.squeeze(0)], dim=-1)  # (nodes, node_feature_dim + lstm_hidden_dim)
            node_out = self.relu(self.node_fc(node_combined))  # (nodes, hidden_dim)

            # Edge processing
            edge_seq = edge_sequences[i].unsqueeze(0)  # (1, edges, seq_len)
            if edge_seq.size(1) > 0:
                edge_seq_out, _ = self.edge_lstm(edge_seq)  # (1, edges, lstm_hidden_dim)
                edge_feat = edge_features[i]  # (edges, edge_feature_dim)
                edge_combined = torch.cat([edge_feat, edge_seq_out.squeeze(0)], dim=-1)  # (edges, edge_feature_dim + lstm_hidden_dim)
                edge_out = self.relu(self.edge_fc(edge_combined))  # (edges, hidden_dim)
            else:
                edge_out = torch.zeros(1, hidden_dim, device=node_out.device)

            # Recursive aggregation using adj_list
            adj = adj_list[i]  # List of lists: [node_idx: [neighbor_indices]]
            node_agg = node_out.clone()
            for node_idx, neighbors in enumerate(adj):
                if neighbors:
                    neighbor_outs = node_out[neighbors]  # (num_neighbors, hidden_dim)
                    node_agg[node_idx] = self.relu(node_out[node_idx] + torch.mean(neighbor_outs, dim=0))

            # Graph-level aggregation
            node_mean = torch.mean(node_agg, dim=0, keepdim=True)  # (1, hidden_dim)
            edge_mean = torch.mean(edge_out, dim=0, keepdim=True)  # (1, hidden_dim)
            graph_out = self.relu(self.graph_fc(node_mean + edge_mean))  # (1, hidden_dim)
            pred = self.output_fc(graph_out)  # (1, 1)
            graph_outputs.append(pred)

        return torch.cat(graph_outputs, dim=0)  # (batch_size, 1)

# Collate function
def collate_fn(batch):
    return {
        'adj_list': [item['adj_list'] for item in batch],
        'node_features': [item['node_features'] for item in batch],
        'node_sequences': [item['node_sequences'] for item in batch],
        'edge_features': [item['edge_features'] for item in batch],
        'edge_sequences': [item['edge_sequences'] for item in batch],
        'execution_time': torch.stack([item['execution_time'] for item in batch])
    }

# Load and split dataset
def load_and_split_data(file_path='data_r.pkl'):
    with open(file_path, 'rb') as f:
        data_r = pickle.load(f)
    
    valid_data = [d for d in data_r if d['execution_time'] > 0]
    test_data = valid_data[:10]
    remaining_data = valid_data[10:]
    train_data, val_data = train_test_split(remaining_data, test_size=0.2, random_state=42)
    
    # Fit scaler on training data only
    train_dataset = HalideDataset(train_data)
    scaler = train_dataset.scaler
    val_dataset = HalideDataset(val_data, scaler)
    test_dataset = HalideDataset(test_data, scaler)
    
    return train_dataset, val_dataset, test_dataset

# Training function
def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.0001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            adj_list = batch['adj_list']
            node_features = [nf.to(device) for nf in batch['node_features']]
            node_sequences = [ns.to(device) for ns in batch['node_sequences']]
            edge_features = [ef.to(device) for ef in batch['edge_features']]
            edge_sequences = [es.to(device) for es in batch['edge_sequences']]
            targets = batch['execution_time'].to(device).view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(adj_list, node_features, node_sequences, edge_features, edge_sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                adj_list = batch['adj_list']
                node_features = [nf.to(device) for nf in batch['node_features']]
                node_sequences = [ns.to(device) for ns in batch['node_sequences']]
                edge_features = [ef.to(device) for ef in batch['edge_features']]
                edge_sequences = [es.to(device) for es in batch['edge_sequences']]
                targets = batch['execution_time'].to(device).view(-1, 1)
                
                outputs = model(adj_list, node_features, node_sequences, edge_features, edge_sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses

# Plot loss
def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Plot (loss_r)')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_r.png')
    plt.close()
    print("Loss plot saved as loss_r.png")

# Evaluate model
def evaluate_model(model, test_loader, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            adj_list = batch['adj_list']
            node_features = [nf.to(device) for nf in batch['node_features']]
            node_sequences = [ns.to(device) for ns in batch['node_sequences']]
            edge_features = [ef.to(device) for ef in batch['edge_features']]
            edge_sequences = [es.to(device) for es in batch['edge_sequences']]
            targets = batch['execution_time'].to(device).view(-1, 1)
            
            outputs = model(adj_list, node_features, node_sequences, edge_features, edge_sequences)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(targets.cpu().numpy().flatten())
    
    # Inverse transform predictions and actuals
    predictions = dataset.inverse_transform(np.array(predictions))
    actuals = dataset.inverse_transform(np.array(actuals))
    
    error_percentages = [abs(pred - actual) / actual * 100 for pred, actual in zip(predictions, actuals) if actual != 0]
    mean_error_percentage = np.mean(error_percentages)
    
    print("\nTest Set Predictions vs Actuals:")
    for i, (pred, actual) in enumerate(zip(predictions, actuals)):
        print(f"Sample {i+1}: Predicted: {pred:.4f}, Actual: {actual:.4f}, Error%: {error_percentages[i]:.2f}%")
    print(f"Mean Error Percentage: {mean_error_percentage:.2f}%")
    
    return mean_error_percentage

# Main execution
if __name__ == "__main__":
    train_dataset, val_dataset, test_dataset = load_and_split_data('data_r.pkl')
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    sample = train_dataset.data[0]
    model = RecursiveLSTM(
        node_feature_dim=sample['node_features'].shape[-1],
        node_seq_dim=sample['node_sequences'].shape[-1],
        edge_feature_dim=sample['edge_features'].shape[-1],
        edge_seq_dim=sample['edge_sequences'].shape[-1],
        hidden_dim=128,
        lstm_hidden_dim=64
    )
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=100, lr=0.0001)
    plot_loss(train_losses, val_losses)
    mean_error_percentage = evaluate_model(model, test_loader, test_dataset)
