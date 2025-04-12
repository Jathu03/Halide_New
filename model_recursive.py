import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import List, Dict, Any

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Custom Dataset class with feature normalization
class HalideDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], time_scaler=None, feature_scaler=None):
        self.data = data
        self.time_scaler = time_scaler if time_scaler else MinMaxScaler()
        self.feature_scaler = feature_scaler if feature_scaler else StandardScaler()
        
        # Scale execution times
        execution_times = np.array([d['execution_time'] for d in data]).reshape(-1, 1)
        self.time_scaler.fit(execution_times) if not time_scaler else None
        self.scaled_times = self.time_scaler.transform(execution_times)
        
        # Scale node and edge features
        node_features = np.concatenate([d['node_features'] for d in data], axis=0)
        edge_features = np.concatenate([d['edge_features'] for d in data if d['edge_features'].size > 0], axis=0) if any(d['edge_features'].size > 0 for d in data) else np.array([])
        self.feature_scaler.fit(node_features) if not feature_scaler and node_features.size > 0 else None
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        node_feats = self.feature_scaler.transform(sample['node_features']) if sample['node_features'].size > 0 else sample['node_features']
        edge_feats = self.feature_scaler.transform(sample['edge_features']) if sample['edge_features'].size > 0 else sample['edge_features']
        return {
            'adj_list': sample['adj_list'],
            'node_features': torch.tensor(node_feats, dtype=torch.float32),
            'node_sequences': torch.tensor(sample['node_sequences'], dtype=torch.float32),
            'edge_features': torch.tensor(edge_feats, dtype=torch.float32),
            'edge_sequences': torch.tensor(sample['edge_sequences'], dtype=torch.float32),
            'execution_time': torch.tensor(self.scaled_times[idx], dtype=torch.float32)
        }

    def inverse_transform_time(self, scaled_values):
        return self.time_scaler.inverse_transform(scaled_values.reshape(-1, 1)).flatten()

# Graph Convolution Layer
class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GraphConv, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, node_features, adj_list, device):
        batch_size = len(node_features)
        updated_nodes = []
        
        for i in range(batch_size):
            nodes = node_features[i]  # (num_nodes, in_dim)
            adj = adj_list[i]
            num_nodes = nodes.size(0)
            
            # Build adjacency matrix
            adj_matrix = torch.zeros(num_nodes, num_nodes, device=device)
            for src, neighbors in enumerate(adj):
                for dst in neighbors:
                    adj_matrix[src, dst] = 1.0
            
            # Normalize adjacency
            degree = torch.sum(adj_matrix, dim=1).clamp(min=1)
            norm_adj = adj_matrix / degree.unsqueeze(1)
            
            # Graph convolution
            aggregated = torch.matmul(norm_adj, nodes)
            updated = self.leaky_relu(self.linear(aggregated + nodes))  # Residual connection
            updated_nodes.append(updated)
        
        return updated_nodes

# Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features):
        scores = self.attn(torch.cat([features, features], dim=-1))  # (num_nodes, 1)
        weights = self.softmax(scores)
        return torch.sum(features * weights, dim=0, keepdim=True)  # (1, hidden_dim)

# Enhanced Recursive LSTM Model
class RecursiveLSTM(nn.Module):
    def __init__(self, node_feature_dim, node_seq_dim, edge_feature_dim, edge_seq_dim, hidden_dim=256, lstm_hidden_dim=128):
        super(RecursiveLSTM, self).__init__()
        self.node_lstm = nn.LSTM(node_seq_dim, lstm_hidden_dim, batch_first=True, num_layers=2, dropout=0.2, bidirectional=True)
        self.edge_lstm = nn.LSTM(edge_seq_dim, lstm_hidden_dim, batch_first=True, num_layers=2, dropout=0.2, bidirectional=True)
        self.node_embed = nn.Linear(node_feature_dim + lstm_hidden_dim * 2, hidden_dim)
        self.edge_embed = nn.Linear(edge_feature_dim + lstm_hidden_dim * 2, hidden_dim)
        self.graph_conv = GraphConv(hidden_dim, hidden_dim)
        self.node_attention = Attention(hidden_dim)
        self.edge_attention = Attention(hidden_dim)
        self.graph_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_fc = nn.Linear(hidden_dim, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, adj_list, node_features, node_sequences, edge_features, edge_sequences):
        batch_size = len(node_features)
        graph_outputs = []
        device = node_features[0].device

        for i in range(batch_size):
            # Node processing
            node_seq = node_sequences[i].unsqueeze(0)  # (1, nodes, seq_len)
            node_seq_out, _ = self.node_lstm(node_seq)  # (1, nodes, lstm_hidden_dim * 2)
            node_feat = node_features[i]  # (nodes, node_feature_dim)
            node_combined = torch.cat([node_feat, node_seq_out.squeeze(0)], dim=-1)
            node_embed = self.leaky_relu(self.node_embed(node_combined))  # (nodes, hidden_dim)

            # Edge processing
            edge_seq = edge_sequences[i].unsqueeze(0)  # (1, edges, seq_len)
            if edge_seq.size(1) > 0:
                edge_seq_out, _ = self.edge_lstm(edge_seq)  # (1, edges, lstm_hidden_dim * 2)
                edge_feat = edge_features[i]  # (edges, edge_feature_dim)
                edge_combined = torch.cat([edge_feat, edge_seq_out.squeeze(0)], dim=-1)
                edge_embed = self.leaky_relu(self.edge_embed(edge_combined))  # (edges, hidden_dim)
            else:
                edge_embed = torch.zeros(1, hidden_dim, device=device)

            # Graph convolution
            node_updated = self.graph_conv([node_embed], [adj_list[i]], device)[0]  # (nodes, hidden_dim)

            # Attention-based aggregation
            node_agg = self.node_attention(node_updated)  # (1, hidden_dim)
            edge_agg = self.edge_attention(edge_embed) if edge_embed.size(0) > 0 else torch.zeros(1, hidden_dim, device=device)

            # Graph-level processing
            graph_combined = torch.cat([node_agg, edge_agg], dim=-1)  # (1, hidden_dim * 2)
            graph_out = self.dropout(self.leaky_relu(self.graph_fc(graph_combined)))  # (1, hidden_dim)
            residual = self.graph_fc(graph_combined)  # (1, hidden_dim) to match graph_out
            pred = self.output_fc(self.leaky_relu(graph_out + residual))  # Fixed residual connection
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
    
    train_dataset = HalideDataset(train_data)
    time_scaler = train_dataset.time_scaler
    feature_scaler = train_dataset.feature_scaler
    val_dataset = HalideDataset(val_data, time_scaler, feature_scaler)
    test_dataset = HalideDataset(test_data, time_scaler, feature_scaler)
    
    return train_dataset, val_dataset, test_dataset

# Training function
def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, patience=10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    model.load_state_dict(torch.load('best_model.pt'))
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
    
    predictions = dataset.inverse_transform_time(np.array(predictions))
    actuals = dataset.inverse_transform_time(np.array(actuals))
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    sample = train_dataset.data[0]
    model = RecursiveLSTM(
        node_feature_dim=sample['node_features'].shape[-1],
        node_seq_dim=sample['node_sequences'].shape[-1],
        edge_feature_dim=sample['edge_features'].shape[-1],
        edge_seq_dim=sample['edge_sequences'].shape[-1],
        hidden_dim=256,
        lstm_hidden_dim=128
    )
    
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, patience=10)
    plot_loss(train_losses, val_losses)
    mean_error_percentage = evaluate_model(model, test_loader, test_dataset)
