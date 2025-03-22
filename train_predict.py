import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the processed dataset
DATASET_FILE = 'tiramisu_graph_dataset.json'

with open(DATASET_FILE, 'r') as f:
    dataset = json.load(f)

# Function to serialize graph into a sequence
def graph_to_sequence(graph):
    sequence = []
    sequence.append(f"SCHED:{graph['schedule_str']}")
    for node, attrs in graph['attributes'].items():
        node_type = attrs['type']
        attr_str = ' '.join([f"{k}={str(v).replace(' ', '_')}" for k, v in attrs.items()])
        sequence.append(f"NODE:{node} TYPE:{node_type} {attr_str}")
    for src, dst, edge_type in graph['edges']:
        sequence.append(f"EDGE:{src}->{dst} TYPE:{edge_type}")
    tree_str = str(graph['tree_structure']).replace(' ', '_')
    sequence.append(f"TREE:{tree_str}")
    return ' '.join(sequence)

# Prepare data
sequences = [graph_to_sequence(entry['graph']) for entry in dataset]
execution_times = np.array([entry['avg_execution_time'] for entry in dataset])

# Normalize execution times
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(execution_times.reshape(-1, 1)).flatten()

# Tokenize sequences and build vocabulary
tokenized_sequences = [seq.split() for seq in sequences]
all_tokens = [token for seq in tokenized_sequences for token in seq]
vocab = {token: idx + 1 for idx, token in enumerate(sorted(set(all_tokens)))}
vocab_size = len(vocab) + 1

# Convert sequences to indices
def sequence_to_indices(sequence, vocab, max_length=200):
    indices = [vocab.get(token, 0) for token in sequence.split()[:max_length]]
    while len(indices) < max_length:
        indices.append(0)
    return indices

X_indices = np.array([sequence_to_indices(seq, vocab) for seq in sequences])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_indices, y_scaled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define LSTM Dataset
class TiramisuDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoaders
train_dataset = TiramisuDataset(X_train, y_train)
test_dataset = TiramisuDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define Enhanced LSTM Model
class EnhancedLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_size=256, num_layers=3, dropout=0.3):
        super(EnhancedLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        out, _ = self.lstm(embedded)  # (batch_size, seq_length, hidden_size)
        out = self.dropout(out[:, -1, :])  # Last time step: (batch_size, hidden_size)
        out = self.relu(self.fc1(out))    # (batch_size, 64)
        out = self.fc2(out)               # (batch_size, 1)
        return out

# Initialize model, loss, optimizer, and scheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EnhancedLSTMModel(vocab_size=vocab_size, hidden_size=256, num_layers=3).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * X_batch.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # Evaluation
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            test_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
    
    test_loss /= len(test_loader.dataset)
    
    # Step the scheduler
    scheduler.step(test_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, LR: {current_lr:.6f}")

# Save the model and scaler
torch.save(model.state_dict(), 'tiramisu_lstm_model.pth')
np.save('scaler_params.npy', [scaler.scale_, scaler.min_])
print("Model saved to 'tiramisu_lstm_model.pth', Scaler params saved to 'scaler_params.npy'")

# Example: Denormalize a prediction
model.eval()
with torch.no_grad():
    sample_input = X_test[:1].to(device)
    sample_pred = model(sample_input).cpu().numpy()
    denormalized_pred = scaler.inverse_transform(sample_pred)
    print(f"Sample Prediction (normalized): {sample_pred[0][0]}, Denormalized: {denormalized_pred[0][0]}")
