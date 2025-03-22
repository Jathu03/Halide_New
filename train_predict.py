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

# Split into train, validation, and test sets (70% train, 15% val, 15% test)
X_temp, X_test, y_temp, y_test = train_test_split(X_indices, y_scaled, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)  # 0.1765 of 85% is ~15% of total

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val = torch.tensor(X_val, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
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
val_dataset = TiramisuDataset(X_val, y_val)
test_dataset = TiramisuDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
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
        embedded = self.embedding(x)
        out, _ = self.lstm(embedded)
        out = self.dropout(out[:, -1, :])
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# Initialize model, loss, optimizer, and scheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EnhancedLSTMModel(vocab_size=vocab_size).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Function to calculate MAPE
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0  # Avoid division by zero
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else float('inf')

# Training loop
num_epochs = 50
best_val_loss = float('inf')
for epoch in range(num_epochs):
    # Training
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

    # Validation
    model.eval()
    val_loss = 0
    val_preds, val_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            val_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
            val_preds.extend(outputs.cpu().numpy().flatten())
            val_true.extend(y_batch.cpu().numpy().flatten())
    val_loss /= len(val_loader.dataset)
    val_mape = calculate_mape(scaler.inverse_transform(np.array(val_true).reshape(-1, 1)),
                              scaler.inverse_transform(np.array(val_preds).reshape(-1, 1)))

    # Test
    test_loss = 0
    test_preds, test_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            test_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
            test_preds.extend(outputs.cpu().numpy().flatten())
            test_true.extend(y_batch.cpu().numpy().flatten())
    test_loss /= len(test_loader.dataset)
    test_mape = calculate_mape(scaler.inverse_transform(np.array(test_true).reshape(-1, 1)),
                               scaler.inverse_transform(np.array(test_preds).reshape(-1, 1)))

    # Scheduler step
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    # Save best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'tiramisu_lstm_model_best.pth')

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Test Loss: {test_loss:.6f}, "
          f"Val MAPE: {val_mape:.2f}%, Test MAPE: {test_mape:.2f}%, LR: {current_lr:.6f}")

# Save final model and scaler
torch.save(model.state_dict(), 'tiramisu_lstm_model_final.pth')
np.save('scaler_params.npy', [scaler.scale_, scaler.min_])
print("Final model saved to 'tiramisu_lstm_model_final.pth', Best model saved to 'tiramisu_lstm_model_best.pth', "
      "Scaler params saved to 'scaler_params.npy'")

# Final evaluation on test set
model.eval()
test_preds, test_true = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        test_preds.extend(outputs.cpu().numpy().flatten())
        test_true.extend(y_batch.cpu().numpy().flatten())

# Denormalize and calculate final MAPE
test_true_denorm = scaler.inverse_transform(np.array(test_true).reshape(-1, 1)).flatten()
test_preds_denorm = scaler.inverse_transform(np.array(test_preds).reshape(-1, 1)).flatten()
final_test_mape = calculate_mape(test_true_denorm, test_preds_denorm)

print(f"\nFinal Test MAPE: {final_test_mape:.2f}%")
print(f"Sample True vs Predicted (denormalized):")
for i in range(min(5, len(test_true_denorm))):
    print(f"True: {test_true_denorm[i]:.4f}, Predicted: {test_preds_denorm[i]:.4f}")
