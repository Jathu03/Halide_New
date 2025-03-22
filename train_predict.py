import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from collections import defaultdict

# Load the processed dataset
DATASET_FILE = 'tiramisu_graph_dataset.json'

with open(DATASET_FILE, 'r') as f:
    dataset = json.load(f)

# Function to serialize graph into a sequence
def graph_to_sequence(graph):
    sequence = []
    
    # Add schedule string
    sequence.append(f"SCHED:{graph['schedule_str']}")
    
    # Add nodes and their attributes
    for node, attrs in graph['attributes'].items():
        node_type = attrs['type']
        attr_str = ' '.join([f"{k}={str(v).replace(' ', '_')}" for k, v in attrs.items()])
        sequence.append(f"NODE:{node} TYPE:{node_type} {attr_str}")
    
    # Add edges
    for src, dst, edge_type in graph['edges']:
        sequence.append(f"EDGE:{src}->{dst} TYPE:{edge_type}")
    
    # Add tree structure (simplified)
    tree_str = str(graph['tree_structure']).replace(' ', '_')
    sequence.append(f"TREE:{tree_str}")
    
    return ' '.join(sequence)

# Prepare data: Convert graphs to sequences
sequences = [graph_to_sequence(entry['graph']) for entry in dataset]
execution_times = [entry['avg_execution_time'] for entry in dataset]

# Tokenize sequences
tokenized_sequences = [seq.split() for seq in sequences]

# Train Word2Vec model for embeddings
w2v_model = Word2Vec(sentences=tokenized_sequences, vector_size=100, window=5, min_count=1, workers=4)
w2v_model.save("tiramisu_w2v.model")

# Function to convert sequence to embedding
def sequence_to_embedding(sequence, w2v_model, max_length=200):
    embedding = []
    for token in sequence.split()[:max_length]:  # Truncate or pad to max_length
        embedding.append(w2v_model.wv[token])
    while len(embedding) < max_length:  # Pad with zeros
        embedding.append(np.zeros(100))
    return np.array(embedding)

# Convert all sequences to embeddings
X = np.array([sequence_to_embedding(seq, w2v_model) for seq in sequences])
y = np.array(execution_times)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
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

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=100, hidden_size=128, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        out, _ = self.lstm(x)  # out: (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])  # Take the last time step: (batch_size, 1)
        return out

# Initialize model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

# Save the model
torch.save(model.state_dict(), 'tiramisu_lstm_model.pth')
print("Model saved to 'tiramisu_lstm_model.pth'")
