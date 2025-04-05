import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# The JSON document (replace with your document)
document = '''
<JSON_DOCUMENT_FROM_YOUR_INPUT>
'''

# Parse the JSON document
data = json.loads(document)

# Step 1: Extract execution time and scheduling features
def extract_execution_details(data):
    execution_details = {
        "total_execution_time_ms": None,
        "nodes": []
    }
    scheduling = data["programming_details"]["Scheduling"]
    for entry in scheduling:
        if "name" in entry and entry["name"] == "total_execution_time_ms":
            execution_details["total_execution_time_ms"] = entry["value"]
            continue
        if "Details" in entry and "scheduling_feature" in entry["Details"]:
            node_name = entry["Name"]
            features = entry["Details"]["scheduling_feature"]
            node_features = {
                "name": node_name,
                "points_computed_total": features.get("points_computed_total", 0.0),
                "vector_loads_per_vector": features.get("vector_loads_per_vector", 0.0),
                "scalar_loads_per_scalar": features.get("scalar_loads_per_scalar", 0.0),
                "inner_parallelism": features.get("inner_parallelism", 0.0),
                "outer_parallelism": features.get("outer_parallelism", 0.0),
                "working_set": features.get("working_set", 0.0),
                "working_set_at_realization": features.get("working_set_at_realization", 0.0),
                "num_vectors": features.get("num_vectors", 0.0),
                "num_scalars": features.get("num_scalars", 0.0),
                "inlined_calls": features.get("inlined_calls", 0.0),
                "bytes_at_realization": features.get("bytes_at_realization", 0.0),
                "allocation_bytes_read_per_realization": features.get("allocation_bytes_read_per_realization", 0.0)
            }
            execution_details["nodes"].append(node_features)
    return execution_details

# Extract the details
execution_details = extract_execution_details(data)

# Feature keys to include in the model
feature_keys = [
    "points_computed_total", "vector_loads_per_vector", "scalar_loads_per_scalar",
    "inner_parallelism", "outer_parallelism", "working_set", "working_set_at_realization",
    "num_vectors", "num_scalars", "inlined_calls", "bytes_at_realization",
    "allocation_bytes_read_per_realization"
]

# Step 2: Create a feature matrix
def create_feature_matrix(nodes, feature_keys):
    feature_matrix = []
    for node in nodes:
        feature_vector = [node[key] for key in feature_keys]
        feature_matrix.append(feature_vector)
    return np.array(feature_matrix)

# Extract the feature matrix and target
X = create_feature_matrix(execution_details["nodes"], feature_keys)
y = execution_details["total_execution_time_ms"]

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Reshape X for LSTM: [samples, timesteps, features]
timesteps = X_normalized.shape[0]
features = X_normalized.shape[1]
X_lstm = X_normalized.reshape(1, timesteps, features)

# Simulate additional samples by perturbing the data
num_simulated_samples = 100
X_simulated = []
y_simulated = []

for _ in range(num_simulated_samples):
    noise = np.random.uniform(0.9, 1.1, size=X.shape)
    X_perturbed = X * noise
    X_perturbed_normalized = scaler.transform(X_perturbed)
    X_simulated.append(X_perturbed_normalized)
    noise_y = np.random.uniform(0.9, 1.1)
    y_perturbed = y * noise_y
    y_simulated.append(y_perturbed)

X_simulated = np.array(X_simulated)
y_simulated = np.array(y_simulated)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_simulated, dtype=torch.float32)
y_tensor = torch.tensor(y_simulated, dtype=torch.float32).unsqueeze(1)  # Shape: [num_samples, 1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create DataLoader for batch processing
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Step 3: Define the LSTM model using PyTorch
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, output_size)

    def forward(self, x):
        # Initialize hidden states
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, 64).to(x.device)
        c0 = torch.zeros(1, batch_size, 64).to(x.device)
        
        # First LSTM layer
        out, _ = self.lstm1(x, (h0, c0))
        out = self.dropout1(out)
        
        # Second LSTM layer
        h1 = torch.zeros(1, batch_size, 32).to(x.device)
        c1 = torch.zeros(1, batch_size, 32).to(x.device)
        out, _ = self.lstm2(out, (h1, c1))
        out = self.dropout2(out)
        
        # Take the output of the last timestep
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Instantiate the model
input_size = features  # Number of features per timestep
hidden_size1 = 64
hidden_size2 = 32
output_size = 1  # Predicting a single value (execution time)
model = LSTMModel(input_size, hidden_size1, hidden_size2, output_size)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
train_losses = []
val_losses = []
train_maes = []
val_maes = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_mae = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_X.size(0)
        train_mae += torch.mean(torch.abs(outputs - batch_y)).item() * batch_X.size(0)
    
    train_loss /= len(train_loader.dataset)
    train_mae /= len(train_loader.dataset)
    train_losses.append(train_loss)
    train_maes.append(train_mae)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
            val_mae += torch.mean(torch.abs(outputs - batch_y)).item() * batch_X.size(0)
    
    val_loss /= len(test_loader.dataset)
    val_mae /= len(test_loader.dataset)
    val_losses.append(val_loss)
    val_maes.append(val_mae)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

# Evaluate the model on the test set
model.eval()
test_loss = 0.0
test_mae = 0.0
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item() * batch_X.size(0)
        test_mae += torch.mean(torch.abs(outputs - batch_y)).item() * batch_X.size(0)

test_loss /= len(test_loader.dataset)
test_mae /= len(test_loader.dataset)
print(f"\nTest Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Predict on the original data
X_original = torch.tensor(X_lstm, dtype=torch.float32).to(device)
model.eval()
with torch.no_grad():
    y_pred = model(X_original)
y_pred = y_pred.cpu().numpy()
print(f"\nPredicted Execution Time: {y_pred[0][0]:.4f} ms")
print(f"Actual Execution Time: {y:.4f} ms")

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.show()

# Plot training and validation MAE
plt.figure(figsize=(10, 5))
plt.plot(train_maes, label="Training MAE")
plt.plot(val_maes, label="Validation MAE")
plt.title("Training and Validation MAE Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.show()
