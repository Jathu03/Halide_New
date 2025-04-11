import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import json

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 1. Load the preprocessed data
def load_preprocessed_data(data_dir="preprocessed_dataset"):
    try:
        sequence_data = np.load(os.path.join(data_dir, "sequence_data.npy"), allow_pickle=True)
        execution_times = np.load(os.path.join(data_dir, "execution_times.npy"), allow_pickle=True)
        
        sequence_data = sequence_data.astype(np.float32)
        execution_times = execution_times.astype(np.float32)
        
        if sequence_data.size == 0 or execution_times.size == 0:
            raise ValueError("Loaded data is empty. Check the .npy files.")
        
        # Reshape execution_times to (n_samples, 20) assuming 20 schedules
        n_samples = sequence_data.shape[0]
        execution_times = execution_times.reshape(n_samples, 20)
        
        return sequence_data, execution_times
    except Exception as e:
        print(f"Error loading data from {data_dir}: {e}")
        raise

# 2. Custom Dataset for PyTorch
class ExecutionTimeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 3. Prepare data for LSTM
def prepare_lstm_data(sequence_data, execution_times, batch_size=32):
    # Normalize sequence data (X) - 11 features across 38 timesteps
    scaler_X = StandardScaler()
    n_samples, seq_len, n_features = sequence_data.shape
    X_reshaped = sequence_data.reshape(-1, n_features)  # (7999*38, 11)
    X_scaled = scaler_X.fit_transform(X_reshaped).reshape(n_samples, seq_len, n_features)

    # Normalize execution times (y) - now (7999, 20)
    scaler_y = StandardScaler()
    y_reshaped = execution_times.reshape(-1, 1)  # (7999*20, 1)
    y_scaled = scaler_y.fit_transform(y_reshaped).reshape(n_samples, 20)  # (7999, 20)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = ExecutionTimeDataset(X_train, y_train)
    test_dataset = ExecutionTimeDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X_train, X_test, y_train, y_test, scaler_X, scaler_y

# 4. Define LSTM model for 20 outputs
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, dropout=0.2, num_outputs=20):
        super(LSTMModel, self).__init__()
        self.num_outputs = num_outputs
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        # Adjust fully connected layers to output 20 values
        self.fc1 = nn.Linear(hidden_size2 * 38, 128)  # Flatten all timesteps
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_outputs)  # 20 outputs
    
    def forward(self, x):
        # x shape: (batch_size, 38, input_size)
        out, _ = self.lstm1(x)  # (batch_size, 38, hidden_size1)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)  # (batch_size, 38, hidden_size2)
        out = self.dropout2(out)  # (batch_size, 38, hidden_size2)
        # Flatten all timesteps
        out = out.reshape(out.size(0), -1)  # (batch_size, 38 * hidden_size2)
        out = self.fc1(out)  # (batch_size, 128)
        out = self.relu(out)
        out = self.fc2(out)  # (batch_size, 20)
        return out

# 5. Train the model
def train_model(model, train_loader, test_loader, device, epochs=300):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    test_losses = []
    
    os.makedirs("loss_model", exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)  # (batch_size, 20)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                test_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
        
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    # Save losses
    np.save("loss_model/train_losses.npy", np.array(train_losses))
    np.save("loss_model/test_losses.npy", np.array(test_losses))
    
    return train_losses, test_losses

# 6. Evaluate and predict
def evaluate_model(model, X_test, y_test, scaler_y, device):
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy()  # (n_test, 20)
    
    # Inverse transform predictions and true values
    n_test = y_test.shape[0]
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(n_test, 20)
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(n_test, 20)
    
    mae_orig = np.mean(np.abs(y_test_orig - y_pred_orig))
    print(f"MAE in original scale (ms) across 20 schedules: {mae_orig:.4f}")
    
    return y_test_orig, y_pred_orig

# 7. Plot train and validation losses
def plot_results(train_losses, test_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()

# 8. Save scalers
def save_scalers(scaler_X, scaler_y):
    scaler_X_data = {
        "means": scaler_X.mean_.tolist(),
        "scales": scaler_X.scale_.tolist(),
        "feature_names": [f"feature_{i}" for i in range(len(scaler_X.mean_))]
    }
    with open("scaler_X.json", "w") as f:
        json.dump(scaler_X_data, f)
    print("Scaler X saved to scaler_X.json")

    scaler_y_data = {
        "mean": float(scaler_y.mean_[0]),
        "scale": float(scaler_y.scale_[0]),
        "is_log_transformed": False
    }
    with open("scaler_y.json", "w") as f:
        json.dump(scaler_y_data, f)
    print("Scaler Y saved to scaler_y.json")

# Main execution
if __name__ == "__main__":
    # Load data
    data_dir = "preprocessed_dataset"
    sequence_data, execution_times = load_preprocessed_data(data_dir)
    
    print("Sequence Data Shape:", sequence_data.shape)
    print("Execution Times Shape:", execution_times.shape)

    if sequence_data.size == 0 or execution_times.size == 0:
        print("Error: No valid data loaded. Please check the preprocessing step.")
        exit()

    # Prepare data
    batch_size = 32
    train_loader, test_loader, X_train, X_test, y_train, y_test, scaler_X, scaler_y = prepare_lstm_data(
        sequence_data, execution_times, batch_size
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = sequence_data.shape[2]  # 11 features
    model = LSTMModel(input_size=input_size, num_outputs=20).to(device)
    
    # Print model summary
    print(model)
    
    # Train model
    train_losses, test_losses = train_model(
        model, train_loader, test_loader, device, epochs=300
    )
    
    # Evaluate and predict
    y_test_orig, y_pred_orig = evaluate_model(model, X_test, y_test, scaler_y, device)
    
    # Plot results
    plot_results(train_losses, test_losses)
    
    # Save the model
    torch.save(model.state_dict(), "lstm_execution_time_model.pth")
    print("Model saved to lstm_execution_time_model.pth")

    # Save as TorchScript
    model.eval()
    example_input = torch.randn(1, 38, input_size).to(device)
    traced_model = torch.jit.trace(model, example_input)
    traced_model.save("lstm_model.pt")
    print("TorchScript model saved to lstm_model.pt")

    # Save scalers
    save_scalers(scaler_X, scaler_y)
