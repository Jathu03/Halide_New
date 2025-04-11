import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 1. Load the preprocessed data
def load_preprocessed_data(data_dir="preprocessed_dataset"):
    try:
        # Load with allow_pickle=True to handle object arrays
        sequence_data = np.load(os.path.join(data_dir, "sequence_data.npy"), allow_pickle=True)
        execution_times = np.load(os.path.join(data_dir, "execution_times.npy"), allow_pickle=True)
        
        # Convert to float32 to ensure compatibility with PyTorch
        sequence_data = sequence_data.astype(np.float32)
        execution_times = execution_times.astype(np.float32)
        
        # Validate data
        if sequence_data.size == 0 or execution_times.size == 0:
            raise ValueError("Loaded data is empty. Check the .npy files.")
        
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
    # Normalize execution times
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(execution_times.reshape(-1, 1)).flatten()

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        sequence_data, y, test_size=0.2, random_state=42
    )

    # Create datasets
    train_dataset = ExecutionTimeDataset(X_train, y_train)
    test_dataset = ExecutionTimeDataset(X_test, y_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X_train, X_test, y_train, y_test, scaler_y

# 4. Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True, return_sequences=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])  # Take the last timestep
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 5. Train the model
def train_model(model, train_loader, test_loader, device, epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    test_losses = []
    train_maes = []
    test_maes = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            train_mae += torch.abs(outputs - y_batch).sum().item()
        
        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        test_mae = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                test_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
                test_mae += torch.abs(outputs - y_batch).sum().item()
        
        test_loss /= len(test_loader.dataset)
        test_mae /= len(test_loader.dataset)
        test_losses.append(test_loss)
        test_maes.append(test_mae)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, "
              f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
    
    return train_losses, test_losses, train_maes, test_maes

# 6. Evaluate and predict
def evaluate_model(model, X_test, y_test, scaler_y, device):
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred = model(X_test_tensor).squeeze().cpu().numpy()
    
    y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    mae_orig = np.mean(np.abs(y_test_orig - y_pred_orig))
    print(f"MAE in original scale (ms): {mae_orig:.4f}")
    
    return y_test_orig, y_pred_orig

# 7. Plot results
def plot_results(train_losses, test_losses, train_maes, test_maes, y_test_orig, y_pred_orig):
    # Plot training & validation loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    # Plot training & validation MAE
    plt.subplot(1, 2, 2)
    plt.plot(train_maes, label='Training MAE')
    plt.plot(test_maes, label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # Scatter plot of true vs predicted execution times
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_orig, y_pred_orig, alpha=0.5)
    plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--')
    plt.title('True vs Predicted Execution Times')
    plt.xlabel('True Execution Time (ms)')
    plt.ylabel('Predicted Execution Time (ms)')
    plt.show()

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
    train_loader, test_loader, X_train, X_test, y_train, y_test, scaler_y = prepare_lstm_data(
        sequence_data, execution_times, batch_size
    )
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = sequence_data.shape[2]  # Number of features
    model = LSTMModel(input_size=input_size).to(device)
    
    # Print model summary
    print(model)
    
    # Train model
    train_losses, test_losses, train_maes, test_maes = train_model(
        model, train_loader, test_loader, device, epochs=50
    )
    
    # Evaluate and predict
    y_test_orig, y_pred_orig = evaluate_model(model, X_test, y_test, scaler_y, device)
    
    # Plot results
    plot_results(train_losses, test_losses, train_maes, test_maes, y_test_orig, y_pred_orig)
    
    # Save the model
    torch.save(model.state_dict(), "lstm_execution_time_model.pth")
    print("Model saved to lstm_execution_time_model.pth")
