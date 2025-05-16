import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LSTMModel(nn.Module):
    """LSTM model for regression."""
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True, return_sequences=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])  # Take the last time step
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def load_dataset(dataset_path):
    """Load the dataset from the .npz file."""
    data = np.load(dataset_path)
    X = data['sequences']
    y = data['execution_times']
    return X, y

def calculate_mape(actual, predicted):
    """Calculate Mean Absolute Percentage Error for each sample."""
    actual = actual.squeeze().numpy()
    predicted = predicted.squeeze().numpy()
    # Avoid division by zero
    mask = actual != 0
    mape = np.abs((actual[mask] - predicted[mask]) / actual[mask]) * 100
    return mape

def train_and_evaluate(dataset_path, test_size=10, epochs=50, batch_size=32, device='cpu'):
    """Train the LSTM model and evaluate on test set."""
    # Load dataset
    X, y = load_dataset(dataset_path)
    print(f"Dataset loaded. X shape: {X.shape}, y shape: {y.shape}")

    # Ensure test_size does not exceed number of samples
    if test_size >= len(X):
        print(f"Error: test_size ({test_size}) is larger than dataset size ({len(X)}).")
        return

    # Split into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # Split train+val into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, loss, and optimizer
    model = LSTMModel(input_dim=X.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            train_mae += torch.mean(torch.abs(outputs - y_batch)).item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_mae += torch.mean(torch.abs(outputs - y_batch)).item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        test_predictions = model(X_test)
        test_predictions = test_predictions.cpu()
    mape_scores = calculate_mape(y_test, test_predictions)

    # Print individual and average MAPE
    print("\nTest Set Results:")
    print("-----------------")
    for i, (actual, pred, mape) in enumerate(zip(y_test.flatten(), test_predictions.flatten(), mape_scores)):
        print(f"Sample {i+1}: Actual = {actual:.2f} ms, Predicted = {pred:.2f} ms, MAPE = {mape:.2f}%")
    avg_mape = np.mean(mape_scores)
    print(f"\nAverage MAPE on test set: {avg_mape:.2f}%")

    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(train_maes, label='Training MAE')
    plt.plot(val_maes, label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

    # Save model
    torch.save(model.state_dict(), 'lstm_execution_time_model.pth')
    print("Model saved to lstm_execution_time_model.pth")
    print("Training history plot saved to training_history.png")

if __name__ == "__main__":
    # Define dataset path
    dataset_path = 'lstm_dataset.npz'
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Train and evaluate
    train_and_evaluate(dataset_path, test_size=10, epochs=50, batch_size=32, device=device)
