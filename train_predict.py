import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# Custom Dataset class
class ExecutionTimeDataset(Dataset):
    def __init__(self, sequences, exec_times):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.exec_times = torch.tensor(exec_times, dtype=torch.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.exec_times[idx]

# Load and preprocess dataset
def load_and_preprocess_dataset():
    halide_sequences = np.load('halide_sequences.npy')
    halide_exec_times = np.load('halide_exec_times.npy')
    tiramisu_sequences = np.load('tiramisu_sequences.npy')
    tiramisu_exec_times = np.load('tiramisu_exec_times.npy')
    
    X = np.concatenate([halide_sequences, tiramisu_sequences], axis=0)
    y = np.concatenate([halide_exec_times, tiramisu_exec_times], axis=0)
    
    # Normalize input sequences
    X_scaler = StandardScaler()
    X_reshaped = X.reshape(-1, X.shape[-1])  # (n_samples * 100, 44)
    X_scaled = X_scaler.fit_transform(X_reshaped).reshape(X.shape)  # Back to (n_samples, 100, 44)
    
    # Log-transform and normalize execution times
    y_log = np.log1p(y)  # Handle small/zero values
    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y_log.reshape(-1, 1)).flatten()
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)
    
    print("Train Shape:", X_train.shape, y_train.shape)
    print("Validation Shape:", X_val.shape, y_val.shape)
    print("Test Shape:", X_test.shape, y_test.shape)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, X_scaler, y_scaler, y

# Improved LSTM model
class ImprovedExecutionTimePredictor(nn.Module):
    def __init__(self, input_dim=44, hidden_dim1=256, hidden_dim2=128, dropout=0.05):
        super(ImprovedExecutionTimePredictor, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.exp = nn.ReLU()  # Ensures non-negative output
    
    def forward(self, x):
        out, _ = self.lstm1(x)  # x: (batch_size, 100, 44)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)  # out: (batch_size, 100, hidden_dim2)
        out = self.dropout2(out[:, -1, :])  # Take last timestep: (batch_size, hidden_dim2)
        out = self.fc1(out)  # (batch_size, 64)
        out = self.relu(out)
        out = self.fc2(out)  # (batch_size, 1)
        out = self.exp(out)  # Ensure non-negative
        return out

# Training function
def train_model(model, train_loader, val_loader, device, epochs=200, patience=20):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses, val_losses, train_maes, val_maes = [], [], [], []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * sequences.size(0)
            train_mae += torch.abs(outputs.squeeze() - targets).sum().item()
        
        train_loss /= len(train_loader.dataset)
        train_mae /= len(train_loader.dataset)
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item() * sequences.size(0)
                val_mae += torch.abs(outputs.squeeze() - targets).sum().item()
        
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
    
    return train_losses, val_losses, train_maes, val_maes

# Plot training history
def plot_history(train_losses, val_losses, train_maes, val_maes):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_maes, label='Train MAE')
    plt.plot(val_maes, label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_improved.png')
    plt.close()

# Predict function for new data
def predict(model, X, X_scaler, y_scaler, device):
    model.eval()
    # Reshape and scale input
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = X_scaler.transform(X_reshaped).reshape(X.shape)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).squeeze().cpu().numpy()
    
    # Inverse transform predictions
    y_pred_log = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_pred = np.expm1(y_pred_log)  # Reverse log1p
    return y_pred

# Main execution
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, X_scaler, y_scaler, y_original = load_and_preprocess_dataset()
    
    # Create datasets and dataloaders
    train_dataset = ExecutionTimeDataset(X_train, y_train)
    val_dataset = ExecutionTimeDataset(X_val, y_val)
    test_dataset = ExecutionTimeDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Initialize model
    model = ImprovedExecutionTimePredictor(input_dim=44).to(device)
    
    # Train model
    train_losses, val_losses, train_maes, val_maes = train_model(model, train_loader, val_loader, device)
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    
    # Evaluate on test set
    test_loss = 0.0
    test_mae = 0.0
    y_pred_scaled = []
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            loss = nn.MSELoss()(outputs.squeeze(), targets)
            test_loss += loss.item() * sequences.size(0)
            test_mae += torch.abs(outputs.squeeze() - targets).sum().item()
            y_pred_scaled.extend(outputs.squeeze().cpu().numpy())
    
    test_loss /= len(test_loader.dataset)
    test_mae /= len(test_loader.dataset)
    print(f"\nTest Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Inverse transform test predictions
    y_pred_scaled = np.array(y_pred_scaled)
    y_pred_log = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_pred = np.expm1(y_pred_log)
    y_test_log = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_test_original = np.expm1(y_test_log)
    
    print("\nSample Predictions (seconds):", y_pred[:5])
    print("Sample Actuals (seconds):", y_test_original[:5])
    
    # Plot training history
    plot_history(train_losses, val_losses, train_maes, val_maes)
    
    # Save model and scalers
    torch.save(model.state_dict(), 'execution_predictor_improved.pt')
    np.save('X_scaler_mean.npy', X_scaler.mean_)
    np.save('X_scaler_scale.npy', X_scaler.scale_)
    np.save('y_scaler_mean.npy', y_scaler.mean_)
    np.save('y_scaler_scale.npy', y_scaler.scale_)
    print("\nModel and scalers saved.")
    
    # Example prediction on a single test sample
    sample_idx = 0
    sample_X = X_test[sample_idx:sample_idx+1]  # Shape: (1, 100, 44)
    sample_pred = predict(model, sample_X, X_scaler, y_scaler, device)
    print(f"\nExample Prediction for Test Sample {sample_idx}:")
    print(f"Predicted Execution Time: {sample_pred[0]:.6f} seconds")
    print(f"Actual Execution Time: {y_test_original[sample_idx]:.6f} seconds")

if __name__ == "__main__":
    required_files = ['halide_sequences.npy', 'halide_exec_times.npy', 
                      'tiramisu_sequences.npy', 'tiramisu_exec_times.npy']
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: {file} not found. Run transformer.py first.")
            exit(1)
    
    main()
