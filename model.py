import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import matplotlib.pyplot as plt
import os
import random
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class HalideDataset(Dataset):
    """
    PyTorch Dataset for Halide execution time prediction
    """
    def __init__(self, sequences, execution_times):
        self.sequences = torch.FloatTensor(sequences)
        self.execution_times = torch.FloatTensor(execution_times).reshape(-1, 1)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.execution_times[idx]


def load_dataset(file_path='halide_data.npz'):
    """
    Load the dataset created by the preprocessing script
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file {file_path} not found. Please run the preprocessing script first.")
    
    data = np.load(file_path)
    sequences = data['sequences']
    execution_times = data['execution_times']
    
    print(f"Loaded dataset with {len(sequences)} samples")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Execution times shape: {execution_times.shape}")
    
    return sequences, execution_times


def prepare_train_val_test_split(sequences, execution_times, test_size=20):
    """
    Split the dataset into training, validation and test sets.
    Keep exactly 20 samples for testing as specified.
    """
    # Get total number of samples
    n_samples = len(sequences)
    
    if n_samples <= test_size:
        raise ValueError(f"Not enough samples ({n_samples}) to create a test set of {test_size} samples")
    
    # Create indices and shuffle them
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Select test indices (exactly 20 samples)
    test_indices = indices[:test_size]
    remaining_indices = indices[test_size:]
    
    # Split the remaining data into training and validation (80/20 split)
    train_indices, val_indices = train_test_split(remaining_indices, test_size=0.2, random_state=42)
    
    # Create the entire dataset
    full_dataset = HalideDataset(sequences, execution_times)
    
    # Create dataset splits using indices
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f"Split dataset into:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset


class LSTMModel(nn.Module):
    """
    LSTM model for execution time prediction
    """
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size1,
            batch_first=True
        )
        
        self.dropout1 = nn.Dropout(dropout)
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_size1,
            hidden_size=hidden_size2,
            batch_first=True
        )
        
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(hidden_size2, 32)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # First LSTM layer
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        
        # Second LSTM layer
        out, _ = self.lstm2(out)
        # Take the output from the last time step
        out = out[:, -1, :]
        out = self.dropout2(out)
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        
        return out


def train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001, patience=10):
    """
    Train the PyTorch LSTM model with early stopping
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # For early stopping
    best_val_loss = float('inf')
    counter = 0
    best_model = None
    
    # Training history
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(outputs - targets)).item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_mae /= len(train_loader)
        train_maes.append(train_mae)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_mae += torch.mean(torch.abs(outputs - targets)).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_mae /= len(val_loader)
        val_maes.append(val_mae)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            counter = 0
            # Save best model
            torch.save(best_model, 'best_lstm_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load the best model
    model.load_state_dict(best_model)
    
    return model, {'train_loss': train_losses, 'val_loss': val_losses, 
                  'train_mae': train_maes, 'val_mae': val_maes}


def evaluate_model(model, test_loader):
    """
    Evaluate the model on the test set and calculate error percentages
    """
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            
            y_true.extend(targets.cpu().numpy().flatten())
            y_pred.extend(outputs.cpu().numpy().flatten())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate absolute errors
    absolute_errors = np.abs(y_pred - y_true)
    
    # Calculate percentage errors
    percentage_errors = (absolute_errors / np.clip(np.abs(y_true), 1e-10, None)) * 100
    
    # Calculate mean and standard deviation of errors
    mean_absolute_error = np.mean(absolute_errors)
    mean_percentage_error = np.mean(percentage_errors)
    
    print("\nTest Set Evaluation:")
    print(f"Mean Absolute Error: {mean_absolute_error:.4f}")
    print(f"Mean Percentage Error: {mean_percentage_error:.2f}%")
    
    # Print individual predictions and errors
    print("\nIndividual Test Sample Results:")
    print("Sample | Actual Time | Predicted Time | Abs Error | Error %")
    print("-" * 70)
    
    for i in range(len(y_true)):
        print(f"{i:6d} | {y_true[i]:11.2f} | {y_pred[i]:14.2f} | {absolute_errors[i]:9.2f} | {percentage_errors[i]:7.2f}%")
    
    return y_pred, absolute_errors, percentage_errors


def plot_results(history, y_true, y_pred):
    """
    Plot training history and prediction results
    """
    # Create a figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot training & validation loss
    ax1.plot(history['train_loss'])
    ax1.plot(history['val_loss'])
    ax1.set_title('Model Loss')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper right')
    ax1.grid(True)
    
    # Plot training & validation MAE
    ax2.plot(history['train_mae'])
    ax2.plot(history['val_mae'])
    ax2.set_title('Mean Absolute Error')
    ax2.set_ylabel('MAE')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')
    ax2.grid(True)
    
    # Plot actual vs predicted values
    ax3.scatter(y_true, y_pred)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--')
    ax3.set_title('Actual vs Predicted Execution Times')
    ax3.set_xlabel('Actual Time (ms)')
    ax3.set_ylabel('Predicted Time (ms)')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('lstm_results_pytorch.png')
    plt.show()


def main():
    # Load the dataset
    sequences, execution_times = load_dataset()
    
    # Split data into training, validation, and test sets
    train_dataset, val_dataset, test_dataset = prepare_train_val_test_split(
        sequences, execution_times, test_size=20
    )
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Get sample input shape (assuming we're dealing with a sequence dataset)
    sample_input = next(iter(train_loader))[0]
    input_size = sample_input.shape[2]  # Feature dimension
    print(f"Input feature size: {input_size}")
    
    # Initialize the model
    model = LSTMModel(input_size=input_size).to(device)
    print(model)
    
    # Count the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    # Train the model
    model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=100, 
        learning_rate=0.001,
        patience=10
    )
    
    # Evaluate the model on the test set
    # Extract all test data for evaluation
    all_test_sequences = []
    all_test_targets = []
    
    for test_sequences, test_targets in test_loader:
        all_test_sequences.extend(test_sequences.numpy())
        all_test_targets.extend(test_targets.numpy().flatten())
    
    all_test_sequences = np.array(all_test_sequences)
    all_test_targets = np.array(all_test_targets)
    
    # Create a new DataLoader with batch size 1 for individual predictions
    individual_test_dataset = HalideDataset(all_test_sequences, all_test_targets)
    individual_test_loader = DataLoader(individual_test_dataset, batch_size=1)
    
    # Evaluate
    y_pred, absolute_errors, percentage_errors = evaluate_model(model, individual_test_loader)
    
    # Plot results
    plot_results(history, all_test_targets, y_pred)
    
    print("\nModel training and evaluation complete!")


if __name__ == "__main__":
    main()
