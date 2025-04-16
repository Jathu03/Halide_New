import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import matplotlib.pyplot as plt
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time

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
    Split the dataset into training, validation and test sets with improved preprocessing.
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
    
    # Apply MinMaxScaler to the execution times to improve training stability
    y_scaler = MinMaxScaler()
    execution_times_scaled = y_scaler.fit_transform(execution_times.reshape(-1, 1)).flatten()
    
    # Print some statistics about the target variable
    print(f"Execution times statistics before scaling:")
    print(f"  Min: {np.min(execution_times)}, Max: {np.max(execution_times)}")
    print(f"  Mean: {np.mean(execution_times)}, Std: {np.std(execution_times)}")
    print(f"Execution times statistics after scaling:")
    print(f"  Min: {np.min(execution_times_scaled)}, Max: {np.max(execution_times_scaled)}")
    
    # Create the entire dataset with scaled execution times
    full_dataset = HalideDataset(sequences, execution_times_scaled)
    
    # Create dataset splits using indices
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f"Split dataset into:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Test: {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset, y_scaler


class ImprovedLSTMModel(nn.Module):
    """
    Improved LSTM model for execution time prediction with residual connections and layer normalization
    """
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, dropout=0.3):
        super(ImprovedLSTMModel, self).__init__()
        
        # First layer: bidirectional LSTM
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size1,
            batch_first=True,
            bidirectional=True  # Use bidirectional LSTM for better feature extraction
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_size1 * 2)  # *2 because bidirectional
        self.dropout1 = nn.Dropout(dropout)
        
        # Second layer: standard LSTM
        self.lstm2 = nn.LSTM(
            input_size=hidden_size1 * 2,
            hidden_size=hidden_size2,
            batch_first=True
        )
        
        self.layer_norm2 = nn.LayerNorm(hidden_size2)
        self.dropout2 = nn.Dropout(dropout)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size2, 1),
            nn.Softmax(dim=1)
        )
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size2, 32)
        self.layer_norm3 = nn.LayerNorm(32)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 16)
        self.layer_norm4 = nn.LayerNorm(16)
        self.dropout4 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(16, 1)
        
    def forward(self, x):
        # First LSTM layer (bidirectional)
        out, _ = self.lstm1(x)
        out = self.layer_norm1(out)
        out = self.dropout1(out)
        
        # Second LSTM layer
        out, _ = self.lstm2(out)
        out = self.layer_norm2(out)
        out = self.dropout2(out)
        
        # Attention mechanism
        attention_weights = self.attention(out)
        out = torch.sum(attention_weights * out, dim=1)
        
        # Fully connected layers with residual connections
        residual = out
        out = self.fc1(out)
        out = self.layer_norm3(out)
        out = self.relu(out)
        out = self.dropout3(out)
        
        out = self.fc2(out)
        out = self.layer_norm4(out)
        out = self.relu(out)
        out = self.dropout4(out)
        
        out = self.fc3(out)
        
        return out


def train_model(model, train_loader, val_loader, epochs=150, learning_rate=0.0005, patience=15, 
                weight_decay=1e-5, scheduler_factor=0.5, scheduler_patience=5):
    """
    Train the PyTorch LSTM model with improved training procedure
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=True
    )
    
    # For early stopping
    best_val_loss = float('inf')
    counter = 0
    best_model = None
    
    # Training history
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    
    print(f"Starting training with learning rate: {learning_rate}")
    start_time = time.time()
    
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
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Print stats
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, '
              f'Train MAE: {train_mae:.4f}, Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            counter = 0
            # Save best model
            torch.save(best_model, 'best_lstm_model.pth')
            print(f"Saved new best model with validation loss: {val_loss:.4f}")
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load the best model
    model.load_state_dict(best_model)
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    return model, {'train_loss': train_losses, 'val_loss': val_losses, 
                  'train_mae': train_maes, 'val_mae': val_maes}


def evaluate_model(model, test_loader, y_scaler):
    """
    Evaluate the model on the test set and calculate error percentages
    with inverse transform to get original scale predictions
    """
    model.eval()
    
    y_true_scaled = []
    y_pred_scaled = []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            
            y_true_scaled.extend(targets.cpu().numpy().flatten())
            y_pred_scaled.extend(outputs.cpu().numpy().flatten())
    
    # Inverse transform the scaled predictions and targets
    y_true_scaled = np.array(y_true_scaled).reshape(-1, 1)
    y_pred_scaled = np.array(y_pred_scaled).reshape(-1, 1)
    
    y_true = y_scaler.inverse_transform(y_true_scaled).flatten()
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    
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
    
    return y_pred, absolute_errors, percentage_errors, y_true


def plot_results(history, y_true, y_pred):
    """
    Plot training history and prediction results with improved visualizations
    """
    # Create a figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot training & validation loss with log scale
    ax1.semilogy(history['train_loss'], label='Train')
    ax1.semilogy(history['val_loss'], label='Validation')
    ax1.set_title('Model Loss (log scale)')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_xlabel('Epoch')
    ax1.legend(loc='upper right')
    ax1.grid(True, which="both", ls="-")
    
    # Plot training & validation MAE
    ax2.plot(history['train_mae'], label='Train')
    ax2.plot(history['val_mae'], label='Validation')
    ax2.set_title('Mean Absolute Error')
    ax2.set_ylabel('MAE')
    ax2.set_xlabel('Epoch')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    
    # Plot actual vs predicted values
    ax3.scatter(y_true, y_pred, alpha=0.8)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred)) * 1.1
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    ax3.set_title('Actual vs Predicted Execution Times')
    ax3.set_xlabel('Actual Time (ms)')
    ax3.set_ylabel('Predicted Time (ms)')
    ax3.legend()
    ax3.grid(True)
    
    # Add equal aspect ratio to make the comparison clearer
    ax3.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig('lstm_results_pytorch_improved.png', dpi=300)
    
    # Create additional plot for error analysis
    plt.figure(figsize=(12, 6))
    error_percentages = (np.abs(y_pred - y_true) / np.clip(np.abs(y_true), 1e-10, None)) * 100
    plt.bar(range(len(y_true)), error_percentages)
    plt.axhline(np.mean(error_percentages), color='r', linestyle='--', label=f'Mean Error: {np.mean(error_percentages):.2f}%')
    plt.title('Percentage Error by Test Sample')
    plt.xlabel('Test Sample')
    plt.ylabel('Error (%)')
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig('error_analysis.png', dpi=300)
    
    plt.show()


def main():
    # Load the dataset
    sequences, execution_times = load_dataset()
    
    # Split data into training, validation, and test sets
    train_dataset, val_dataset, test_dataset, y_scaler = prepare_train_val_test_split(
        sequences, execution_times, test_size=20
    )
    
    # Create data loaders with smaller batch size for more gradient updates
    batch_size = 16  # Reduced from 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=1)  # Use batch size 1 for detailed evaluation
    
    # Get sample input shape
    sample_input = next(iter(train_loader))[0]
    input_size = sample_input.shape[2]  # Feature dimension
    print(f"Input feature size: {input_size}")
    
    # Initialize the improved model
    model = ImprovedLSTMModel(input_size=input_size).to(device)
    print(model)
    
    # Count the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    # Train the model with improved hyperparameters
    model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        epochs=200,  # More epochs with early stopping
        learning_rate=0.0005,  # Lower learning rate
        patience=20,  # More patience for early stopping
        weight_decay=1e-5,  # L2 regularization
        scheduler_factor=0.7,  # More gradual learning rate reduction
        scheduler_patience=8  # Wait more epochs before reducing LR
    )
    
    # Evaluate the model on the test set
    y_pred, absolute_errors, percentage_errors, y_true = evaluate_model(model, test_loader, y_scaler)
    
    # Plot results
    plot_results(history, y_true, y_pred)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'y_scaler': y_scaler
    }, 'final_lstm_model.pth')
    
    print("\nModel training and evaluation complete!")
    print("Full model saved to 'final_lstm_model.pth'")


if __name__ == "__main__":
    main()
