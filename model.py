import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import OneCycleLR
import random
from scipy import stats

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Enhanced Dataset class with data augmentation
class ScheduleDataset(Dataset):
    def __init__(self, sequences, execution_times, augment=False):
        self.sequences = torch.FloatTensor(sequences.astype(np.float32))
        self.execution_times = torch.FloatTensor(execution_times).view(-1, 1)
        self.augment = augment
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target = self.execution_times[idx]
        
        # Apply data augmentation during training
        if self.augment and random.random() < 0.5:
            # Add small random noise to features
            noise_factor = 0.05
            noise = torch.randn_like(seq) * noise_factor
            seq = seq + noise
            
            # Randomly mask some timesteps (simulate missing data)
            mask_prob = 0.1
            mask = torch.rand(seq.shape[0]) > mask_prob
            masked_seq = seq.clone()
            masked_seq[~mask] = 0.0
            seq = masked_seq
            
        return seq, target

# Self-Attention Module
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).cuda()
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.shape
        
        Q = self.query(x)  # (batch_size, seq_len, hidden_dim)
        K = self.key(x)    # (batch_size, seq_len, hidden_dim)
        V = self.value(x)  # (batch_size, seq_len, hidden_dim)
        
        # Scaled dot-product attention
        # (batch_size, seq_len, hidden_dim) x (batch_size, hidden_dim, seq_len)
        energy = torch.matmul(Q, K.permute(0, 2, 1)) / self.scale
        
        # (batch_size, seq_len, seq_len)
        attention = torch.softmax(energy, dim=-1)
        
        # (batch_size, seq_len, seq_len) x (batch_size, seq_len, hidden_dim)
        x = torch.matmul(attention, V)
        
        return x, attention

# Enhanced LSTM model with attention and advanced features
class EnhancedLSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, dropout=0.3):
        super(EnhancedLSTMRegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Feature extraction layer
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Self-attention mechanism
        self.attention = SelfAttention(hidden_dim * 2)  # *2 for bidirectional
        
        # Global context aggregation
        self.global_context = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Prediction network with skip connections
        self.fc1 = nn.Linear(hidden_dim * 4, 512)  # *4 due to concatenation
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_dim)
        
        # Skip connections
        self.skip1 = nn.Linear(hidden_dim * 4, 256)
        self.skip2 = nn.Linear(512, 128)
        self.skip3 = nn.Linear(256, 64)
        
        # Normalization and activation
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(256)
        self.norm3 = nn.LayerNorm(128)
        self.norm4 = nn.LayerNorm(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Output activation - ensures positive predictions
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Initial feature extraction
        x = self.feature_extractor(x)  # (batch_size, seq_len, hidden_dim)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim*2)
        
        # Self-attention
        attended_out, _ = self.attention(lstm_out)  # (batch_size, seq_len, hidden_dim*2)
        
        # Global temporal context
        avg_pool = torch.mean(lstm_out, dim=1)  # (batch_size, hidden_dim*2)
        max_pool, _ = torch.max(lstm_out, dim=1)  # (batch_size, hidden_dim*2)
        
        # Enhanced representation
        context = self.global_context(avg_pool + max_pool)  # (batch_size, hidden_dim*2)
        
        # Attention-weighted sequence representation
        att_avg_pool = torch.mean(attended_out, dim=1)  # (batch_size, hidden_dim*2)
        
        # Concatenate multiple views of the sequence
        combined = torch.cat([context, att_avg_pool], dim=1)  # (batch_size, hidden_dim*4)
        
        # Dense layers with skip connections for gradient flow
        skip_connection1 = self.skip1(combined)  # (batch_size, 256)
        
        out = self.fc1(combined)  # (batch_size, 512)
        out = self.relu(out)
        out = self.norm1(out)
        out = self.dropout(out)
        
        skip_connection2 = self.skip2(out)  # (batch_size, 128)
        
        out = self.fc2(out)  # (batch_size, 256)
        out = out + skip_connection1  # First skip connection
        out = self.relu(out)
        out = self.norm2(out)
        out = self.dropout(out)
        
        skip_connection3 = self.skip3(out)  # (batch_size, 64)
        
        out = self.fc3(out)  # (batch_size, 128)
        out = out + skip_connection2  # Second skip connection
        out = self.relu(out)
        out = self.norm3(out)
        out = self.dropout(out)
        
        out = self.fc4(out)  # (batch_size, 64)
        out = out + skip_connection3  # Third skip connection
        out = self.relu(out)
        out = self.norm4(out)
        out = self.dropout(out)
        
        out = self.fc5(out)  # (batch_size, output_dim)
        
        # Scaling factor for better initial convergence
        return out

# Enhanced data preprocessing with feature engineering
def load_and_preprocess_dataset(data_dir="preprocessed_dataset"):
    sequence_data = np.load(f"{data_dir}/sequence_data.npy", allow_pickle=True)
    if sequence_data.dtype == object:
        sequence_data = np.stack(sequence_data).astype(np.float32)
    else:
        sequence_data = sequence_data.astype(np.float32)
    
    edge_df = pd.read_csv(f"{data_dir}/edge_features.csv")
    node_df = pd.read_csv(f"{data_dir}/node_features.csv")
    execution_times = np.load(f"{data_dir}/execution_times.npy", allow_pickle=True).astype(np.float32)
    
    # Enhance sequence data with additional statistical features
    enhanced_sequences = []
    for sequence in sequence_data:
        # Calculate statistical features across the sequence
        # For each feature column, add: trend, variance, skewness
        features = []
        
        # Original features
        features.append(sequence)
        
        # Get sequence shape
        seq_len, feat_dim = sequence.shape
        
        # Add temporal features
        if seq_len > 1:
            # Add trend information (first derivative)
            trend = np.zeros_like(sequence)
            trend[1:, :] = sequence[1:, :] - sequence[:-1, :]
            features.append(trend)
            
            # Add feature interactions
            for i in range(feat_dim):
                for j in range(i+1, feat_dim):
                    interaction = sequence[:, i:i+1] * sequence[:, j:j+1]
                    features.append(interaction)
        
        # Concatenate all features
        enhanced_sequence = np.concatenate(features, axis=1)
        enhanced_sequences.append(enhanced_sequence)
    
    enhanced_sequences = np.array(enhanced_sequences)
    
    # Enhanced execution time transformation
    # Identify and handle outliers with winsorization
    execution_times_win = stats.mstats.winsorize(execution_times, limits=[0.05, 0.05])
    
    # Log transform for skewness reduction
    execution_times_log = np.log1p(execution_times_win)
    
    # Use Robust Scaler to handle remaining outliers better
    scaler = RobustScaler()
    execution_times_scaled = scaler.fit_transform(execution_times_log.reshape(-1, 1)).flatten()
    
    return enhanced_sequences, edge_df, node_df, execution_times, execution_times_scaled, scaler

# Advanced training function with learning rate finder and gradient accumulation
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, grad_accum_steps=1):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stop_patience = 50
    early_stop_counter = 0
    
    # Learning rate warm-up
    warmup_epochs = 5
    warmup_factor = 0.1
    
    # Exponential Moving Average (EMA) model
    ema_model = None
    ema_decay = 0.999
    
    # Gradient accumulation setup
    optimizer.zero_grad()
    accumulated_steps = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # For warm-up
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = optimizer.param_groups[0]['lr'] * (warmup_factor + (1 - warmup_factor) * (epoch / warmup_epochs))
        
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / grad_accum_steps
            loss.backward()
            
            accumulated_steps += 1
            
            # Step optimizer and reset gradients after accumulation steps
            if accumulated_steps % grad_accum_steps == 0:
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += (loss.item() * grad_accum_steps) * sequences.size(0)
        
        # Make sure to handle any remaining accumulated gradients
        if accumulated_steps % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * sequences.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Update learning rate scheduler
        if epoch >= warmup_epochs:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, "best_model.pth")
            print(f"Saved new best model with validation loss: {val_loss:.6f}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered after {early_stop_patience} epochs without improvement")
                break
    
    # Load best model
    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1} with validation loss: {checkpoint['val_loss']:.6f}")
    
    return train_losses, val_losses

# Enhanced evaluation with confidence intervals
def evaluate_model(model, test_loader, scaler, device):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            predictions.append(outputs.cpu().numpy())
            actuals.append(targets.numpy())
    
    # Convert to numpy arrays
    y_pred_scaled = np.concatenate(predictions).flatten()
    y_true_scaled = np.concatenate(actuals).flatten()
    
    # Inverse transform
    y_pred_log = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true_log = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    
    # Convert back to original scale
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_true_log)
    
    # Calculate metrics
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-7))) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Calculate confidence intervals using bootstrap
    n_bootstrap = 1000
    mae_samples = []
    mape_samples = []
    rmse_samples = []
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_sample = y_true[indices]
        y_pred_sample = y_pred[indices]
        
        mae_sample = np.mean(np.abs(y_true_sample - y_pred_sample))
        mape_sample = np.mean(np.abs((y_true_sample - y_pred_sample) / np.maximum(y_true_sample, 1e-7))) * 100
        rmse_sample = np.sqrt(np.mean((y_true_sample - y_pred_sample) ** 2))
        
        mae_samples.append(mae_sample)
        mape_samples.append(mape_sample)
        rmse_samples.append(rmse_sample)
    
    # Calculate 95% confidence intervals
    mae_ci = np.percentile(mae_samples, [2.5, 97.5])
    mape_ci = np.percentile(mape_samples, [2.5, 97.5])
    rmse_ci = np.percentile(rmse_samples, [2.5, 97.5])
    
    # Return results
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'metrics': {
            'mae': mae,
            'mae_ci': mae_ci,
            'mape': mape,
            'mape_ci': mape_ci,
            'rmse': rmse,
            'rmse_ci': rmse_ci
        }
    }

# Plot and save detailed evaluation results
def plot_and_save_results(train_losses, val_losses, results):
    # Create a directory for results
    os.makedirs("evaluation_results", exist_ok=True)
    
    # 1. Training and validation loss
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("evaluation_results/loss_curves.png")
    plt.close()
    
    # 2. Predictions vs Actuals
    plt.figure(figsize=(12, 8))
    
    # Scatter plot
    plt.subplot(2, 2, 1)
    plt.scatter(results['y_true'], results['y_pred'], alpha=0.6)
    max_val = max(np.max(results['y_true']), np.max(results['y_pred']))
    min_val = min(np.min(results['y_true']), np.min(results['y_pred']))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title('Predictions vs Actuals')
    plt.xlabel('Actual Execution Time (ms)')
    plt.ylabel('Predicted Execution Time (ms)')
    plt.grid(True)
    
    # Histogram of errors
    plt.subplot(2, 2, 2)
    errors = results['y_pred'] - results['y_true']
    plt.hist(errors, bins=20, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error (ms)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Error percentage plot
    plt.subplot(2, 2, 3)
    error_percentage = np.abs((results['y_pred'] - results['y_true']) / np.maximum(results['y_true'], 1e-7)) * 100
    plt.bar(range(len(error_percentage)), error_percentage)
    plt.axhline(y=np.mean(error_percentage), color='r', linestyle='--', label=f'Mean: {np.mean(error_percentage):.2f}%')
    plt.title('Error Percentage by Sample')
    plt.xlabel('Sample Index')
    plt.ylabel('Error Percentage (%)')
    plt.legend()
    plt.grid(True)
    
    # Detailed metrics
    plt.subplot(2, 2, 4)
    metrics = results['metrics']
    plt.axis('off')
    info_text = f"""
    Evaluation Metrics:
    
    MAE: {metrics['mae']:.4f} ms
    95% CI: [{metrics['mae_ci'][0]:.4f}, {metrics['mae_ci'][1]:.4f}]
    
    MAPE: {metrics['mape']:.2f}%
    95% CI: [{metrics['mape_ci'][0]:.2f}%, {metrics['mape_ci'][1]:.2f}%]
    
    RMSE: {metrics['rmse']:.4f} ms
    95% CI: [{metrics['rmse_ci'][0]:.4f}, {metrics['rmse_ci'][1]:.4f}]
    """
    plt.text(0.1, 0.5, info_text, fontsize=12)
    
    plt.tight_layout()
    plt.savefig("evaluation_results/prediction_analysis.png")
    plt.close()
    
    # 3. Sample-by-sample comparison
    plt.figure(figsize=(14, 7))
    x = np.arange(len(results['y_true']))
    width = 0.35
    plt.bar(x - width/2, results['y_true'], width, label='Actual')
    plt.bar(x + width/2, results['y_pred'], width, label='Predicted')
    plt.xlabel('Sample Index')
    plt.ylabel('Execution Time (ms)')
    plt.title('Sample-by-Sample Comparison')
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig("evaluation_results/sample_comparison.png")
    plt.close()
    
    # Save detailed results to CSV
    df_results = pd.DataFrame({
        'Actual': results['y_true'],
        'Predicted': results['y_pred'],
        'Error': results['y_pred'] - results['y_true'],
        'Error_Percentage': np.abs((results['y_pred'] - results['y_true']) / np.maximum(results['y_true'], 1e-7)) * 100
    })
    df_results.to_csv("evaluation_results/detailed_results.csv", index=False)
    
    # Print summary statistics
    print("\nEvaluation Results Summary:")
    print(f"MAE: {metrics['mae']:.4f} ms (95% CI: [{metrics['mae_ci'][0]:.4f}, {metrics['mae_ci'][1]:.4f}])")
    print(f"MAPE: {metrics['mape']:.2f}% (95% CI: [{metrics['mape_ci'][0]:.2f}%, {metrics['mape_ci'][1]:.2f}%])")
    print(f"RMSE: {metrics['rmse']:.4f} ms (95% CI: [{metrics['rmse_ci'][0]:.4f}, {metrics['rmse_ci'][1]:.4f}])")
    
    return df_results

# Main execution function
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enhanced data loading and preprocessing
    sequence_data, edge_df, node_df, execution_times, execution_times_scaled, scaler = load_and_preprocess_dataset()
    print("Enhanced Sequence Data Shape:", sequence_data.shape)
    print("Execution Times Shape:", execution_times.shape)
    
    # Create stratified splits based on execution time distribution
    # First, create bins for stratification
    n_bins = 5
    bins = pd.qcut(execution_times_scaled, n_bins, labels=False, duplicates='drop')
    
    # Split into train+val and holdout test set (10 samples)
    X_temp, X_holdout, y_temp, y_holdout, bins_temp, _ = train_test_split(
        sequence_data, execution_times_scaled, bins, test_size=10, random_state=42, stratify=bins
    )
    
    # Ensure stratification in train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=bins_temp
    )
    
    print("Train Shape:", X_train.shape, y_train.shape)
    print("Validation Shape:", X_val.shape, y_val.shape)
    print("Holdout Test Shape:", X_holdout.shape, y_holdout.shape)
    
    # Create datasets and dataloaders with augmentation
    train_dataset = ScheduleDataset(X_train, y_train, augment=True)
    val_dataset = ScheduleDataset(X_val, y_val, augment=False)
    test_dataset = ScheduleDataset(X_holdout, y_holdout, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    # Model parameters
    input_dim = X_train.shape[2]  # Enhanced features dimension
    hidden_dim = 320  # Increased from 256
    output_dim = 1
    num_layers = 5  # Increased from 4
    
    # Initialize model
    model = EnhancedLSTMRegressor(input_dim, hidden_dim, output_dim, num_layers=num_layers).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss function - combine Huber and MSE
    criterion = nn.HuberLoss(delta=0.3)  # Reduced delta for finer control
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)
    
    # Learning rate scheduler - OneCycleLR for better convergence
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.002,
        steps_per_epoch=steps_per_epoch,
        epochs=300,
        pct_start=0.1,  # Spend 10% of training time in warm-up
        div_factor=10.0,  # Initial LR is max_lr/div_factor
        final_div_factor=100.0,  # Min LR is initial_lr/final_div_factor
        anneal_strategy='cos'  # Cosine annealing
    )
    
    # Train the model
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        scheduler, 
        num_epochs=300, 
        device=device,
        grad_accum_steps=2  # Gradient accumulation for stable training
    )
    
    # Evaluate on holdout test set
    results = evaluate_model(model, test_loader, scaler, device)
    
    # Plot and save detailed results
    df_results = plot_and_save_results(train_losses, val_losses, results)
    
    # Create ensemble prediction
    # Predict with slightly different dropout patterns
    ensemble_predictions = []
    model.train()  # Enable dropout
    with torch.no_grad():
        for _ in range(5):  # Generate 5 different predictions
            ensemble_pred = []
            for sequences, _ in test_loader:
                sequences = sequences.to(device)
                outputs = model(sequences)
                ensemble_pred.append(outputs.cpu().numpy())
            ensemble_pred = np.concatenate(ensemble_pred).flatten()
            ensemble_predictions.append(ensemble_pred)
    
    # Average ensemble predictions
    ensemble_pred_scaled = np.mean(ensemble_predictions, axis=0)
    
    # Inverse transform
    ensemble_pred_log = scaler.inverse_transform(ensemble_pred_scaled.reshape(-1, 1)).flatten()
    ensemble_pred = np.expm1(ensemble_pred_log)
    
    # Calculate ensemble metrics
    ensemble_mae = np.mean(np.abs(results['y_true'] - ensemble_pred))
    ensemble_mape = np.mean(np.abs((results['y_true'] - ensemble_pred) / np.maximum(results['y_true'], 1e-7))) * 100
    
    print("\nEnsemble Prediction Results:")
    print(f"Ensemble MAE: {ensemble_mae:.4f} ms")
    print(f"Ensemble MAPE: {ensemble_mape:.2f}%")
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'num_layers': num_layers
        },
        'scaler': [scaler.center_, scaler.scale_]
    }, "enhanced_lstm_regressor_model.pth")
    print("Enhanced model saved to enhanced_lstm_regressor_model.pth")

if __name__ == "__main__":
    main()
</antArtifact
