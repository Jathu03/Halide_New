import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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

# Enhanced Dataset class with data augmentation and Mixup
class ScheduleDataset(Dataset):
    def __init__(self, sequences, execution_times, augment=False, mixup_alpha=0.2):
        self.sequences = torch.FloatTensor(sequences.astype(np.float32))
        self.execution_times = torch.FloatTensor(execution_times).view(-1, 1)
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target = self.execution_times[idx]
        
        # Apply data augmentation during training
        if self.augment:
            # Add random noise
            if random.random() < 0.5:
                noise_factor = 0.05
                noise = torch.randn_like(seq) * noise_factor
                seq = seq + noise
            
            # Randomly mask timesteps
            if random.random() < 0.5:
                mask_prob = 0.1
                mask = torch.rand(seq.shape[0]) > mask_prob
                masked_seq = seq.clone()
                masked_seq[~mask] = 0.0
                seq = masked_seq
            
            # Mixup augmentation
            if random.random() < 0.3:
                mix_idx = random.randint(0, len(self.sequences) - 1)
                mix_seq = self.sequences[mix_idx]
                mix_target = self.execution_times[mix_idx]
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                seq = lam * seq + (1 - lam) * mix_seq
                target = lam * target + (1 - lam) * mix_target
        
        return seq, target

# Multi-Head Self-Attention Module
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
    
    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        scale = self.scale.to(x.device)
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(attention, V).permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, seq_len, hidden_dim)
        x = self.fc_out(x)
        return x, attention

# Enhanced LSTM model with multi-head attention and residual connections
class EnhancedLSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6, dropout=0.4):
        super(EnhancedLSTMRegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Bidirectional LSTM with residual connections
        self.lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim * 2
            self.lstm_layers.append(
                nn.LSTM(
                    in_dim,
                    hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    dropout=0.0  # Dropout handled externally
                )
            )
        
        # Multi-head self-attention
        self.attention = MultiHeadSelfAttention(hidden_dim * 2, num_heads=8)
        
        # Global context
        self.global_context = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Dense layers with skip connections
        self.fc1 = nn.Linear(hidden_dim * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, output_dim)
        
        self.skip1 = nn.Linear(hidden_dim * 4, 512)
        self.skip2 = nn.Linear(1024, 256)
        self.skip3 = nn.Linear(512, 128)
        
        self.norm1 = nn.LayerNorm(1024)
        self.norm2 = nn.LayerNorm(512)
        self.norm3 = nn.LayerNorm(256)
        self.norm4 = nn.LayerNorm(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        
        # LSTM with residual connections
        for i, lstm in enumerate(self.lstm_layers):
            lstm_out, _ = lstm(x)
            if i > 0:
                x = x + lstm_out  # Residual connection
            else:
                x = lstm_out
        
        # Multi-head attention
        attended_out, _ = self.attention(x)
        
        # Global context
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)
        context = self.global_context(avg_pool + max_pool)
        att_avg_pool = torch.mean(attended_out, dim=1)
        
        combined = torch.cat([context, att_avg_pool], dim=1)
        
        skip_connection1 = self.skip1(combined)
        out = self.fc1(combined)
        out = self.relu(out)
        out = self.norm1(out)
        out = self.dropout(out)
        
        skip_connection2 = self.skip2(out)
        out = self.fc2(out)
        out = out + skip_connection1
        out = self.relu(out)
        out = self.norm2(out)
        out = self.dropout(out)
        
        skip_connection3 = self.skip3(out)
        out = self.fc3(out)
        out = out + skip_connection2
        out = self.relu(out)
        out = self.norm3(out)
        out = self.dropout(out)
        
        out = self.fc4(out)
        out = out + skip_connection3
        out = self.relu(out)
        out = self.norm4(out)
        out = self.dropout(out)
        
        out = self.fc5(out)
        return out

# Enhanced data preprocessing with advanced feature engineering
def load_and_preprocess_dataset(data_dir="preprocessed_dataset"):
    sequence_data = np.load(f"{data_dir}/sequence_data.npy", allow_pickle=True)
    if sequence_data.dtype == object:
        sequence_data = np.stack(sequence_data).astype(np.float32)
    else:
        sequence_data = sequence_data.astype(np.float32)
    
    edge_df = pd.read_csv(f"{data_dir}/edge_features.csv")
    node_df = pd.read_csv(f"{data_dir}/node_features.csv")
    execution_times = np.load(f"{data_dir}/execution_times.npy", allow_pickle=True).astype(np.float32)
    
    enhanced_sequences = []
    for sequence in sequence_data:
        features = [sequence]
        seq_len, feat_dim = sequence.shape
        
        if seq_len > 1:
            # Trend (first derivative)
            trend = np.zeros_like(sequence)
            trend[1:, :] = sequence[1:, :] - sequence[:-1, :]
            features.append(trend)
            
            # Rolling statistics
            sequence_df = pd.DataFrame(sequence)
            rolling_mean = sequence_df.rolling(window=3, min_periods=1).mean().values
            rolling_std = sequence_df.rolling(window=3, min_periods=1).std().fillna(0).values
            features.append(rolling_mean)
            features.append(rolling_std)
            
            # Feature interactions
            for i in range(feat_dim):
                for j in range(i+1, min(i+3, feat_dim)):
                    interaction = sequence[:, i:i+1] * sequence[:, j:j+1]
                    features.append(interaction)
        
        enhanced_sequence = np.concatenate(features, axis=1)
        enhanced_sequences.append(enhanced_sequence)
    
    enhanced_sequences = np.array(enhanced_sequences)
    
    # Robust outlier handling
    execution_times_win = stats.mstats.winsorize(execution_times, limits=[0.02, 0.02])
    execution_times_log = np.log1p(execution_times_win)
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    execution_times_scaled = scaler.fit_transform(execution_times_log.reshape(-1, 1)).flatten()
    
    return enhanced_sequences, edge_df, node_df, execution_times, execution_times_scaled, scaler

# Custom loss with label smoothing
class SmoothHuberLoss(nn.Module):
    def __init__(self, delta=0.5, smoothing=0.1):
        super(SmoothHuberLoss, self).__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.smoothing = smoothing
    
    def forward(self, outputs, targets):
        smoothed_targets = targets * (1 - self.smoothing) + self.smoothing * torch.mean(targets)
        return self.huber(outputs, smoothed_targets)

# Advanced training function with k-fold validation
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, grad_accum_steps=4):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stop_patience = 30
    early_stop_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        accumulated_steps = 0
        
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss = loss / grad_accum_steps
            loss.backward()
            
            accumulated_steps += 1
            if accumulated_steps % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += (loss.item() * grad_accum_steps) * sequences.size(0)
        
        if accumulated_steps % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
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
        
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")
        
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
    
    y_pred_scaled = np.concatenate(predictions).flatten()
    y_true_scaled = np.concatenate(actuals).flatten()
    
    y_pred_log = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true_log = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_true_log)
    
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-7))) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
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
    
    mae_ci = np.percentile(mae_samples, [2.5, 97.5])
    mape_ci = np.percentile(mape_samples, [2.5, 97.5])
    rmse_ci = np.percentile(rmse_samples, [2.5, 97.5])
    
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
    os.makedirs("evaluation_results", exist_ok=True)
    
    # Loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("evaluation_results/loss_curves.png")
    plt.show()  # Display the plot
    plt.close()
    
    # Additional plots (scatter, error distribution)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.scatter(results['y_true'], results['y_pred'], alpha=0.6)
    max_val = max(np.max(results['y_true']), np.max(results['y_pred']))
    min_val = min(np.min(results['y_true']), np.min(results['y_pred']))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title('Predictions vs Actuals')
    plt.xlabel('Actual Execution Time (ms)')
    plt.ylabel('Predicted Execution Time (ms)')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    errors = results['y_pred'] - results['y_true']
    plt.hist(errors, bins=20, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error (ms)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    error_percentage = np.abs((results['y_pred'] - results['y_true']) / np.maximum(results['y_true'], 1e-7)) * 100
    plt.bar(range(len(error_percentage)), error_percentage)
    plt.axhline(y=np.mean(error_percentage), color='r', linestyle='--', label=f'Mean: {np.mean(error_percentage):.2f}%')
    plt.title('Error Percentage by Sample')
    plt.xlabel('Sample Index')
    plt.ylabel('Error Percentage (%)')
    plt.legend()
    plt.grid(True)
    
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
    
    df_results = pd.DataFrame({
        'Actual': results['y_true'],
        'Predicted': results['y_pred'],
        'Error': results['y_pred'] - results['y_true'],
        'Error_Percentage': np.abs((results['y_pred'] - results['y_true']) / np.maximum(results['y_true'], 1e-7)) * 100
    })
    df_results.to_csv("evaluation_results/detailed_results.csv", index=False)
    
    print("\nEvaluation Results Summary:")
    print(f"MAE: {metrics['mae']:.4f} ms (95% CI: [{metrics['mae_ci'][0]:.4f}, {metrics['mae_ci'][1]:.4f}])")
    print(f"MAPE: {metrics['mape']:.2f}% (95% CI: [{metrics['mape_ci'][0]:.2f}%, {metrics['mape_ci'][1]:.2f}%])")
    print(f"RMSE: {metrics['rmse']:.4f} ms (95% CI: [{metrics['rmse_ci'][0]:.4f}, {metrics['rmse_ci'][1]:.4f}])")
    
    return df_results

# Ensemble predictions with more samples
def create_ensemble_predictions(model, test_loader, scaler, device, num_samples=10):
    ensemble_predictions = []
    model.train()
    
    with torch.no_grad():
        for _ in range(num_samples):
            batch_predictions = []
            for sequences, _ in test_loader:
                sequences = sequences.to(device)
                outputs = model(sequences)
                batch_predictions.append(outputs.cpu().numpy())
            sample_predictions = np.concatenate(batch_predictions).flatten()
            ensemble_predictions.append(sample_predictions)
    
    y_pred_scaled = np.mean(ensemble_predictions, axis=0)
    y_pred_log = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_pred = np.expm1(y_pred_log)
    return y_pred

# Main execution function with k-fold validation
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    sequence_data, edge_df, node_df, execution_times, execution_times_scaled, scaler = load_and_preprocess_dataset()
    print("Enhanced Sequence Data Shape:", sequence_data.shape)
    print("Execution Times Shape:", execution_times.shape)
    
    n_bins = min(5, len(np.unique(execution_times_scaled)))
    bins = pd.qcut(execution_times_scaled, n_bins, labels=False, duplicates='drop')
    
    X_temp, X_holdout, y_temp, y_holdout, bins_temp, _ = train_test_split(
        sequence_data, execution_times_scaled, bins, test_size=10, random_state=42, stratify=bins
    )
    
    # K-fold cross-validation
    n_splits = 5
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_val_loss = float('inf')
    best_model = None
    best_train_losses = []
    best_val_losses = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_temp, y_temp, bins_temp)):
        print(f"\nFold {fold+1}/{n_splits}")
        X_train, X_val = X_temp[train_idx], X_temp[val_idx]
        y_train, y_val = y_temp[train_idx], y_temp[val_idx]
        
        train_dataset = ScheduleDataset(X_train, y_train, augment=True)
        val_dataset = ScheduleDataset(X_val, y_val, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        input_dim = X_train.shape[2]
        hidden_dim = 512  # Increased
        output_dim = 1
        num_layers = 6
        
        model = EnhancedLSTMRegressor(input_dim, hidden_dim, output_dim, num_layers=num_layers).to(device)
        criterion = SmoothHuberLoss(delta=0.5, smoothing=0.05)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=2e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        
        train_losses, val_losses = train_model(
            model, 
            train_loader, 
            val_loader, 
            criterion, 
            optimizer, 
            scheduler, 
            num_epochs=200, 
            device=device,
            grad_accum_steps=4
        )
        
        checkpoint = torch.load("best_model.pth")
        if checkpoint['val_loss'] < best_val_loss:
            best_val_loss = checkpoint['val_loss']
            best_model = model
            best_train_losses = train_losses
            best_val_losses = val_losses
            torch.save(checkpoint, "best_model_kfold.pth")
    
    print(f"\nBest model from k-fold with validation loss: {best_val_loss:.6f}")
    
    test_dataset = ScheduleDataset(X_holdout, y_holdout, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    results = evaluate_model(best_model, test_loader, scaler, device)
    df_results = plot_and_save_results(best_train_losses, best_val_losses, results)
    
    ensemble_pred = create_ensemble_predictions(best_model, test_loader, scaler, device, num_samples=10)
    
    ensemble_mae = np.mean(np.abs(results['y_true'] - ensemble_pred))
    ensemble_mape = np.mean(np.abs((results['y_true'] - ensemble_pred) / np.maximum(results['y_true'], 1e-7))) * 100
    
    print("\nEnsemble Prediction Results:")
    print(f"Ensemble MAE: {ensemble_mae:.4f} ms")
    print(f"Ensemble MAPE: {ensemble_mape:.2f}%")
    
    print("\nPredictions for Holdout Test Set (10 Samples):")
    print("Sample | Actual (ms) | Predicted (ms) | Ensemble Pred (ms) | Pred Error (%) | Ens Error (%)")
    print("-" * 90)
    
    for i in range(len(results['y_true'])):
        actual = results['y_true'][i]
        pred = results['y_pred'][i]
        ens_pred = ensemble_pred[i]
        pred_err = abs(pred - actual) / max(actual, 1e-7) * 100
        ens_err = abs(ens_pred - actual) / max(actual, 1e-7) * 100
        print(f"{i+1:6d} | {actual:11.4f} | {pred:13.4f} | {ens_pred:16.4f} | {pred_err:13.2f} | {ens_err:13.2f}")
    
    # Save predictions
    pred_df = pd.DataFrame({
        'Sample': range(1, len(results['y_true']) + 1),
        'Actual': results['y_true'],
        'Predicted': results['y_pred'],
        'Ensemble_Predicted': ensemble_pred,
        'Prediction_Error_Percentage': np.abs((results['y_pred'] - results['y_true']) / np.maximum(results['y_true'], 1e-7)) * 100,
        'Ensemble_Error_Percentage': np.abs((ensemble_pred - results['y_true']) / np.maximum(results['y_true'], 1e-7)) * 100
    })
    pred_df.to_csv("evaluation_results/predictions.csv", index=False)
    print("\nPredictions saved to 'evaluation_results/predictions.csv'")

if __name__ == "__main__":
    main()
