import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import matplotlib.pyplot as plt

# Define important metrics for scheduling sequence (schedule-specific)
important_metrics = [
    'bytes_at_production', 'bytes_at_realization', 'inner_parallelism', 'outer_parallelism',
    'num_vectors', 'points_computed_total', 'working_set'
]

# ... (keep extract_features_from_file, get_execution_time, process_directory, process_main_directory, prepare_data_for_model unchanged)

def create_data_loaders(train_sequences, y_train, test_sequences, y_test, batch_size=32):  # Increased batch size
    train_dataset = TensorDataset(train_sequences, y_train)
    test_dataset = TensorDataset(test_sequences, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        weights = torch.softmax(self.attention(x), dim=1)
        return torch.sum(x * weights, dim=1)

class EnhancedRecursiveLSTMModel(nn.Module):
    def __init__(self, seq_input_size, hidden_sizes=[256, 128, 64], output_size=1, dropout_rate=0.3, num_heads=4):  # Reduced hidden sizes, increased dropout
        super(EnhancedRecursiveLSTMModel, self).__init__()
        
        self.lstm_layers = nn.ModuleList()
        self.ln_layers = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        self.lstm_layers.append(nn.LSTM(seq_input_size, hidden_sizes[0], batch_first=True, bidirectional=True))
        self.ln_layers.append(nn.LayerNorm(hidden_sizes[0] * 2))
        self.residual_projs.append(nn.Linear(seq_input_size, hidden_sizes[0] * 2) if seq_input_size != hidden_sizes[0] * 2 else None)
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(nn.LSTM(hidden_sizes[i-1] * 2, hidden_sizes[i], batch_first=True, bidirectional=True))
            self.ln_layers.append(nn.LayerNorm(hidden_sizes[i] * 2))
            self.residual_projs.append(nn.Linear(hidden_sizes[i-1] * 2, hidden_sizes[i] * 2) if hidden_sizes[i-1] * 2 != hidden_sizes[i] * 2 else None)
        
        self.attention = nn.MultiheadAttention(hidden_sizes[-1] * 2, num_heads, dropout=dropout_rate, batch_first=True)
        self.attn_pool = AttentionPooling(hidden_sizes[-1] * 2)
        
        self.fc1 = nn.Linear(hidden_sizes[-1] * 2, 128)  # Reduced size
        self.bn1 = nn.BatchNorm1d(128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)  # Reduced size
        self.bn2 = nn.BatchNorm1d(64)
        self.ln2 = nn.LayerNorm(64)
        self.output_layer = nn.Linear(64, output_size)
        
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        self.final_residual_proj = nn.Linear(hidden_sizes[-1] * 2, 64) if hidden_sizes[-1] * 2 != 64 else None
    
    def forward(self, seq_input):
        lstm_out = seq_input
        for lstm, ln, res_proj in zip(self.lstm_layers, self.ln_layers, self.residual_projs):
            residual = lstm_out if res_proj is None else res_proj(lstm_out)
            lstm_out, _ = lstm(lstm_out)
            lstm_out = lstm_out + residual
            lstm_out = ln(lstm_out)
            lstm_out = self.dropout(lstm_out)
        
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        context = self.attn_pool(attn_out)
        
        x = self.fc1(context)
        x = self.bn1(x)
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.ln2(x)
        x = self.gelu(x)
        
        residual = context if self.final_residual_proj is None else self.final_residual_proj(context)
        x = x + residual
        x = self.dropout(x)
        output = self.output_layer(x)
        
        return output

def focal_loss(outputs, targets, alpha=0.5, gamma=1.5):  # Adjusted alpha and gamma for balance
    mse = (outputs - targets) ** 2
    pt = torch.exp(-mse)
    loss = alpha * (1 - pt) ** gamma * mse
    return torch.mean(loss)

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=500, patience=30, accumulation_steps=2, warmup_epochs=20):  # Reduced patience, increased warmup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        model.to(device)
        for lstm in model.lstm_layers:
            lstm.flatten_parameters()
    except RuntimeError as e:
        print(f"Error moving model to CUDA: {e}. Falling back to CPU.")
        device = torch.device('cpu')
        model.to(device)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, verbose=True)  # More aggressive LR reduction
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        # Learning rate warmup
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001 * lr_scale
        
        for i, (seq_inputs, targets) in enumerate(train_loader):
            seq_inputs, targets = seq_inputs.to(device), targets.to(device)
            outputs = model(seq_inputs)
            loss = criterion(outputs, targets)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss detected at epoch {epoch+1}, batch {i+1}")
                return None, None
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Stricter clipping
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps * seq_inputs.size(0)
        
        if len(train_loader) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq_inputs, targets in test_loader:
                seq_inputs, targets = seq_inputs.to(device), targets.to(device)
                outputs = model(seq_inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * seq_inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        
        if epoch >= warmup_epochs:
            scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss and not np.isnan(val_loss) and not np.isinf(val_loss):
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            model.load_state_dict(best_model_state)
            break
    
    if best_model_state is not None and epochs_no_improve > 0:
        model.load_state_dict(best_model_state)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.show()
    
    return train_losses, val_losses

# ... (keep evaluate_model unchanged)

def main(main_dir):
    if torch.cuda.is_available():
        torch.cuda.init()
        print(f"CUDA initialized. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Using CPU.")
    
    print(f"Processing main directory: {main_dir}")
    train_features, test_features, test_file_names = process_main_directory(main_dir)
    
    print(f"Total training samples: {len(train_features)} (randomly selected)")
    print(f"Total test samples: {len(test_features)} (50 randomly selected)")
    
    if len(train_features) == 0 or len(test_features) == 0:
        print("Error: No valid training or test data found")
        return None
    
    (train_sequences, y_train,
     test_sequences, y_test,
     y_scaler, seq_input_size) = prepare_data_for_model(train_features, test_features)
    
    train_loader, test_loader = create_data_loaders(
        train_sequences, y_train,
        test_sequences, y_test,
        batch_size=32  # Increased batch size
    )
    
    global model
    model = EnhancedRecursiveLSTMModel(
        seq_input_size=seq_input_size,
        hidden_sizes=[256, 128, 64],  # Reduced hidden sizes
        output_size=1,
        dropout_rate=0.3,  # Increased dropout
        num_heads=4  # Reduced num_heads
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=5e-3)  # Increased weight decay
    
    print("Building and training Enhanced Recursive LSTM model...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader,
        focal_loss, optimizer,
        num_epochs=500, patience=30, accumulation_steps=2, warmup_epochs=20  # Adjusted parameters
    )
    
    if train_losses is None or val_losses is None:
        print("Training failed due to invalid values")
        return None
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(
        model, test_sequences, y_test,
        y_scaler, test_file_names
    )
    
    print(f"\nSummary for Comparison:")
    print(f"Model: EnhancedRecursiveLSTM")
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "synthetic_data"
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    model, y_scaler, y_test_actual, y_pred_actual = main(main_dir)
