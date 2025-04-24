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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random
from collections import defaultdict
import matplotlib.pyplot as plt

# [Previous code for FIXED_FEATURES, extract_features, process_tree_output_directory, prepare_data_for_model, MultiHeadAttention, EnhancedRecursiveLSTMModel, custom_loss, create_data_loaders remains unchanged]

# Modified train_model function with checkpoint saving
def train_model(model, train_loader, test_loader, criterion, optimizer, feature_indices, feature_importances, num_epochs=700, patience=50, accumulation_steps=2, checkpoint_path='recursive.pth'):
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
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    start_epoch = 0
    
    # Check if a checkpoint exists to resume training
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        epochs_no_improve = checkpoint['epochs_no_improve']
        best_model_state = checkpoint['best_model_state']
        print(f"Resuming training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for i, (seq_inputs, scalar_inputs, targets) in enumerate(train_loader):
            seq_inputs, scalar_inputs, targets = seq_inputs.to(device), scalar_inputs.to(device), targets.to(device)
            outputs = model(seq_inputs, scalar_inputs)
            loss = criterion(outputs, targets, scalar_inputs, feature_indices, feature_importances)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss detected at epoch {epoch+1}, batch {i+1}")
                return None, None
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            for seq_inputs, scalar_inputs, targets in test_loader:
                seq_inputs, scalar_inputs, targets = seq_inputs.to(device), scalar_inputs.to(device), targets.to(device)
                outputs = model(seq_inputs, scalar_inputs)
                loss = criterion(outputs, targets, scalar_inputs, feature_indices, feature_importances)
                val_loss += loss.item() * seq_inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epochs_no_improve': epochs_no_improve,
            'best_model_state': best_model_state
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")
        
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
    plt.close()
    
    return train_losses, val_losses

# New function to resume training explicitly
def resume_training(model, train_loader, test_loader, criterion, optimizer, feature_indices, feature_importances, num_epochs=700, patience=50, accumulation_steps=2, checkpoint_path='recursive.pth'):
    print(f"Attempting to resume training from checkpoint: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}. Starting training from scratch.")
        return train_model(model, train_loader, test_loader, criterion, optimizer, feature_indices, feature_importances, num_epochs, patience, accumulation_steps, checkpoint_path)
    
    return train_model(model, train_loader, test_loader, criterion, optimizer, feature_indices, feature_importances, num_epochs, patience, accumulation_steps, checkpoint_path)

# Modified main function to use resume_training
def main(main_dir):
    if torch.cuda.is_available():
        torch.cuda.init()
        print(f"CUDA initialized. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA not available. Using CPU.")
    
    print(f"Processing main directory: {main_dir}")
    train_features, test_features, test_file_names = process_tree_output_directory(main_dir)
    
    if len(train_features) == 0 or len(test_features) == 0:
        print("Error: No valid training or test data found")
        return None
    
    (train_sequences, train_scalar, y_train,
     test_sequences, test_scalar, y_test,
     y_scaler, seq_input_size, scalar_input_size, feature_columns) = prepare_data_for_model(train_features, test_features)
    
    train_loader, test_loader = create_data_loaders(
        train_sequences, train_scalar, y_train,
        test_sequences, test_scalar, y_test,
        batch_size=64
    )
    
    global model
    model = EnhancedRecursiveLSTMModel(
        seq_input_size=seq_input_size,
        scalar_input_size=scalar_input_size,
        hidden_sizes=[512, 256, 128],
        output_size=1,
        dropout_rate=0.2,
        num_heads=8
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-4)
    
    feature_importances = {
        'cache_hits': 0.5860,
        'bytes_processing_rate': 0.2893,
        'sched_bytes_at_task': 0.0422,
        'sched_working_set_at_root': 0.0248,
        'sched_bytes_at_realization': 0.0055,
        'sched_unique_bytes_read_per_realization': 0.0049
    }
    
    feature_indices = {}
    for feature in feature_importances.keys():
        log_feature = f'log_{feature}' if feature in ['cache_hits', 'bytes_processing_rate'] else feature
        if log_feature in feature_columns:
            feature_indices[feature] = feature_columns.get_loc(log_feature)
        else:
            feature_indices[feature] = feature_columns.get_loc(feature) if feature in feature_columns else -1
    
    print("Building and training Enhanced Recursive LSTM model...")
    train_losses, val_losses = resume_training(
        model, train_loader, test_loader,
        custom_loss, optimizer, feature_indices, feature_importances,
        num_epochs=700, patience=50, accumulation_steps=2, checkpoint_path='recursive.pth'
    )
    
    if train_losses is None or val_losses is None:
        print("Training failed due to invalid values")
        return None
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(
        model, test_sequences, test_scalar, y_test,
        y_scaler, test_file_names
    )
    
    print(f"\nSummary for Comparison:")
    print(f"Model: EnhancedRecursiveLSTM")
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "Tree_Output"
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    model, y_scaler, y_test_actual, y_pred_actual = main(main_dir)
