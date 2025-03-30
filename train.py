import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

# --- (Keep all helper functions: get_execution_time, extract_features_from_file, process_directory, process_main_directory, clean_and_transform_features, prepare_data_for_model) ---

class StackedLSTM(nn.Module):
    """Deep LSTM with multiple layers"""
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=1):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        self.fc = nn.Linear(hidden_sizes[-1], output_size)
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x[:, -1, :]  # Take last timestep
        return self.fc(x)

class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM with attention"""
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(2 * hidden_size, 1)  # 2x for bidirectional
        self.fc = nn.Linear(2 * hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, 2*hidden_size]
        
        # Attention mechanism
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        return self.fc(context)

class LSTMAttention(nn.Module):
    """LSTM with self-attention"""
    def __init__(self, input_size, hidden_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        return self.fc(context)

def train_and_evaluate_model(model, train_loader, test_loader, model_name="LSTM"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, patience=5)
    
    best_val_loss = float('inf')
    for epoch in range(100):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch.to(device))
            loss = criterion(outputs, y_batch.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch.to(device))
                val_loss += criterion(outputs, y_batch.to(device)).item()
        
        scheduler.step(val_loss)
        print(f"{model_name} Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss/len(test_loader):.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_{model_name.lower()}.pth")
    
    return model

def main():
    # --- Data Preparation (Same as before) ---
    train_features, test_features, test_file_names = process_main_directory("synthetic_data")
    X_train, y_train, X_test, y_test, y_scaler, input_size, _ = prepare_data_for_model(train_features, test_features)
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test)
    
    # --- Train Multiple LSTM Variants ---
    models = {
        "StackedLSTM": StackedLSTM(input_size),
        "BidirectionalLSTM": BidirectionalLSTM(input_size),
        "LSTMAttention": LSTMAttention(input_size)
    }
    
    best_model = None
    best_loss = float('inf')
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        trained_model = train_and_evaluate_model(model, train_loader, test_loader, name)
        val_loss = evaluate_model(trained_model, X_test, y_test, y_scaler, test_file_names)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = trained_model
    
    # --- Export Best Model to TorchScript ---
    best_model.eval()
    example_input = torch.randn(1, 1, input_size)  # [batch=1, seq_len=1, features]
    traced_model = torch.jit.trace(best_model, example_input)
    traced_model.save("halide_lstm_best.pt")
    print("Saved best model to 'halide_lstm_best.pt'")

if __name__ == "__main__":
    main()
