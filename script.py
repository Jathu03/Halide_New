import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class HalideDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).view(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_preprocess_data(file_path="halide_execution_dataset.csv"):
    try:
        df = pd.read_csv(file_path)
        
        if 'execution_time_ms' not in df.columns:
            raise KeyError("execution_time_ms column not found in CSV")
        
        # Remove rows with NaN or extreme values
        initial_rows = len(df)
        df = df.dropna()
        df = df[df['execution_time_ms'] > 0]  # Remove non-positive execution times
        if len(df) < initial_rows:
            print(f"Dropped {initial_rows - len(df)} rows due to missing or invalid data")
        
        if len(df) == 0:
            raise ValueError("No valid data remaining after preprocessing")
            
        target = np.log1p(df['execution_time_ms'].values)  # Log transform target
        features = df.drop(columns=['execution_time_ms', 'file_path'])
        
        # Encode categorical variables
        label_encoders = {}
        for column in ['program_name', 'schedule_name']:
            le = LabelEncoder()
            features[column] = le.fit_transform(features[column])
            label_encoders[column] = le
        
        # Scale numerical features
        numerical_features = features.drop(columns=['program_name', 'schedule_name'])
        scaler = StandardScaler()
        scaled_numerical = scaler.fit_transform(numerical_features)
        
        # Combine features
        X = np.hstack([
            features[['program_name', 'schedule_name']].values,
            scaled_numerical
        ])
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target shape: {target.shape}")
        return X, target, label_encoders, scaler, X.shape[1], numerical_features.columns
    
    except Exception as e:
        raise Exception(f"Error in data preprocessing: {str(e)}")

class LSTMModel(nn.Module):
    def __init__(self, input_size, vocab_size=100, embedding_dim=16, hidden_size=128, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        cat_part = x[:, :2].long()
        num_part = x[:, 2:].float()
        embedded = self.embedding(cat_part)
        num_part = num_part.unsqueeze(1).expand(-1, embedded.shape[1], -1)
        x = torch.cat([embedded, num_part], dim=2)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def calculate_error_percentage(y_true, y_pred):
    y_true = np.expm1(y_true)  # Reverse log transform
    y_pred = np.expm1(y_pred)
    return np.abs((y_true - y_pred) / y_true) * 100

def train_model(model, train_loader, val_loader, epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_mae = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(outputs - y_batch)).item()
        
        model.eval()
        val_loss = 0
        val_mae = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
                val_mae += torch.mean(torch.abs(outputs - y_batch)).item()
        
        train_loss /= len(train_loader)
        train_mae /= len(train_loader)
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}")
        print(f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

def main():
    try:
        X, y, label_encoders, scaler, input_size, _ = load_and_preprocess_data()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        train_dataset = HalideDataset(X_train, y_train)
        val_dataset = HalideDataset(X_val, y_val)
        test_dataset = HalideDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        test_loader = DataLoader(test_dataset, batch_size=64)
        
        vocab_size = max(
            len(label_encoders['program_name'].classes_),
            len(label_encoders['schedule_name'].classes_)
        ) + 1
        input_size = 16 + (input_size - 2)  # embedding_dim + numerical features
        model = LSTMModel(input_size, vocab_size)
        
        train_model(model, train_loader, val_loader)
        
        # Load best model
        model.load_state_dict(torch.load('best_model.pt'))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        test_mae = 0
        y_pred = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                test_mae += torch.mean(torch.abs(outputs - y_batch)).item()
                y_pred.extend(outputs.cpu().numpy().flatten())
        
        y_pred = np.array(y_pred)
        test_mae = test_mae / len(test_loader)
        print(f"\nTest MAE (log scale): {test_mae:.4f}")
        
        # Convert back to original scale for error percentage
        sample_indices = np.random.choice(len(y_test), min(10, len(y_test)), replace=False)
        sample_errors = []
        
        print("\nError percentages for test samples:")
        for idx in sample_indices:
            true_time = y_test[idx]
            pred_time = y_pred[idx]
            error_pct = calculate_error_percentage(true_time, pred_time)
            sample_errors.append(error_pct)
            print(f"Sample {idx}:")
            print(f"  True execution time: {np.expm1(true_time):.2f} ms")
            print(f"  Predicted: {np.expm1(pred_time):.2f} ms")
            print(f"  Error percentage: {error_pct:.2f}%")
        
        print(f"\nMean error percentage: {np.mean(sample_errors):.2f}%")
        print(f"Median error percentage: {np.median(sample_errors):.2f}%")
        
        torch.save(model.state_dict(), 'halide_execution_time_model.pt')
        print("Final model saved to 'halide_execution_time_model.pt'")
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")

if __name__ == "__main__":
    main()
