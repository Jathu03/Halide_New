import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Custom Dataset class
class HalideDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).view(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Load and preprocess the dataset
def load_and_preprocess_data(file_path="halide_execution_dataset.csv"):
    df = pd.read_csv(file_path)
    
    target = df['execution_time_ms']
    features = df.drop(columns=['execution_time_ms', 'file_path'])
    
    # Encode categorical variables
    label_encoders = {}
    for column in ['program_name', 'schedule_name']:
        le = LabelEncoder()
        features[column] = le.fit_transform(features[column])
        label_encoders[column] = le
    
    # Numerical features
    numerical_features = features.drop(columns=['program_name', 'schedule_name'])
    
    # Scale numerical features
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(numerical_features)
    
    # Create sequences
    sequences = []
    for i in range(len(features)):
        seq = [
            features['program_name'].iloc[i],
            features['schedule_name'].iloc[i]
        ] + scaled_numerical[i].tolist()
        sequences.append(seq)
    
    X = np.array(sequences, dtype=np.float32)
    return X, target.values, label_encoders, scaler, len(sequences[0]), numerical_features.columns

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, vocab_size=100, embedding_dim=32, hidden_size1=64, hidden_size2=32):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size2, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Split input into categorical and numerical parts
        cat_part = x[:, :2].long()  # program_name and schedule_name
        num_part = x[:, 2:].float()
        
        # Embedding for categorical features
        embedded = self.embedding(cat_part)  # [batch, 2, embedding_dim]
        
        # Combine with numerical features
        num_part = num_part.unsqueeze(1).expand(-1, 2, -1)  # [batch, 2, num_features]
        x = torch.cat([embedded, num_part], dim=2)  # [batch, 2, embedding_dim + num_features]
        
        # LSTM layers
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]  # Take last output
        
        # Dense layers
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# Calculate error percentage
def calculate_error_percentage(y_true, y_pred):
    return np.abs((y_true - y_pred) / y_true) * 100

def train_model(model, train_loader, val_loader, epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
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
            optimizer.step()
            
            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(outputs - y_batch)).item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_mae = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
                val_mae += torch.mean(torch.abs(outputs - y_batch)).item()
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, MAE: {train_mae/len(train_loader):.2f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, MAE: {val_mae/len(val_loader):.2f}")

def main():
    try:
        # Load and preprocess data
        X, y, label_encoders, scaler, input_size, _ = load_and_preprocess_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create validation split from training data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Create data loaders
        train_dataset = HalideDataset(X_train, y_train)
        val_dataset = HalideDataset(X_val, y_val)
        test_dataset = HalideDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # Initialize model
        vocab_size = max(
            len(label_encoders['program_name'].classes_),
            len(label_encoders['schedule_name'].classes_)
        ) + 1
        input_size = 32 + (input_size - 2)  # embedding_dim + numerical features
        model = LSTMModel(input_size, vocab_size)
        
        # Train model
        train_model(model, train_loader, val_loader)
        
        # Evaluate on test set
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
        
        print(f"\nTest MAE: {test_mae/len(test_loader):.2f} ms")
        
        # Calculate error percentages for 10 random test samples
        y_pred = np.array(y_pred)
        indices = np.random.choice(len(y_test), 10, replace=False)
        sample_errors = []
        
        print("\nError percentages for 10 random test samples:")
        for idx in indices:
            true_time = y_test[idx]
            pred_time = y_pred[idx]
            error_pct = calculate_error_percentage(true_time, pred_time)
            sample_errors.append(error_pct)
            print(f"Sample {idx}:")
            print(f"  True execution time: {true_time:.2f} ms")
            print(f"  Predicted: {pred_time:.2f} ms")
            print(f"  Error percentage: {error_pct:.2f}%")
        
        # Overall statistics
        mean_error_pct = np.mean(sample_errors)
        median_error_pct = np.median(sample_errors)
        print(f"\nMean error percentage: {mean_error_pct:.2f}%")
        print(f"Median error percentage: {median_error_pct:.2f}%")
        
        # Save model
        torch.save(model.state_dict(), 'halide_execution_time_model.pt')
        print("Model saved to 'halide_execution_time_model.pt'")
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")

if __name__ == "__main__":
    main()
