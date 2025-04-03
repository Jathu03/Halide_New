import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import joblib

# Placeholder for data processing (same as above)
def prepare_data_for_model(train_features, test_features):
    input_size = 10  # Adjust as needed
    X_train = torch.randn(100, 1, input_size)
    y_train = torch.randn(100, 1)
    X_test = torch.randn(50, 1, input_size)
    y_test = torch.randn(50, 1)
    y_scaler = MinMaxScaler().fit(y_train.numpy())
    return X_train, y_train, X_test, y_test, y_scaler, input_size, False

class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=1, dropout_rate=0.3):
        super(EnhancedLSTMModel, self).__init__()
        self.hidden_sizes = hidden_sizes
        # Same architecture as above...
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.lstm_layers.append(nn.LSTM(input_size, hidden_sizes[0], batch_first=True))
        self.dropout_layers.append(nn.Dropout(dropout_rate))
        for i in range(1, len(hidden_sizes)):
            self.lstm_layers.append(nn.LSTM(hidden_sizes[i-1], hidden_sizes[i], batch_first=True))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        self.attention = nn.Linear(hidden_sizes[-1], 1)
        self.fc_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2))
        self.bn_layers.append(nn.BatchNorm1d(hidden_sizes[-1] // 2))
        self.fc_layers.append(nn.Linear(hidden_sizes[-1] // 2, hidden_sizes[-1] // 4))
        self.bn_layers.append(nn.BatchNorm1d(hidden_sizes[-1] // 4))
        self.output_layer = nn.Linear(hidden_sizes[-1] // 4, output_size)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.has_residual = (hidden_sizes[-1] // 4 == hidden_sizes[-1] // 2)
        if not self.has_residual:
            self.residual_adapter = nn.Linear(hidden_sizes[-1] // 2, hidden_sizes[-1] // 4)
        
    def attention_net(self, lstm_output):
        attn_weights = self.attention(lstm_output).squeeze(2)
        soft_attn_weights = torch.softmax(attn_weights, 1)
        context = torch.bmm(soft_attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context
        
    def forward(self, x):
        batch_size = x.size(0)
        lstm_out = x
        device = torch.device('cuda')  # Explicitly set to CUDA
        
        for i, (lstm, dropout) in enumerate(zip(self.lstm_layers, self.dropout_layers)):
            hidden_size = self.hidden_sizes[i]
            h_0 = torch.zeros(1, batch_size, hidden_size, device=device)
            c_0 = torch.zeros(1, batch_size, hidden_size, device=device)
            lstm_out, _ = lstm(lstm_out, (h_0, c_0))
            if i < len(self.lstm_layers) - 1:
                lstm_out = dropout(lstm_out)
        
        attn_output = self.attention_net(lstm_out)
        fc_out = self.fc_layers[0](attn_output)
        fc_out = self.bn_layers[0](fc_out)
        fc_out = self.leaky_relu(fc_out)
        residual = fc_out
        if not self.has_residual:
            residual = self.residual_adapter(residual)
        fc_out = self.fc_layers[1](fc_out)
        fc_out = self.bn_layers[1](fc_out)
        fc_out = self.leaky_relu(fc_out)
        fc_out = fc_out + residual
        output = self.output_layer(fc_out)
        return output

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, patience=20):
    device = torch.device('cuda')  # Force CUDA
    model.to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
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
    return model

def main():
    train_features, test_features = [], []  # Placeholder
    X_train, y_train, X_test, y_test, y_scaler, input_size, is_log_transformed = prepare_data_for_model(train_features, test_features)
    
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16)
    
    model = EnhancedLSTMModel(input_size=input_size, hidden_sizes=[128, 64, 32], output_size=1, dropout_rate=0.3)
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Train model on CUDA
    print("Training on CUDA...")
    model = train_model(model, train_loader, test_loader, criterion, optimizer)
    
    # Save model using TorchScript on CUDA
    model.eval()
    model.to('cuda')
    sample_input = torch.randn(1, 1, input_size, device='cuda')
    
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Sample input device: {sample_input.device}")
    
    traced_model = torch.jit.trace(model, sample_input)
    traced_model.save("lstm_model.pt")
    print("Model saved as 'lstm_model.pt' on CUDA")
    
    joblib.dump(y_scaler, "y_scaler.pkl")
    print("Scaler saved as 'y_scaler.pkl'")

if __name__ == "__main__":
    random.seed(42)
    main()
