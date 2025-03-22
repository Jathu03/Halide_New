import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Dataset class
class TiramisuDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            "comps": self.data[idx]["comps_tensor"],
            "loops": self.data[idx]["loops_tensor"],
            "expr": self.data[idx]["expr_tensor"]
        }, self.data[idx]["exec_time"]

# LSTM Model
class TiramisuLSTM(nn.Module):
    def __init__(self, comp_input_dim, loop_input_dim, expr_input_dim, hidden_size=256, num_layers=2):
        super(TiramisuLSTM, self).__init__()
        self.comp_lstm = nn.LSTM(comp_input_dim, hidden_size, num_layers, batch_first=True)
        self.loop_lstm = nn.LSTM(loop_input_dim, hidden_size, num_layers, batch_first=True)
        self.expr_lstm = nn.LSTM(expr_input_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        comp_out, _ = self.comp_lstm(x["comps"])  # (batch, seq, hidden)
        loop_out, _ = self.loop_lstm(x["loops"])
        expr_out, _ = self.expr_lstm(x["expr"])
        
        # Take the last output of each LSTM
        comp_out = comp_out[:, -1, :]
        loop_out = loop_out[:, -1, :]
        expr_out = expr_out[:, -1, :]
        
        combined = torch.cat([comp_out, loop_out, expr_out], dim=1)
        return self.fc(combined)

# Training function
def train_model():
    # Load dataset
    dataset = torch.load("tiramisu_dataset.pt")
    
    # Normalize execution times
    exec_times = np.array([d["exec_time"] for d in dataset]).reshape(-1, 1)
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(exec_times).flatten()
    for i, d in enumerate(dataset):
        d["exec_time"] = y_scaled[i]
    
    # Split into train, val, test (70%, 15%, 15%)
    train_val, test = train_test_split(dataset, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1765, random_state=42)  # 0.1765 of 85% = 15% of total
    
    train_dataset = TiramisuDataset(train)
    val_dataset = TiramisuDataset(val)
    test_dataset = TiramisuDataset(test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model setup
    comp_input_dim = dataset[0]["comps_tensor"].shape[1]  # e.g., 263 from the document
    loop_input_dim = dataset[0]["loops_tensor"].shape[1]  # e.g., 8
    expr_input_dim = dataset[0]["expr_tensor"].shape[2]   # 11 (8 expr + 3 type)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TiramisuLSTM(comp_input_dim, loop_input_dim, expr_input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    
    # Training loop
    num_epochs = 50
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_y = batch_y.to(device).view(-1, 1)
            for k in batch_x:
                batch_x[k] = batch_x[k].to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x["comps"].size(0)
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_y = batch_y.to(device).view(-1, 1)
                for k in batch_x:
                    batch_x[k] = batch_x[k].to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item() * batch_x["comps"].size(0)
        val_loss /= len(val_loader.dataset)
        
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "tiramisu_lstm_best.pth")
    
    # Test evaluation
    model.load_state_dict(torch.load("tiramisu_lstm_best.pth"))
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_y = batch_y.to(device).view(-1, 1)
            for k in batch_x:
                batch_x[k] = batch_x[k].to(device)
            outputs = model(batch_x)
            test_preds.extend(outputs.cpu().numpy().flatten())
            test_true.extend(batch_y.cpu().numpy().flatten())
    
    # Denormalize
    test_true_denorm = scaler.inverse_transform(np.array(test_true).reshape(-1, 1)).flatten()
    test_preds_denorm = scaler.inverse_transform(np.array(test_preds).reshape(-1, 1)).flatten()
    
    # Calculate MAPE
    mape = np.mean(np.abs((test_true_denorm - test_preds_denorm) / test_true_denorm)) * 100
    print(f"\nFinal Test MAPE: {mape:.2f}%")
    print("Sample True vs Predicted (denormalized):")
    for i in range(min(5, len(test_true_denorm))):
        print(f"True: {test_true_denorm[i]:.4f}, Predicted: {test_preds_denorm[i]:.4f}")
    
    torch.save({"model": model.state_dict(), "scaler": scaler}, "tiramisu_lstm_final.pt")

if __name__ == "__main__":
    train_model()
