import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Custom collate function to handle variable-sized tensors
def custom_collate_fn(batch):
    comps_list = [item[0]["comps"] for item in batch]
    loops_list = [item[0]["loops"] for item in batch]
    expr_list = [item[0]["expr"] for item in batch]
    exec_times = torch.tensor([item[1] for item in batch], dtype=torch.float32)

    max_comps = max(c.shape[0] for c in comps_list)
    max_loops = max(l.shape[0] for l in loops_list)
    comp_feature_size = comps_list[0].shape[1]
    loop_feature_size = loops_list[0].shape[1]
    expr_feature_size = expr_list[0].shape[2]
    max_expr_len = expr_list[0].shape[1]

    padded_comps = torch.zeros(len(batch), max_comps, comp_feature_size)
    padded_loops = torch.zeros(len(batch), max_loops, loop_feature_size)
    padded_expr = torch.zeros(len(batch), max_comps, max_expr_len, expr_feature_size)

    for i in range(len(batch)):
        comps = comps_list[i]
        loops = loops_list[i]
        expr = expr_list[i]
        padded_comps[i, :comps.shape[0], :] = comps
        padded_loops[i, :loops.shape[0], :] = loops
        padded_expr[i, :expr.shape[0], :, :] = expr

    return {
        "comps": padded_comps,
        "loops": padded_loops,
        "expr": padded_expr
    }, exec_times

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

# Multi-Head Attention module
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.fc_out(context)  # Return full sequence for further processing

# Hierarchical LSTM Model
class TiramisuHierarchicalLSTM(nn.Module):
    def __init__(self, comp_input_dim, loop_input_dim, expr_input_dim, hidden_size=256, num_layers=2):
        super(TiramisuHierarchicalLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        # Embeddings for computations, loops, and expressions
        self.comp_lstm = nn.LSTM(comp_input_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.loop_lstm = nn.LSTM(loop_input_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.expr_lstm = nn.LSTM(expr_input_dim, hidden_size, num_layers, batch_first=True)
        
        # Attention layers
        self.comp_attention = MultiHeadAttention(hidden_size * 2)
        self.loop_attention = MultiHeadAttention(hidden_size * 2)
        self.expr_attention = MultiHeadAttention(hidden_size)
        
        # Project expr_seq to match comp_seq size
        self.expr_proj = nn.Linear(hidden_size, hidden_size * 2)
        
        # Final aggregation LSTM to combine components hierarchically
        self.agg_lstm = nn.LSTM(hidden_size * 2, hidden_size, num_layers=1, batch_first=True)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.ReLU()  # Ensure non-negative predictions
        )
    
    def forward(self, x):
        batch_size = x["comps"].size(0)
        
        # Comps embedding
        comp_out, _ = self.comp_lstm(x["comps"])  # [batch, max_comps, hidden*2]
        comp_seq = self.comp_attention(comp_out)  # [batch, max_comps, hidden*2]
        
        # Loops embedding
        loop_out, _ = self.loop_lstm(x["loops"])  # [batch, max_loops, hidden*2]
        loop_seq = self.loop_attention(loop_out)  # [batch, max_loops, hidden*2]
        
        # Expr embedding (process per computation)
        num_comps, expr_len = x["expr"].size(1), x["expr"].size(2)
        expr_input = x["expr"].view(batch_size * num_comps, expr_len, -1)  # [batch*num_comps, expr_len, input_size]
        expr_out, _ = self.expr_lstm(expr_input)  # [batch*num_comps, expr_len, hidden]
        expr_out = expr_out.view(batch_size, num_comps, expr_len, -1)  # [batch, num_comps, expr_len, hidden]
        expr_seq = self.expr_attention(expr_out.reshape(batch_size, num_comps * expr_len, -1))  # [batch, num_comps*expr_len, hidden]
        expr_seq = expr_seq.view(batch_size, num_comps, expr_len, -1).mean(dim=2)  # [batch, num_comps, hidden]
        expr_seq = self.expr_proj(expr_seq)  # [batch, num_comps, hidden*2]
        
        # Combine comps and expr hierarchically
        comp_expr_seq = torch.cat([comp_seq, expr_seq], dim=2)  # [batch, num_comps, hidden*4]
        comp_expr_seq = comp_expr_seq.mean(dim=1, keepdim=True)  # [batch, 1, hidden*4]
        comp_expr_seq = nn.Linear(hidden_size * 4, hidden_size * 2)(comp_expr_seq)  # [batch, 1, hidden*2]
        
        # Pad loop sequence to match dimensions for aggregation
        loop_seq = loop_seq.mean(dim=1, keepdim=True)  # [batch, 1, hidden*2]
        
        # Aggregate hierarchically with LSTM
        agg_input = torch.cat([comp_expr_seq, loop_seq], dim=1)  # [batch, 2, hidden*2]
        agg_out, (agg_h, _) = self.agg_lstm(agg_input)  # [batch, 2, hidden]
        agg_context = agg_h[-1]  # [batch, hidden]
        
        # Final prediction
        return self.fc(agg_context)

# Training function
def train_model():
    # Load dataset
    dataset = torch.load("tiramisu_dataset.pt")
    
    # Normalize execution times (log transform to handle wide range)
    exec_times = np.array([d["exec_time"] for d in dataset])
    exec_times = np.log1p(exec_times)  # log(1+x) to handle small values
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(exec_times.reshape(-1, 1)).flatten()
    for i, d in enumerate(dataset):
        d["exec_time"] = y_scaled[i]
    
    # Split into train, val, test (70%, 15%, 15%)
    train_val, test = train_test_split(dataset, test_size=0.15, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1765, random_state=42)
    
    train_dataset = TiramisuDataset(train)
    val_dataset = TiramisuDataset(val)
    test_dataset = TiramisuDataset(test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate_fn)
    
    # Model setup
    comp_input_dim = dataset[0]["comps_tensor"].shape[1]  # e.g., 704
    loop_input_dim = dataset[0]["loops_tensor"].shape[1]  # e.g., 8
    expr_input_dim = dataset[0]["expr_tensor"].shape[2]   # 11
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TiramisuHierarchicalLSTM(comp_input_dim, loop_input_dim, expr_input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    
    # Denormalize (reverse log transform)
    test_true_denorm = np.expm1(scaler.inverse_transform(np.array(test_true).reshape(-1, 1)).flatten())
    test_preds_denorm = np.expm1(scaler.inverse_transform(np.array(test_preds).reshape(-1, 1)).flatten())
    
    # Calculate MAPE
    mape = np.mean(np.abs((test_true_denorm - test_preds_denorm) / test_true_denorm)) * 100
    print(f"\nFinal Test MAPE: {mape:.2f}%")
    print("Sample True vs Predicted (denormalized):")
    for i in range(min(5, len(test_true_denorm))):
        print(f"True: {test_true_denorm[i]:.4f}, Predicted: {test_preds_denorm[i]:.4f}")
    
    torch.save({"model": model.state_dict(), "scaler": scaler}, "tiramisu_lstm_final.pt")

if __name__ == "__main__":
    train_model()
