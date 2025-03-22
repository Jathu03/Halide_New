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

# Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

# Transformer-based Model
class TiramisuTransformer(nn.Module):
    def __init__(self, comp_input_dim, loop_input_dim, expr_input_dim, d_model=256, n_heads=8, n_layers=4):
        super(TiramisuTransformer, self).__init__()
        self.d_model = d_model
        
        # Input projections to d_model
        self.expr_proj = nn.Linear(expr_input_dim, d_model)
        self.comp_proj = nn.Linear(comp_input_dim, d_model)
        self.loop_proj = nn.Linear(loop_input_dim, d_model)
        
        # Positional encodings
        self.expr_pos_enc = PositionalEncoding(d_model)
        self.comp_pos_enc = PositionalEncoding(d_model)
        self.loop_pos_enc = PositionalEncoding(d_model)
        
        # Transformer encoders
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.expr_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.comp_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.loop_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Attention for aggregation
        self.global_attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.ReLU()  # Ensure non-negative predictions
        )
    
    def forward(self, x):
        batch_size = x["comps"].size(0)
        
        # Step 1: Process expressions
        num_comps, expr_len = x["expr"].size(1), x["expr"].size(2)
        expr_input = x["expr"].view(batch_size * num_comps, expr_len, -1)  # [batch*num_comps, expr_len, expr_input_dim]
        expr_embed = self.expr_proj(expr_input)  # [batch*num_comps, expr_len, d_model]
        expr_embed = self.expr_pos_enc(expr_embed)
        expr_out = self.expr_transformer(expr_embed)  # [batch*num_comps, expr_len, d_model]
        expr_embed = expr_out.mean(dim=1).view(batch_size, num_comps, -1)  # [batch, num_comps, d_model]
        
        # Step 2: Process computations with expression embeddings
        comp_input = self.comp_proj(x["comps"])  # [batch, max_comps, d_model]
        comp_input = comp_input + expr_embed  # Add expression embeddings to comp features
        comp_input = self.comp_pos_enc(comp_input)
        comp_out = self.comp_transformer(comp_input)  # [batch, max_comps, d_model]
        comp_embed = comp_out.mean(dim=1, keepdim=True)  # [batch, 1, d_model]
        
        # Step 3: Process loops
        loop_input = self.loop_proj(x["loops"])  # [batch, max_loops, d_model]
        loop_input = self.loop_pos_enc(loop_input)
        loop_out = self.loop_transformer(loop_input)  # [batch, max_loops, d_model]
        loop_embed = loop_out.mean(dim=1, keepdim=True)  # [batch, 1, d_model]
        
        # Step 4: Global attention to combine comp and loop embeddings
        combined_embed = torch.cat([comp_embed, loop_embed], dim=1)  # [batch, 2, d_model]
        attn_output, _ = self.global_attention(combined_embed, combined_embed, combined_embed)  # [batch, 2, d_model]
        global_embed = attn_output.mean(dim=1)  # [batch, d_model]
        
        # Step 5: Final prediction
        return self.fc(global_embed)

# Training function
def train_model():
    # Load dataset
    dataset = torch.load("tiramisu_dataset.pt")
    
    # Normalize execution times (log transform to handle wide range)
    exec_times = np.array([d["exec_time"] for d in dataset])
    exec_times = np.log1p(exec_times)
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
    model = TiramisuTransformer(comp_input_dim, loop_input_dim, expr_input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)  # AdamW for better regularization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=15)
    
    # Training loop
    num_epochs = 200  # More epochs for Transformer convergence
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
            torch.save(model.state_dict(), "tiramisu_transformer_best.pth")
    
    # Test evaluation
    model.load_state_dict(torch.load("tiramisu_transformer_best.pth"))
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
    
    torch.save({"model": model.state_dict(), "scaler": scaler}, "tiramisu_transformer_final.pt")

if __name__ == "__main__":
    train_model()
