import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import random

# ---- Fixed feature keys (should match your dataset) ----
FIXED_FEATURES = [
    # Add all features you want to extract from each node
    'cache_hits', 'cache_misses', 'execution_time_ms',
    'sched_num_realizations', 'sched_num_productions',
    'sched_points_computed_total', 'sched_innermost_loop_extent',
    'sched_inner_parallelism', 'sched_outer_parallelism',
    'sched_bytes_at_realization', 'sched_bytes_at_production',
    'sched_bytes_at_root', 'sched_unique_bytes_read_per_realization',
    'sched_working_set', 'sched_vector_size', 'sched_num_vectors',
    'sched_num_scalars', 'sched_bytes_at_task', 'sched_working_set_at_task',
    'sched_working_set_at_production', 'sched_working_set_at_realization',
    'sched_working_set_at_root', 'total_parallelism', 'scheduling_count',
    'total_bytes_at_production', 'total_vectors', 'computation_efficiency',
    'memory_pressure', 'memory_utilization_ratio', 'bytes_processing_rate',
    'bytes_per_parallelism', 'bytes_per_vector', 'nodes_count', 'edges_count',
    'node_edge_ratio', 'nodes_per_schedule', 'op_diversity',
    # Example operation features (add more as needed)
    'op_add', 'op_mul', 'op_sub', 'op_div'
]

# ---- Tree feature extraction ----
def extract_tree_features(node):
    # Extract features for this node
    features = {k: node.get(k, 0.0) for k in FIXED_FEATURES}
    # Recursively extract children
    children = node.get('children', [])
    child_features = [extract_tree_features(child) for child in children]
    return {'features': features, 'children': child_features}

# ---- Dataset for tree-structured data ----
class HalideTreeDataset(Dataset):
    def __init__(self, file_list):
        self.trees = []
        self.targets = []
        for file_path in file_list:
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Root node must have valid execution time
            if 'children' not in data:
                continue
            root = extract_tree_features(data)
            exec_time = root['features'].get('execution_time_ms', 0)
            if exec_time > 0 and np.isfinite(exec_time):
                self.trees.append(root)
                self.targets.append(exec_time)
        print(f"Loaded {len(self.trees)} valid trees.")

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, idx):
        return self.trees[idx], self.targets[idx]

# ---- Recursive TreeLSTM ----
class TreeLSTMCell(nn.Module):
    def __init__(self, feature_size, hidden_size):
        super().__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.W_iou = nn.Linear(feature_size, 3 * hidden_size)
        self.U_iou = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.W_f = nn.Linear(feature_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, features, child_states):
        # child_states: list of (h, c) tuples
        if child_states:
            h_sum = torch.sum(torch.stack([h for h, c in child_states]), dim=0, keepdim=True)
        else:
            h_sum = torch.zeros(1, self.hidden_size, device=features.device)
        iou = self.W_iou(features) + self.U_iou(h_sum)
        i, o, u = torch.chunk(iou, 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        if child_states:
            f = torch.stack([torch.sigmoid(self.W_f(features) + self.U_f(h)) for h, c in child_states])
            fc = torch.sum(f * torch.stack([c for h, c in child_states]), dim=0)
        else:
            fc = torch.zeros(1, self.hidden_size, device=features.device)
        c = i * u + fc
        h = o * torch.tanh(c)
        return h, c

class RecursiveTreeLSTM(nn.Module):
    def __init__(self, feature_size, hidden_size=128):
        super().__init__()
        self.cell = TreeLSTMCell(feature_size, hidden_size)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, tree):
        h, c = self._recurse(tree)
        return self.regressor(h)

    def _recurse(self, node):
        device = next(self.parameters()).device
        features = torch.tensor([list(node['features'].values())], dtype=torch.float32, device=device)
        child_states = [self._recurse(child) for child in node['children']]
        h, c = self.cell(features, child_states)
        return h, c

# ---- Data preparation utilities ----
def list_all_tree_files(main_dir):
    file_list = []
    for root, dirs, files in os.walk(main_dir):
        for fname in files:
            if fname == "tree_representation.json":
                file_list.append(os.path.join(root, fname))
    return file_list

def split_train_test(files, test_size=30, seed=42):
    random.seed(seed)
    files = [f for f in files]
    random.shuffle(files)
    return files[test_size:], files[:test_size]

# ---- Training and evaluation ----
def train_model(model, train_dataset, val_dataset, scaler, epochs=100, lr=1e-3, batch_size=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for trees, targets in train_loader:
            optimizer.zero_grad()
            preds = []
            for i in range(len(trees)):
                pred = model(trees[i])
                preds.append(pred)
            preds = torch.cat(preds, dim=0)
            targets = torch.tensor(targets, dtype=torch.float32, device=device).view(-1, 1)
            targets_scaled = scaler.transform(targets.cpu().numpy())
            targets_scaled = torch.tensor(targets_scaled, dtype=torch.float32, device=device)
            loss = criterion(preds, targets_scaled)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for trees, targets in val_loader:
                preds = []
                for i in range(len(trees)):
                    pred = model(trees[i])
                    preds.append(pred)
                preds = torch.cat(preds, dim=0)
                targets = torch.tensor(targets, dtype=torch.float32, device=device).view(-1, 1)
                targets_scaled = scaler.transform(targets.cpu().numpy())
                targets_scaled = torch.tensor(targets_scaled, dtype=torch.float32, device=device)
                loss = criterion(preds, targets_scaled)
                val_losses.append(loss.item())
        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        print(f"Epoch {epoch+1}: Train Loss {avg_train:.4f}, Val Loss {avg_val:.4f}")
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), "best_tree_lstm.pth")
        else:
            patience_counter += 1
        if patience_counter > patience:
            print("Early stopping.")
            break
    model.load_state_dict(torch.load("best_tree_lstm.pth"))
    return model

def evaluate_model(model, dataset, scaler, file_list):
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=1)
    model.eval()
    all_preds = []
    all_targets = []
    for i, (trees, targets) in enumerate(loader):
        with torch.no_grad():
            pred = model(trees[0])
            pred = pred.cpu().numpy().flatten()
            pred_unscaled = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
            all_preds.append(pred_unscaled[0])
            all_targets.append(targets[0].item())
    errors = [abs(p-t)/t*100 if t > 0 else 0 for p, t in zip(all_preds, all_targets)]
    for fname, t, p, e in zip(file_list, all_targets, all_preds, errors):
        print(f"{fname}: Actual={t:.2f} ms, Predicted={p:.2f} ms, Error={e:.2f}%")
    print(f"MAE: {np.mean(np.abs(np.array(all_preds)-np.array(all_targets))):.2f} ms")
    print(f"MAPE: {np.mean(errors):.2f}%")

# ---- Main script ----
def main():
    main_dir = "Tree_Output"
    all_files = list_all_tree_files(main_dir)
    train_files, test_files = split_train_test(all_files, test_size=30)
    print(f"Train files: {len(train_files)}, Test files: {len(test_files)}")
    train_dataset = HalideTreeDataset(train_files)
    test_dataset = HalideTreeDataset(test_files)
    # Use execution times for scaling
    y_train = np.array(train_dataset.targets).reshape(-1, 1)
    scaler = RobustScaler().fit(y_train)
    # Model
    feature_size = len(FIXED_FEATURES)
    model = RecursiveTreeLSTM(feature_size, hidden_size=128)
    model = train_model(model, train_dataset, test_dataset, scaler, epochs=100, lr=1e-3)
    print("\nEvaluation on test set:")
    evaluate_model(model, test_dataset, scaler, test_files)

if __name__ == "__main__":
    main()
