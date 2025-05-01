import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# ----------- Feature Extraction Utilities -----------
FEATURE_NAMES = [
    "cache_hits", "cache_misses",
    "num_realizations", "num_productions", "points_computed_per_realization",
    "points_computed_per_production", "points_computed_total",
    "points_computed_minimum", "innermost_loop_extent", "innermost_pure_loop_extent",
    "unrolled_loop_extent", "inner_parallelism", "outer_parallelism",
    "bytes_at_realization", "bytes_at_production", "bytes_at_root",
    "innermost_bytes_at_realization", "innermost_bytes_at_production", "innermost_bytes_at_root",
    "inlined_calls", "unique_bytes_read_per_realization", "unique_lines_read_per_realization",
    "allocation_bytes_read_per_realization", "working_set", "vector_size", "native_vector_size",
    "num_vectors", "num_scalars", "scalar_loads_per_vector", "vector_loads_per_vector",
    "scalar_loads_per_scalar", "bytes_at_task", "innermost_bytes_at_task",
    "unique_bytes_read_per_vector", "unique_lines_read_per_vector", "unique_bytes_read_per_task",
    "unique_lines_read_per_task", "working_set_at_task", "working_set_at_production",
    "working_set_at_realization", "working_set_at_root"
]

OP_NAMES = [
    "Constant", "Cast", "Variable", "Param", "Add", "Sub", "Mod", "Mul", "Div", "Min", "Max",
    "EQ", "NE", "LT", "LE", "And", "Or", "Not", "Select", "ImageCall", "FuncCall", "SelfCall",
    "ExternCall", "Let"
]
MEMORY_PATTERN_NAMES = ["Pointwise", "Transpose", "Broadcast", "Slice"]

def extract_node_features(node):
    features = []
    features.append(node.get("cache_hits", 0))
    features.append(node.get("cache_misses", 0))
    sched = node.get("scheduling", {})
    for fname in FEATURE_NAMES[2:]:
        features.append(sched.get(fname, 0))
    op_hist = node.get("op_histogram", {})
    for op in OP_NAMES:
        features.append(op_hist.get(op, 0))
    mem_patterns = node.get("memory_patterns", {})
    for mp in MEMORY_PATTERN_NAMES:
        pattern = mem_patterns.get(mp, [0, 0, 0, 0])
        features.extend(pattern)
    return np.array(features, dtype=np.float32)

def build_tree(node):
    features = extract_node_features(node)
    children = [build_tree(child) for child in node.get("children", [])]
    return {"features": features, "children": children}

class HalideTreeDataset(Dataset):
    def __init__(self, root_dir, scaler=None, fit_scaler=False):
        self.samples = []
        self.scaler = scaler
        feature_list = []
        for prog_folder in os.listdir(root_dir):
            prog_path = os.path.join(root_dir, prog_folder)
            if not os.path.isdir(prog_path): continue
            for sched_folder in os.listdir(prog_path):
                sched_path = os.path.join(prog_path, sched_folder)
                json_file = os.path.join(sched_path, "tree_representation.json")
                if not os.path.isfile(json_file): continue
                with open(json_file) as f:
                    data = json.load(f)
                exec_time = None
                for child in data.get("children", []):
                    if child.get("name") == "Global Features":
                        exec_time = child.get("execution_time_ms", None)
                if exec_time is None or exec_time <= 0:
                    continue
                tree = build_tree(data)
                self.samples.append((tree, exec_time))
                if fit_scaler:
                    def collect_feats(node):
                        feats = [node["features"]]
                        for c in node["children"]:
                            feats.extend(collect_feats(c))
                        return feats
                    feature_list.extend(collect_feats(tree))
        if fit_scaler:
            self.scaler = StandardScaler().fit(np.vstack(feature_list))
        for i, (tree, exec_time) in enumerate(self.samples):
            def norm_tree(node):
                node["features"] = self.scaler.transform([node["features"]])[0]
                for c in node["children"]:
                    norm_tree(c)
            norm_tree(tree)
            self.samples[i] = (tree, exec_time)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ----------- Recursive LSTM Model -----------
class RecursiveLSTM(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.node_embed = nn.Linear(feature_dim, hidden_dim)
        self.lstm_cell = nn.LSTMCell(hidden_dim, hidden_dim)
        self.out_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, tree):
        def recur(node):
            node_emb = self.node_embed(torch.tensor(node["features"], dtype=torch.float32))
            if node["children"]:
                child_embs = [recur(c)[0] for c in node["children"]]
                child_emb = torch.mean(torch.stack(child_embs), dim=0)
            else:
                child_emb = torch.zeros_like(node_emb)
            h, c = self.lstm_cell(node_emb, (child_emb, torch.zeros_like(child_emb)))
            return h, c
        root_emb, _ = recur(tree)
        return self.out_net(root_emb).squeeze(-1)

# ----------- Custom Collate Function -----------
def tree_collate_fn(batch):
    return batch[0]

# ----------- Checkpointing Utilities -----------
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        epoch = state['epoch']
        loss = state['loss']
        print(f"Checkpoint loaded from {checkpoint_path} at epoch {epoch}")
        return epoch, loss
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return 0, None

# ----------- Prediction and Error Percentage -----------
def predict_and_evaluate(model, dataset, scaler, num_schedules=30):
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(min(num_schedules, len(dataset))):
            tree, true_time = dataset[i]
            pred_time = model(tree).item()
            error_percentage = abs(pred_time - true_time) / true_time * 100 if true_time != 0 else 0
            errors.append(error_percentage)
            print(f"Schedule {i+1}: True={true_time:.4f}, Predicted={pred_time:.4f}, Error%={error_percentage:.2f}")
    mean_error = np.mean(errors) if errors else 0
    print(f"Mean Error Percentage over {len(errors)} schedules: {mean_error:.2f}%")
    return mean_error

# ----------- Training Loop -----------
def train_model(root_dir, model_save_path, scaler_save_path, checkpoint_path,
                epochs=20, hidden_dim=128, lr=1e-3, resume=False):

    # First pass: fit scaler
    dataset = HalideTreeDataset(root_dir, fit_scaler=True)
    scaler = dataset.scaler
    joblib.dump(scaler, scaler_save_path)
    # Second pass: normalized dataset
    dataset = HalideTreeDataset(root_dir, scaler=scaler)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=tree_collate_fn)
    feature_dim = len(extract_node_features({}))
    model = RecursiveLSTM(feature_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    start_epoch = 0
    last_loss = None
    if resume:
        start_epoch, last_loss = load_checkpoint(model, optimizer, checkpoint_path)

    model.train()
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        for tree, exec_time in dataloader:
            optimizer.zero_grad()
            pred = model(tree)
            loss = loss_fn(pred, torch.tensor(exec_time, dtype=torch.float32))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        save_checkpoint(model, optimizer, epoch+1, avg_loss, checkpoint_path)
    torch.save(model.state_dict(), model_save_path)
    print("Training complete. Model and scaler saved.")

    # Prediction and error percentage for 30 schedules
    print("\n--- Prediction and Error Percentage for 30 Schedules ---")
    predict_and_evaluate(model, dataset, scaler, num_schedules=30)

# ----------- Usage -----------
if __name__ == "__main__":
    train_model(
        root_dir="Tree_Output",
        model_save_path="halide_recursive_lstm.pt",
        scaler_save_path="halide_scaler.pkl",
        checkpoint_path="halide_checkpoint.pt",
        epochs=20,
        hidden_dim=128,
        lr=1e-3,
        resume=True  # Set to True to resume from checkpoint if available
    )
