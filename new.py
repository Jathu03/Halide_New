import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import sys

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torch.cuda.amp import GradScaler, autocast
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch is not available. Some functionality will be limited.")
    TORCH_AVAILABLE = False

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
if TORCH_AVAILABLE:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

class GraphDataset(Dataset):
    def __init__(self, features, execution_times):
        self.features = features
        self.execution_times = execution_times
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.execution_times[idx]

class TransformerLSTMExecutionTimePredictor(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_lstm_layers=3, num_transformer_layers=2, num_heads=8, dropout=0.4):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for this model")
            
        super(TransformerLSTMExecutionTimePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.num_directions = 2
        
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.layer_norm_input = nn.LayerNorm(hidden_size)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, hidden_size))  # Max sequence length 100
        
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=hidden_size if i == 0 else hidden_size * self.num_directions,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0,
                bidirectional=True
            ) for i in range(num_lstm_layers)
        ])
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size * self.num_directions,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_transformer_layers)
        self.layer_norm_transformer = nn.LayerNorm(hidden_size * self.num_directions)
        
        self.batch_norm = nn.BatchNorm1d(hidden_size * self.num_directions)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        out = self.input_proj(x)
        out = self.layer_norm_input(out)
        
        # Add positional encoding
        out = out + self.pos_encoder[:, :seq_len, :].to(x.device)
        
        for i, lstm in enumerate(self.lstm_layers):
            h0 = torch.zeros(1 * self.num_directions, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(1 * self.num_directions, batch_size, self.hidden_size).to(x.device)
            lstm_out, _ = lstm(out, (h0, c0))
            if i > 0:
                out = lstm_out + out
            else:
                out = lstm_out
        
        out = self.transformer(out)
        out = self.layer_norm_transformer(out)
        
        attention_weights = self.attention(out)
        context_vector = torch.sum(attention_weights * out, dim=1)
        context_vector = self.batch_norm(context_vector)
        
        out = self.fc(context_vector)
        return out.squeeze()

def focal_loss(outputs, targets, gamma=2.0, alpha=0.25):
    mse_loss = nn.MSELoss(reduction='none')(outputs, targets)
    pt = torch.exp(-mse_loss)
    focal_loss = alpha * (1 - pt) ** gamma * mse_loss
    return focal_loss.mean()

def custom_loss(outputs, targets):
    mse_loss = nn.MSELoss()(outputs, targets)
    l1_loss = nn.L1Loss()(outputs, targets)
    percentage_error = torch.abs((outputs - targets) / (targets + 1e-6))
    weights = 1.0 / (torch.abs(targets) + 1e-2)
    weighted_percentage_loss = torch.mean(percentage_error * weights)
    focal = focal_loss(outputs, targets)
    return 0.3 * mse_loss + 0.2 * l1_loss + 0.3 * weighted_percentage_loss + 0.2 * focal

def examine_json_structure(file_path):
    try:
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        
        has_without_extern = "without_extern" in json_data
        if has_without_extern:
            without_extern = json_data.get("without_extern", {})
            has_global_features = "global_features" in without_extern
            has_execution_time = "execution_time_ms" in without_extern.get("global_features", {})
            execution_time = without_extern.get("global_features", {}).get("execution_time_ms", "N/A")
            has_nodes = "nodes" in without_extern
            num_nodes = len(without_extern.get("nodes", []))
        else:
            has_global_features = "global_features" in json_data
            has_execution_time = "execution_time_ms" in json_data.get("global_features", {})
            execution_time = json_data.get("global_features", {}).get("execution_time_ms", "N/A")
            has_nodes = "nodes" in json_data
            num_nodes = len(json_data.get("nodes", []))
        
        node_structure = {}
        if num_nodes > 0:
            if has_without_extern:
                first_node = without_extern["nodes"][0] if isinstance(without_extern["nodes"], list) and len(without_extern["nodes"]) > 0 else {}
            else:
                first_node = json_data["nodes"][0] if isinstance(json_data["nodes"], list) and len(json_data["nodes"]) > 0 else {}
                
            if isinstance(first_node, dict):
                node_structure = {key: type(value).__name__ for key, value in first_node.items()}
                if "stages" in first_node:
                    if isinstance(first_node["stages"], list):
                        node_structure["stages"] = f"list[{len(first_node['stages'])}]"
                        if first_node["stages"] and isinstance(first_node["stages"][0], dict):
                            node_structure["stages_keys"] = list(first_node["stages"][0].keys())
                    else:
                        node_structure["stages"] = type(first_node["stages"]).__name__
        
        return {
            "has_without_extern": has_without_extern,
            "has_global_features": has_global_features,
            "has_execution_time": has_execution_time,
            "execution_time": execution_time,
            "has_nodes": has_nodes,
            "num_nodes": num_nodes,
            "node_structure": node_structure
        }
    except Exception as e:
        return {"error": str(e)}

def extract_features_from_json(json_data, debug=False):
    if "without_extern" in json_data:
        json_data = json_data["without_extern"]
    
    if debug:
        print("JSON keys:", json_data.keys())
    
    global_features = json_data.get("global_features", {})
    if debug and not global_features:
        print("Warning: No global_features found in JSON")
    
    execution_time = global_features.get("execution_time_ms", 0)
    if debug:
        print(f"Execution time: {execution_time}")
    
    if execution_time <= 0:
        if debug:
            print(f"Skipping file due to invalid execution time: {execution_time}")
        return None, None
    
    nodes = json_data.get("nodes", [])
    if debug:
        print(f"Number of nodes: {len(nodes)}")
        if nodes and isinstance(nodes, list) and len(nodes) > 0:
            print(f"First node keys: {nodes[0].keys() if isinstance(nodes[0], dict) else 'Not a dict'}")
        elif nodes and isinstance(nodes, dict):
            print(f"Nodes keys: {nodes.keys()}")
    
    node_features = []
    num_nodes = len(nodes)
    total_ops = global_features.get("total_ops", 0.0)
    total_bytes = global_features.get("total_bytes", 0.0)
    
    global_feature_vector = [
        np.log1p(total_bytes) if total_bytes > 0 else 0.0,
        np.log1p(total_ops) if total_ops > 0 else 0.0,
        num_nodes,
        np.log1p(total_bytes / (num_nodes + 1e-6)) if total_bytes > 0 else 0.0,
        np.log1p(total_ops / (total_bytes + 1e-6)) if total_ops > 0 and total_bytes > 0 else 0.0,
        num_nodes / (total_ops + 1e-6)  # Node density
    ]
    
    if isinstance(nodes, list):
        node_list = nodes
    elif isinstance(nodes, dict):
        node_list = [value for key, value in nodes.items() if isinstance(value, dict)]
    else:
        node_list = []
    
    # Compute node centrality (approximated by number of connections)
    centrality_scores = []
    for node in node_list:
        if not isinstance(node, dict):
            centrality_scores.append(0.0)
            continue
        connections = sum([1 for key in ['input', 'output', 'boundary_condition', 'wrapper'] if node.get(key, False)])
        centrality_scores.append(connections)
    if centrality_scores:
        max_centrality = max(centrality_scores) + 1e-6
        centrality_scores = [score / max_centrality for score in centrality_scores]
    
    for i, node in enumerate(node_list):
        if not isinstance(node, dict):
            continue
            
        node_feature_vector = []
        node_feature_vector.append(1 if node.get("input", False) else 0)
        node_feature_vector.append(1 if node.get("output", False) else 0)
        node_feature_vector.append(1 if node.get("pointwise", False) else 0)
        node_feature_vector.append(1 if node.get("boundary_condition", False) else 0)
        node_feature_vector.append(1 if node.get("wrapper", False) else 0)
        node_feature_vector.append(centrality_scores[i] if i < len(centrality_scores) else 0.0)
        
        stages = node.get("stages", [])
        if stages:
            first_stage = stages[0] if isinstance(stages, list) and len(stages) > 0 else stages if isinstance(stages, dict) else {}
            pipeline_features = first_stage.get("pipeline_features", {})
            schedule_features = pipeline_features.get("schedule_features", {})
            
            if debug and i == 0:
                print(f"Stage type: {type(stages)}")
                print(f"Pipeline features keys: {pipeline_features.keys() if pipeline_features else 'None'}")
                print(f"Schedule features keys: {schedule_features.keys() if schedule_features else 'None'}")
            
            important_features = [
                "allocation_bytes_read_per_realization",
                "bytes_at_production",
                "bytes_at_realization",
                "bytes_at_root",
                "bytes_at_task",
                "inlined_calls",
                "inner_parallelism",
                "num_productions",
                "num_realizations",
                "num_scalars",
                "num_vectors",
                "outer_parallelism",
                "points_computed_total",
                "vector_size",
                "working_set"
            ]
            
            for feature in important_features:
                try:
                    value = float(schedule_features.get(feature, 0.0))
                    node_feature_vector.append(np.log1p(value) if value > 0 else 0.0)
                except (ValueError, TypeError):
                    if debug:
                        print(f"Warning: Could not convert {feature} to float")
                    node_feature_vector.append(0.0)
            
            op_histogram = pipeline_features.get("op_histogram", {})
            float_ops = op_histogram.get("Float", {})
            
            if debug and i == 0:
                print(f"Op histogram keys: {op_histogram.keys() if op_histogram else 'None'}")
                print(f"Float ops keys: {float_ops.keys() if float_ops else 'None'}")
            
            try:
                total_ops = sum(float_ops.values())
                node_feature_vector.append(np.log1p(total_ops) if total_ops > 0 else 0.0)
            except:
                node_feature_vector.append(0.0)
                if debug:
                    print("Warning: Could not calculate total_ops")
            
            try:
                compute_ops = float_ops.get("Add", 0) + float_ops.get("Sub", 0) + float_ops.get("Mul", 0) + float_ops.get("Div", 0)
                node_feature_vector.append(np.log1p(compute_ops) if compute_ops > 0 else 0.0)
            except:
                node_feature_vector.append(0.0)
                if debug:
                    print("Warning: Could not calculate compute_ops")
            
            try:
                memory_ops = float_ops.get("Variable", 0) + float_ops.get("Param", 0) + float_ops.get("ImageCall", 0)
                node_feature_vector.append(np.log1p(memory_ops) if memory_ops > 0 else 0.0)
            except:
                node_feature_vector.append(0.0)
                if debug:
                    print("Warning: Could not calculate memory_ops")
            
            try:
                control_ops = float_ops.get("Select", 0) + float_ops.get("Let", 0) + float_ops.get("FuncCall", 0)
                node_feature_vector.append(np.log1p(control_ops) if control_ops > 0 else 0.0)
            except:
                node_feature_vector.append(0.0)
                if debug:
                    print("Warning: Could not calculate control_ops")
        
        else:
            node_feature_vector.extend([0.0] * (15 + 4))
        
        node_feature_vector.extend(global_feature_vector)
        node_features.append(node_feature_vector)
    
    if not node_features:
        if debug:
            print("No valid nodes found in JSON data")
        return None, None
    
    max_feature_len = max(len(f) for f in node_features) if node_features else 0
    if debug:
        print(f"Max feature length: {max_feature_len}")
        feature_lengths = [len(f) for f in node_features]
        if len(set(feature_lengths)) > 1:
            print(f"Warning: Inconsistent feature lengths: {feature_lengths}")
    
    padded_features = [f + [0.0] * (max_feature_len - len(f)) for f in node_features]
    features = np.array(padded_features, dtype=np.float32)
    if debug:
        print(f"Features shape: {features.shape}")
    
    return features, execution_time

def process_data_directory(root_dir, debug=False, max_files=None):
    all_features = []
    all_execution_times = []
    file_paths = []
    
    skipped_format = 0
    skipped_execution = 0
    skipped_other = 0
    processed = 0
    
    all_json_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.json'):
                all_json_files.append(os.path.join(dirpath, filename))
                if max_files is not None and len(all_json_files) >= max_files:
                    break
        if max_files is not None and len(all_json_files) >= max_files:
            break
    
    if debug:
        print(f"Found {len(all_json_files)} JSON files")
        if not all_json_files:
            print(f"No JSON files found in {root_dir}")
            if not os.path.exists(root_dir):
                print(f"Directory {root_dir} does not exist!")
            else:
                print(f"Contents of {root_dir}:")
                for item in os.listdir(root_dir):
                    print(f"  {item}")
    
    for file_path in tqdm(all_json_files, desc="Processing JSON files"):
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            if debug and processed < 2:
                print(f"\nExamining structure of {file_path}:")
                structure = examine_json_structure(file_path)
                for key, value in structure.items():
                    print(f"  {key}: {value}")
            
            features, execution_time = extract_features_from_json(json_data, debug=(debug and processed < 2))
            
            if features is None:
                if execution_time is None:
                    if "without_extern" in json_data:
                        global_features = json_data["without_extern"].get("global_features", {})
                    else:
                        global_features = json_data.get("global_features", {})
                    
                    exec_time = global_features.get("execution_time_ms", 0)
                    if exec_time <= 0:
                        skipped_execution += 1
                        if debug and skipped_execution < 5:
                            print(f"Skipped due to zero/negative execution time: {file_path}")
                    else:
                        skipped_other += 1
                        if debug and skipped_other < 5:
                            print(f"Skipped due to other reasons: {file_path}")
                else:
                    skipped_other += 1
            elif execution_time is not None and execution_time > 0:
                all_features.append(features)
                all_execution_times.append(execution_time)
                file_paths.append(file_path)
                processed += 1
        except json.JSONDecodeError as e:
            skipped_format += 1
            if debug and skipped_format < 5:
                print(f"JSON decode error in {file_path}: {e}")
        except Exception as e:
            skipped_format += 1
            if debug and skipped_format < 5:
                print(f"Error processing {file_path}: {e}")
    
    print(f"Processed {len(all_features)} valid JSON files")
    print(f"Skipped {skipped_format} files with format errors")
    print(f"Skipped {skipped_execution} files with zero/negative execution time")
    print(f"Skipped {skipped_other} files for other reasons")
    
    return all_features, all_execution_times, file_paths

def pad_sequences(sequences, max_length=None):
    if not sequences:
        return np.array([])
        
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            padded = np.pad(seq, ((0, max_length - len(seq)), (0, 0)), 'constant')
        else:
            padded = seq[:max_length]
        padded_sequences.append(padded)
    
    return np.array(padded_sequences)

def augment_data(features, execution_times, file_paths):
    augmented_features = []
    augmented_times = []
    augmented_paths = []
    
    percentiles = [10, 30, 50, 70, 90]
    for perc in percentiles:
        threshold_lower = np.percentile(execution_times, max(0, perc - 10))
        threshold_upper = np.percentile(execution_times, perc)
        indices = np.where((execution_times > threshold_lower) & (execution_times <= threshold_upper))[0]
        
        for idx in indices:
            for i in range(1):  # Minimal augmentation
                noise_scale = 0.01
                noise = np.random.normal(0, noise_scale, features[idx].shape)
                augmented_features.append(features[idx] + noise)
                augmented_times.append(execution_times[idx] * np.random.uniform(0.99, 1.01))
                augmented_paths.append(file_paths[idx] + f"_augmented_{perc}_{i}")
    
    if augmented_features:
        return np.vstack([features, np.array(augmented_features)]), \
               np.concatenate([execution_times, np.array(augmented_times)]), \
               file_paths + augmented_paths
    else:
        return features, execution_times, file_paths

# Ranger optimizer (combination of RAdam and Lookahead)
class Ranger(optim.Optimizer):
    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, betas=(0.95, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, alpha=alpha, k=k, betas=betas, eps=eps, weight_decay=weight_decay)
        super(Ranger, self).__init__(params, defaults)
        
        self.k = k
        self.alpha = alpha
        self.beta1, self.beta2 = betas
        
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['slow_buffer'] = torch.empty_like(p.data)
                state['slow_buffer'].copy_(p.data)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Ranger does not support sparse gradients')
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr']
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                if state['step'] % group['k'] == 0:
                    slow_p = state['slow_buffer']
                    slow_p.add_(p.data - slow_p, alpha=group['alpha'])
                    p.data.copy_(slow_p)
                
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])
        
        return loss

def train_pytorch_model(X_train, y_train, X_val, y_val, X_test, y_test, file_paths_test, feature_dim):
    if not TORCH_AVAILABLE:
        print("PyTorch is not available. Skipping model training.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = GraphDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = GraphDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = GraphDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    batch_size = 16
    accumulation_steps = 8  # Effective batch size = 16 * 8 = 128
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    input_size = feature_dim
    hidden_size = 512
    num_lstm_layers = 3
    num_transformer_layers = 2
    num_heads = 8
    dropout = 0.4
    
    model = TransformerLSTMExecutionTimePredictor(
        input_size, hidden_size, num_lstm_layers, num_transformer_layers, num_heads, dropout
    )
    model = model.to(device)
    print(f"Model input size: {input_size}")
    print(model)
    
    optimizer = Ranger(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=5e-6)
    
    scaler = GradScaler() if device.type == 'cuda' else None
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    num_epochs = 400
    patience = 60
    train_losses = []
    val_losses = []
    
    warmup_epochs = 15
    initial_lr = 5e-5
    for param_group in optimizer.param_groups:
        param_group['lr'] = initial_lr
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for i, (features, targets) in enumerate(train_loader):
            features, targets = features.to(device), targets.to(device)
            
            if scaler:
                with autocast():
                    outputs = model(features)
                    loss = custom_loss(outputs, targets)
                    loss = loss / accumulation_steps
                scaler.scale(loss).backward()
                
                if (i + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(features)
                loss = custom_loss(outputs, targets)
                loss = loss / accumulation_steps
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = custom_loss(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        if epoch < warmup_epochs:
            lr = initial_lr + (0.001 - initial_lr) * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    model.load_state_dict(torch.load('best_model.pth'))
    
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    predictions_original = np.expm1(predictions)
    actuals_original = np.expm1(actuals)
    
    absolute_errors = np.abs(predictions_original - actuals_original)
    percentage_errors = (absolute_errors / actuals_original) * 100
    
    mean_absolute_error = np.mean(absolute_errors)
    mean_percentage_error = np.mean(percentage_errors)
    median_percentage_error = np.median(percentage_errors)
    
    print(f"Mean Absolute Error: {mean_absolute_error:.4f}")
    print(f"Mean Percentage Error: {mean_percentage_error:.2f}%")
    print(f"Median Percentage Error: {median_percentage_error:.2f}%")
    
    print("\nTest File Predictions:")
    for i, (pred, actual, error, path) in enumerate(zip(predictions_original, actuals_original, percentage_errors, file_paths_test)):
        print(f"{i+1}. {os.path.basename(path)}: Predicted={pred:.2f}ms, Actual={actual:.2f}ms, Error={error:.2f}%")
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(2, 2, 2)
    plt.scatter(actuals_original, predictions_original, alpha=0.5)
    plt.plot([min(actuals_original), max(actuals_original)], [min(actuals_original), max(actuals_original)], 'r--')
    plt.xlabel('Actual Execution Time (ms)')
    plt.ylabel('Predicted Execution Time (ms)')
    plt.title('Prediction vs Actual')
    
    plt.subplot(2, 1, 2)
    indices = np.arange(len(actuals_original))
    plt.plot(indices, actuals_original, label='Actual Execution Times', linewidth=2)
    plt.plot(indices, predictions_original, label='Predicted Execution Times', linewidth=2)
    plt.xlabel('Sample Index')
    plt.ylabel('Execution Time (ms)')
    plt.legend()
    plt.title('Actual vs Predicted Execution Times')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('execution_time_prediction_results_improved.png')
    plt.show()
    
    return model, predictions_original, actuals_original

def main(debug=True, max_files=None):
    root_dir = "Graph_Output"
    all_features, all_execution_times, file_paths = process_data_directory(root_dir, debug=debug, max_files=max_files)
    
    if not all_features:
        print("No valid data found. Exiting.")
        return
    
    execution_times = np.array(all_execution_times, dtype=np.float32)
    upper_limit = np.percentile(execution_times, 99)
    lower_limit = np.percentile(execution_times, 1)
    valid_indices = (execution_times <= upper_limit) & (execution_times >= lower_limit)
    all_features = [f for f, v in zip(all_features, valid_indices) if v]
    execution_times = execution_times[valid_indices]
    file_paths = [p for p, v in zip(file_paths, valid_indices) if v]
    
    padded_features = pad_sequences(all_features)
    print(f"Padded features shape: {padded_features.shape}")
    
    print(f"Execution times shape: {execution_times.shape}")
    print(f"Execution times range: {execution_times.min()} to {execution_times.max()}")
    
    padded_features, execution_times, file_paths = augment_data(padded_features, execution_times, file_paths)
    print(f"After augmentation - Features shape: {padded_features.shape}, Execution times shape: {execution_times.shape}, Paths count: {len(file_paths)}")
    
    log_execution_times = np.log1p(execution_times)
    
    test_size = min(20, len(padded_features) // 5)
    X_train_val, X_test, y_train_val, y_test, paths_train_val, paths_test = train_test_split(
        padded_features, log_execution_times, file_paths, test_size=test_size, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    print(f"Test set size: {X_test.shape}")
    
    feature_dim = X_train.shape[2]
    
    X_train_reshaped = X_train.reshape(-1, feature_dim)
    X_val_reshaped = X_val.reshape(-1, feature_dim)
    X_test_reshaped = X_test.reshape(-1, feature_dim)
    
    scaler = RobustScaler(quantile_range=(2.5, 97.5))
    X_train_reshaped = scaler.fit_transform(X_train_reshaped)
    X_val_reshaped = scaler.transform(X_val_reshaped)
    X_test_reshaped = scaler.transform(X_test_reshaped)
    
    X_train = X_train_reshaped.reshape(X_train.shape)
    X_val = X_val_reshaped.reshape(X_val.shape)
    X_test = X_test_reshaped.reshape(X_test.shape)
    
    if TORCH_AVAILABLE:
        train_pytorch_model(X_train, y_train, X_val, y_val, X_test, y_test, paths_test, feature_dim)
    else:
        print("PyTorch is not available. Skipping model training.")

if __name__ == "__main__":
    main(debug=True, max_files=500)
