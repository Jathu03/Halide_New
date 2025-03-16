import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
import random
import matplotlib.pyplot as plt

def get_execution_time(schedule_data):
    if "execution_times" in schedule_data:
        exec_times = schedule_data["execution_times"]
        return float(np.mean(exec_times))
    print("Warning: No execution times found in schedule")
    return None

def extract_features_from_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None
    
    all_features = []
    
    for func_id, func_data in data.items():
        if "program_annotation" not in func_data or "schedules_list" not in func_data:
            print(f"Warning: Missing required fields in {file_path} for {func_id}")
            continue
        
        prog_annot = func_data["program_annotation"]
        iterators = prog_annot.get("iterators", {})
        computations = prog_annot.get("computations", {})
        
        # Global context features
        global_features = {
            'memory_size': prog_annot.get("memory_size", 0),
            'computation_count': len(computations),
            'access_count': sum(len(comp.get("accesses", [])) for comp in computations.values()),
        }
        
        schedules = func_data["schedules_list"]
        for idx, schedule in enumerate(schedules):
            execution_time = get_execution_time(schedule)
            if execution_time is None or execution_time <= 0:
                continue
            
            features = global_features.copy()
            features['execution_time'] = execution_time
            features['log_execution_time'] = np.log1p(execution_time)
            
            # Tree representation: list of nodes with features and child indices
            tree_nodes = []
            node_to_idx = {}
            
            def traverse_tree(node, depth, parent_idx=None):
                # Node features: [loop_range, tile_factor, unroll_factor, parallel_flag, n_comps, n_accesses, depth]
                node_features = [0] * 7
                node_idx = len(tree_nodes)
                node_to_idx[id(node)] = node_idx
                
                # Loop range
                it_id = node.get("iterator_id", "")
                if it_id in iterators:
                    it = iterators[it_id]
                    loop_range = it.get("upper_bound", 0) - it.get("lower_bound", 0)
                    node_features[0] = loop_range if isinstance(loop_range, (int, float)) else 0
                
                # Transformations and computations
                comp_key = node.get("computation", "")
                if comp_key in schedule and isinstance(schedule[comp_key], dict):
                    comp_data = schedule[comp_key]
                    if "tiling" in comp_data and comp_data["tiling"]:
                        factors = comp_data["tiling"].get("tiling_factors", [])
                        node_features[1] = factors[0] if factors else 0
                    if "unrolling_factor" in comp_data and comp_data["unrolling_factor"]:
                        node_features[2] = comp_data["unrolling_factor"]
                    if "parallelized_dim" in comp_data and comp_data["parallelized_dim"]:
                        node_features[3] = 1
                    if comp_key in computations:
                        comp = computations[comp_key]
                        node_features[4] = 1  # n_comps
                        node_features[5] = len(comp.get("accesses", []))
                
                node_features[6] = depth
                
                # Add node with valid children only
                child_list = node.get("child_list", [])
                children = [node_to_idx[id(child)] for child in child_list if id(child) in node_to_idx]
                tree_nodes.append({'features': node_features, 'children': children})
                
                # Traverse children
                for child in child_list:
                    traverse_tree(child, depth + 1, node_idx)
            
            if "tree_structure" in schedule and "roots" in schedule["tree_structure"]:
                roots = schedule["tree_structure"]["roots"]
                for root in roots:
                    traverse_tree(root, 0)
            
            features['tree_nodes'] = tree_nodes
            all_features.append(features)
    
    return all_features if all_features else None

def process_directory(directory_path):
    all_features = []
    file_names = []
    
    json_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.json')])
    
    for filename in json_files:
        file_path = os.path.join(directory_path, filename)
        features_list = extract_features_from_file(file_path)
        if features_list is not None:
            all_features.extend(features_list)
            file_names.extend([f"{filename}_schedule_{i}" for i in range(len(features_list))])
    
    if len(all_features) < 60:
        print(f"Error: Only {len(all_features)} valid schedules found in {directory_path}")
        return None, None, None
    
    combined = list(zip(all_features, file_names))
    random.shuffle(combined)
    all_features, file_names = zip(*combined)
    
    train_features = list(all_features[:-10])
    test_features = list(all_features[-10:])
    train_file_names = list(file_names[:-10])
    test_file_names = list(file_names[-10:])
    
    print(f"Processed {directory_path}: {len(train_features)} training schedules, {len(test_features)} test schedules")
    
    return train_features, test_features, test_file_names

class TreeDataset(Dataset):
    def __init__(self, features, max_nodes=10):
        self.max_nodes = max_nodes
        self.X_nodes = []
        self.X_children = []
        self.y = []
        self.global_features = []
        
        global_keys = ['memory_size', 'computation_count', 'access_count']
        
        for feat in features:
            nodes = feat.pop("tree_nodes")
            # Pad or truncate to max_nodes
            node_features = np.zeros((max_nodes, 7))
            children = np.full((max_nodes, max_nodes), -1, dtype=int)  # -1 for no child
            for i, node in enumerate(nodes[:max_nodes]):
                node_features[i] = node['features']
                for j, child_idx in enumerate(node['children']):
                    if child_idx is not None and child_idx < max_nodes:  # Check for None and bounds
                        children[i, j] = child_idx
            
            self.X_nodes.append(node_features)
            self.X_children.append(children)
            self.y.append(feat["log_execution_time"])
            self.global_features.append([feat[k] for k in global_keys])
        
        self.X_nodes = np.array(self.X_nodes)  # [n_samples, max_nodes, 7]
        self.X_children = np.array(self.X_children)  # [n_samples, max_nodes, max_nodes]
        self.y = np.array(self.y).reshape(-1, 1)
        self.global_features = np.array(self.global_features)  # [n_samples, 3]
        
        # Scale global features
        scaler_global = StandardScaler()
        self.global_features = scaler_global.fit_transform(self.global_features)
        
        # Scale target
        self.scaler_y = StandardScaler()
        self.y = self.scaler_y.fit_transform(self.y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # Combine node features with repeated global features
        global_expanded = np.repeat(self.global_features[idx][np.newaxis, :], self.max_nodes, axis=0)
        X_combined = np.concatenate([self.X_nodes[idx], global_expanded], axis=1)  # [max_nodes, 10]
        return (torch.FloatTensor(X_combined),
                torch.LongTensor(self.X_children[idx]),
                torch.FloatTensor(self.y[idx]))

class TreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1, dropout_rate=0.5):
        super(TreeLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        # Tree-LSTM gates
        self.W_iou = nn.Linear(input_size, 3 * hidden_size)  # Input, Output, Update gates
        self.U_iou = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.W_f = nn.Linear(input_size, hidden_size)  # Forget gate
        self.U_f = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Output layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, node_features, children):
        batch_size, max_nodes, input_size = node_features.size()
        h = torch.zeros(batch_size, max_nodes, self.hidden_size).to(node_features.device)
        c = torch.zeros(batch_size, max_nodes, self.hidden_size).to(node_features.device)
        
        # Process nodes bottom-up
        for node_idx in range(max_nodes - 1, -1, -1):  # Start from leaves
            x = node_features[:, node_idx, :]
            child_h = []
            child_c = []
            for child_idx in range(max_nodes):
                if children[:, node_idx, child_idx].max() != -1:
                    valid_child_idx = children[:, node_idx, child_idx]
                    child_h.append(h[range(batch_size), valid_child_idx])
                    child_c.append(c[range(batch_size), valid_child_idx])
            
            child_h_sum = torch.sum(torch.stack(child_h, dim=1), dim=1) if child_h else torch.zeros(batch_size, self.hidden_size).to(x.device)
            child_c_stack = torch.stack(child_c, dim=1) if child_c else torch.zeros(batch_size, 0, self.hidden_size).to(x.device)
            
            # Tree-LSTM equations
            iou = self.W_iou(x) + self.U_iou(child_h_sum)
            i, o, u = torch.split(iou, self.hidden_size, dim=1)
            i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
            
            f = torch.sigmoid(self.W_f(x).unsqueeze(1) + self.U_f(child_h_sum).unsqueeze(0))
            if child_c_stack.size(1) > 0:
                c_tilde = torch.sum(f * child_c_stack, dim=1)
            else:
                c_tilde = torch.zeros(batch_size, self.hidden_size).to(x.device)
            
            c[:, node_idx, :] = i * u + c_tilde
            h[:, node_idx, :] = o * torch.tanh(c[:, node_idx, :])
        
        # Use root node (index 0) for prediction
        root_h = h[:, 0, :]
        out = self.dropout(root_h)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.fc2(out)
        return out

def custom_loss(y_pred, y_true):
    epsilon = 1e-8
    rel_error = torch.abs((y_pred - y_true) / (y_true.abs() + epsilon))
    return torch.mean(rel_error) + 0.5 * nn.MSELoss()(y_pred, y_true)

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=500, patience=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    def lr_lambda(current_step):
        warmup_steps = 10
        if current_step < warmup_steps:
            return current_step / warmup_steps
        return 1.0
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=20, verbose=True, min_lr=1e-7)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    learning_rates = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for node_features, children, targets in train_loader:
            node_features, children, targets = node_features.to(device), children.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(node_features, children)
            loss = criterion(outputs, targets)
            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch+1}")
                return None, None, None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            running_loss += loss.item() * node_features.size(0)
        
        if epoch < 10:
            warmup_scheduler.step()
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for node_features, children, targets in test_loader:
                node_features, children, targets = node_features.to(device), children.to(device), targets.to(device)
                outputs = model(node_features, children)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * node_features.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}')
        
        if val_loss < best_val_loss and not np.isnan(val_loss):
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates
    }
    with open('training_metrics.json', 'w') as f:
        json.dump(metrics, f)
    print("Training metrics saved to 'training_metrics.json'")
    
    return train_losses, val_losses, learning_rates

def plot_metrics(train_losses, val_losses, learning_rates):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.close()
    print("Loss plot saved as 'loss_plot.png'")
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rates, label='Learning Rate', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Over Epochs')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('lr_plot.png')
    plt.close()
    print("Learning rate plot saved as 'lr_plot.png'")

def evaluate_model(model, test_loader, y_scaler, file_names_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    y_pred_scaled = []
    y_test = []
    with torch.no_grad():
        for node_features, children, targets in test_loader:
            node_features, children, targets = node_features.to(device), children.to(device), targets.to(device)
            outputs = model(node_features, children)
            y_pred_scaled.append(outputs.cpu().numpy())
            y_test.append(targets.cpu().numpy())
    
    y_pred_scaled = np.concatenate(y_pred_scaled)
    y_test = np.concatenate(y_test)
    
    y_test_transformed = y_scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_transformed = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_test_actual = np.expm1(y_test_transformed)
    y_pred_actual = np.expm1(np.clip(y_pred_transformed, 0, None))
    
    print("\nEvaluation Results:")
    for i, file_name in enumerate(file_names_test):
        print(f"Schedule: {file_name}")
        print(f"  Actual execution time: {y_test_actual[i][0]:.6f} seconds")
        print(f"  Predicted execution time: {y_pred_actual[i][0]:.6f} seconds")
        print(f"  Error percentage: {abs(y_test_actual[i][0] - y_pred_actual[i][0]) / y_test_actual[i][0] * 100:.2f}%")
    
    mse = np.mean((y_test_actual - y_pred_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / (y_test_actual + 1e-8))) * 100
    
    print("\nOverall Model Performance:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"MAPE: {mape:.2f}%")
    
    residuals = y_test_actual - y_pred_actual
    print(f"\nDiagnostics:")
    print(f"Mean Residual: {np.mean(residuals):.6f}")
    print(f"Std of Residuals: {np.std(residuals):.6f}")
    
    return y_test_actual, y_pred_actual

def main(main_dir):
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"Processing directory: {main_dir}")
    train_features, test_features, test_file_names = process_directory(main_dir)
    
    if train_features is None or test_features is None:
        print("Error: Insufficient data to proceed")
        return None
    
    print(f"Total training samples: {len(train_features)}")
    print(f"Total test samples: {len(test_features)}")
    
    if len(train_features) < 50 or len(test_features) == 0:
        print("Error: Insufficient training data for robust model training")
        return None
    
    # Prepare datasets
    train_dataset = TreeDataset(train_features, max_nodes=10)
    test_dataset = TreeDataset(test_features, max_nodes=10)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Model
    model = TreeLSTM(
        input_size=10,  # 7 node features + 3 global features
        hidden_size=128,
        output_size=1,
        dropout_rate=0.5
    )
    
    criterion = custom_loss
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print("Building and training Tree-LSTM model...")
    train_losses, val_losses, learning_rates = train_model(model, train_loader, test_loader, criterion, optimizer)
    
    if train_losses is None or val_losses is None or learning_rates is None:
        print("Training failed due to NaN losses")
        return None
    
    plot_metrics(train_losses, val_losses, learning_rates)
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(model, test_loader, train_dataset.scaler_y, test_file_names)
    
    return model, train_dataset.scaler_y, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "Tiramisu"
    result = main(main_dir)
    if result is not None:
        model, y_scaler, y_test_actual, y_pred_actual = result
        print("\nTree-LSTM model training and prediction completed!")
    else:
        print("\nModel training failed!")
