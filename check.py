import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import OrderedDict, Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import random

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Feature extraction
def flatten_json(data, parent_key=''):
    features = {}
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}_{key}" if parent_key else key
            if key in ['metadata', 'description', 'comments']:
                continue
            nested_features = flatten_json(value, new_key)
            features.update(nested_features)
    elif isinstance(data, list):
        if parent_key:
            features[f"{parent_key}_length"] = len(data)
            if data and all(not isinstance(x, (dict, list)) for x in data):
                numeric_values = [float(item) for item in data if isinstance(item, (int, float)) and not np.isnan(item)]
                if numeric_values:
                    features[f"{parent_key}_mean"] = np.mean(numeric_values)
                    features[f"{parent_key}_median"] = np.median(numeric_values)
                    features[f"{parent_key}_std"] = np.std(numeric_values) if len(numeric_values) > 1 else 0
                    features[f"{parent_key}_max"] = np.max(numeric_values)
                    features[f"{parent_key}_min"] = np.min(numeric_values)
                    features[f"{parent_key}_range"] = features[f"{parent_key}_max"] - features[f"{parent_key}_min"]
            for i, value in enumerate(data[:3]):
                nested_features = flatten_json(value, f"{parent_key}_{i}")
                features.update(nested_features)
    else:
        try:
            val = float(data)
            if not np.isnan(val) and not np.isinf(val):
                features[parent_key] = val
            else:
                features[parent_key] = 0.0
        except (ValueError, TypeError):
            if isinstance(data, str):
                features[parent_key] = hash(str(data)) % 1000 / 1000.0
            elif data is True:
                features[parent_key] = 1.0
            elif data is False:
                features[parent_key] = 0.0
            else:
                features[parent_key] = 0.0
    return features

def extract_target(data):
    if "scheduling_data" in data:
        for item in data.get("scheduling_data", []):
            if item.get("name") == "total_execution_time_ms":
                val = item["value"]
                return val if isinstance(val, (int, float)) and val > 0 and not np.isnan(val) else None
    elif "schedule_details" in data and "execution_times" in data["schedule_details"]:
        times = [t for t in data["schedule_details"]["execution_times"] if isinstance(t, (int, float)) and t > 0 and not np.isnan(t)]
        return np.mean(times) if times else None
    return None

def extract_important_features(data):
    features = {}
    if "hardware" in data:
        hw = data["hardware"]
        features["num_cores"] = hw.get("num_cores", 0)
        features["has_gpu"] = 1.0 if hw.get("has_gpu", False) else 0.0
        features["memory_gb"] = hw.get("memory_gb", 0)
        if hw.get("memory_gb", 0) > 0 and hw.get("num_cores", 0) > 0:
            features["core_memory_ratio"] = hw.get("num_cores", 0) / hw.get("memory_gb", 1)
    
    if "programming_details" in data:
        nodes = data["programming_details"]["Nodes"]
        features["num_computations"] = len(nodes)
        op_counts = Counter()
        for node in nodes:
            for op in node["Details"]["Op histogram"]:
                op_type, count = op.split(":")
                op_counts[op_type.strip()] += int(count.strip())
        for op, count in op_counts.items():
            features[f"op_{op}"] = count
        features["num_edges"] = len(data["programming_details"]["Edges"])
    elif "program_annotation" in data:
        comps = data["program_annotation"]["computations"]
        features["num_computations"] = len(comps)
        features["num_iterators"] = len(data["program_annotation"]["iterators"])
        reductions = sum(1 for c in comps.values() if c["comp_is_reduction"])
        features["num_reductions"] = reductions
    
    if "scheduling_data" in data:
        sched = data["scheduling_data"]
        for item in sched[:-1]:
            sched_feat = item["Details"]["scheduling_feature"]
            features["inner_parallelism"] = sched_feat.get("inner_parallelism", 0)
            features["outer_parallelism"] = sched_feat.get("outer_parallelism", 0)
            features["vector_size"] = sched_feat.get("vector_size", 0)
            features["unrolled_loop_extent"] = sched_feat.get("unrolled_loop_extent", 0)
    elif "schedule_details" in data:
        sched = data["schedule_details"]
        tiling_total = 0
        unroll_total = 0
        parallel_count = 0
        for comp in sched:
            if comp not in ['fusions', 'sched_str', 'tree_structure', 'legality_check', 'exploration_method', 'execution_times']:
                tiling = sched[comp].get("tiling", {})
                tiling_total += sum(int(v) for v in tiling.values()) if tiling else 0
                unroll = sched[comp].get("unrolling_factor", 0) or 0
                unroll_total += unroll
                parallel_count += 1 if sched[comp].get("parallelized_dim") else 0
        features["total_tiling"] = tiling_total
        features["total_unrolling"] = unroll_total
        features["parallel_count"] = parallel_count
    return features

class EnhancedScheduleDataset(Dataset):
    def __init__(self, json_data_list, targets=None, feature_scaler=None, target_scaler=None, 
                 train=True, feature_selector=None, k_best_features=150):
        self.json_data = json_data_list
        
        print("Extracting features...")
        all_features = []
        for data in json_data_list:
            direct_features = extract_important_features(data)
            flat_features = flatten_json(data)
            combined_features = {**flat_features, **direct_features}
            all_features.append(combined_features)
        
        self.feature_names = sorted(set().union(*[set(f.keys()) for f in all_features]))
        print(f"Total features extracted: {len(self.feature_names)}")
        
        X = np.zeros((len(all_features), len(self.feature_names)))
        for i, features in enumerate(all_features):
            for j, name in enumerate(self.feature_names):
                val = features.get(name, 0.0)
                X[i, j] = val if not np.isnan(val) and not np.isinf(val) else 0.0
        
        if train and targets is not None:
            print("Performing feature selection...")
            self.feature_selector = SelectKBest(f_regression, k=min(k_best_features, X.shape[1]))
            X = self.feature_selector.fit_transform(X, targets)
            selected_indices = self.feature_selector.get_support(indices=True)
            self.selected_feature_names = [self.feature_names[i] for i in selected_indices]
            print(f"Selected {len(self.selected_feature_names)} features")
        elif feature_selector is not None:
            X = feature_selector.transform(X)
            self.feature_selector = feature_selector
            self.selected_feature_names = None
        else:
            self.feature_selector = None
            self.selected_feature_names = self.feature_names
        
        if train:
            self.feature_scaler = RobustScaler()
            self.X = self.feature_scaler.fit_transform(X)
        else:
            self.X = feature_scaler.transform(X)
            self.feature_scaler = feature_scaler
        
        if targets is not None:
            if train:
                self.target_scaler = PowerTransformer(method='yeo-johnson')
                targets_array = np.array(targets).reshape(-1, 1)
                self.y = self.target_scaler.fit_transform(targets_array).flatten()
            else:
                self.y = target_scaler.transform(np.array(targets).reshape(-1, 1)).flatten()
                self.target_scaler = target_scaler
            self.targets = torch.tensor(self.y, dtype=torch.float32).to(device)
        else:
            self.targets = None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.X[idx], dtype=torch.float32).to(device)
        if self.targets is not None:
            return features, self.targets[idx]
        return features

class ImprovedExecutionTimePredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128, 64], dropout=0.3):
        super(ImprovedExecutionTimePredictor, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        nn.init.kaiming_normal_(self.input_layer.weight, nonlinearity='leaky_relu')
        self.batch_norm_input = nn.BatchNorm1d(hidden_sizes[0])
        
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        
        for i in range(1, len(hidden_sizes)):
            layer = nn.Linear(hidden_sizes[i-1], hidden_sizes[i])
            nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
            self.hidden_layers.append(layer)
            self.batch_norms.append(nn.BatchNorm1d(hidden_sizes[i]))
            self.dropouts.append(nn.Dropout(dropout))
            if hidden_sizes[i-1] != hidden_sizes[i]:
                self.shortcuts.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i], bias=False))
            else:
                self.shortcuts.append(nn.Identity())
        
        self.output_layer = nn.Linear(hidden_sizes[-1], 1)
        nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity='linear')
        self.relu = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.batch_norm_input(x)
        x = self.relu(x)
        
        for i in range(len(self.hidden_layers)):
            identity = x
            x = self.hidden_layers[i](x)
            x = self.batch_norms[i](x)
            x = self.relu(x)
            x = self.dropouts[i](x)
            x = x + self.shortcuts[i](identity)
        
        return self.output_layer(x)

def load_data(folder_path):
    all_json_data = []
    all_targets = []
    
    for program_folder in os.listdir(folder_path):
        program_path = os.path.join(folder_path, program_folder)
        if not os.path.isdir(program_path):
            continue
        for schedule_file in os.listdir(program_path):
            if not schedule_file.endswith('.json'):
                continue
            file_path = os.path.join(program_path, schedule_file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                target = extract_target(data)
                if target is not None and target > 0 and not np.isnan(target) and not np.isinf(target):
                    all_json_data.append(data)
                    all_targets.append(target)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    if all_targets:
        q1, q3 = np.percentile(all_targets, [1, 99])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered_json_data = []
        filtered_targets = []
        for data, target in zip(all_json_data, all_targets):
            if lower_bound <= target <= upper_bound:
                filtered_json_data.append(data)
                filtered_targets.append(target)
        print(f"Removed {len(all_targets) - len(filtered_targets)} outliers from {len(all_targets)} samples")
        return filtered_json_data, filtered_targets
    return all_json_data, all_targets

def train_model(model, train_loader, val_loader, criterion, num_epochs=100, patience=15):
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=num_epochs, steps_per_epoch=len(train_loader),
        pct_start=0.3, anneal_strategy='cos'
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    model_saved = False
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(X_batch)
                    loss = criterion(outputs.squeeze(), y_batch)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                if not torch.isnan(loss) and not torch.isinf(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
            epoch_train_loss += loss.item() if not torch.isnan(loss) else 0
            scheduler.step()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                epoch_val_loss += loss.item() if not torch.isnan(loss) else 0
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if not np.isnan(avg_val_loss) and not np.isinf(avg_val_loss) and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            print(f"Saving best model with validation loss: {best_val_loss:.6f}")
            torch.save(model.state_dict(), f'best_model_{id(model)}.pth')
            model_saved = True
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    if model_saved:
        model.load_state_dict(torch.load(f'best_model_{id(model)}.pth'))
    else:
        print("No valid model saved due to NaN or inf losses. Using last model state.")
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'training_curve_{id(model)}.png')
    plt.close()
    
    return model, train_losses, val_losses

def evaluate_model(model, test_dataset, target_scaler=None, num_samples=None):
    model.eval()
    if num_samples is None or num_samples > len(test_dataset):
        num_samples = len(test_dataset)
    
    test_loader = DataLoader(test_dataset, batch_size=num_samples, shuffle=False)
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            norm_preds = outputs.cpu().numpy().reshape(-1, 1)
            norm_trues = y_batch.cpu().numpy().reshape(-1, 1)
            predictions = target_scaler.inverse_transform(norm_preds).flatten()
            true_values = target_scaler.inverse_transform(norm_trues).flatten()
            break
    
    abs_errors = np.abs(predictions - true_values)
    rel_errors = abs_errors / np.maximum(true_values, 1e-10) * 100
    
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    mape = np.mean(rel_errors)
    r2 = r2_score(true_values, predictions)
    
    metrics_by_range = {}
    ranges = [(0, 100, "Short (<100ms)"), (100, 1000, "Medium (100-1000ms)"), (1000, float('inf'), "Long (>1000ms)")]
    for min_val, max_val, label in ranges:
        mask = (true_values >= min_val) & (true_values < max_val)
        if np.sum(mask) > 0:
            range_true = true_values[mask]
            range_pred = predictions[mask]
            range_abs_err = abs_errors[mask]
            range_rel_err = rel_errors[mask]
            metrics_by_range[label] = {
                'count': np.sum(mask),
                'mae': np.mean(range_abs_err),
                'mape': np.mean(range_rel_err),
                'rmse': np.sqrt(mean_squared_error(range_true, range_pred)),
                'r2': r2_score(range_true, range_pred) if len(range_true) > 1 else float('nan')
            }
    
    sample_results = [{'predicted': predictions[i], 'true': true_values[i], 'error_pct': rel_errors[i], 'abs_error': abs_errors[i]} 
                      for i in range(min(10, num_samples))]
    
    metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}
    return sample_results, metrics, metrics_by_range, predictions, true_values

def analyze_feature_importance(model, feature_names):
    weights = model.input_layer.weight.data.abs().mean(dim=0).cpu().numpy()
    importance = {name: weight for name, weight in zip(feature_names, weights)}
    return sorted(importance.items(), key=lambda x: x[1], reverse=True)

class ExecutionTimeEnsemble:
    def __init__(self, models, feature_scaler, target_scaler):
        self.models = models
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
    
    def predict(self, features):
        if isinstance(features, np.ndarray):
            features = self.feature_scaler.transform(features)
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            predictions = [model(features).cpu().numpy() for model in self.models]
        avg_pred = np.mean(predictions, axis=0)
        return self.target_scaler.inverse_transform(avg_pred)

def main():
    batch_size = 64
    num_epochs = 300
    
    folder_path = 'synthetic_data'
    print("Loading data...")
    all_json_data, all_targets = load_data(folder_path)
    
    if not all_json_data:
        print("No valid data found.")
        return
    
    print(f"Loaded {len(all_json_data)} samples.")
    
    plt.figure(figsize=(10, 5))
    plt.hist(all_targets, bins=50)
    plt.xlabel('Execution Time (ms)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Execution Times')
    plt.savefig('target_distribution.png')
    plt.close()
    
    plt.figure(figsize=(10, 5))
    plt.hist(np.log1p(all_targets), bins=50)
    plt.xlabel('Log(Execution Time + 1) (ms)')
    plt.ylabel('Frequency')
    plt.title('Log Distribution of Execution Times')
    plt.savefig('log_target_distribution.png')
    plt.close()
    
    time_ranges = [0, 100, 1000, 10000, float('inf')]
    range_labels = ['very_short', 'short', 'medium', 'long']
    data_ranges = [next(label for label, upper in zip(range_labels, time_ranges[1:]) if target < upper)
                   for target in all_targets]
    
    train_data, test_data, train_targets, test_targets, train_ranges, test_ranges = train_test_split(
        all_json_data, all_targets, data_ranges, test_size=0.2, random_state=42, stratify=data_ranges
    )
    train_data, val_data, train_targets, val_targets, train_ranges, val_ranges = train_test_split(
        train_data, train_targets, train_ranges, test_size=0.25, random_state=42, stratify=train_ranges
    )
    
    print(f"Train set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    print("Creating datasets...")
    train_dataset = EnhancedScheduleDataset(train_data, train_targets, train=True, k_best_features=150)
    feature_scaler = train_dataset.feature_scaler
    target_scaler = train_dataset.target_scaler
    feature_selector = train_dataset.feature_selector
    
    val_dataset = EnhancedScheduleDataset(val_data, val_targets, feature_scaler=feature_scaler, 
                                          target_scaler=target_scaler, feature_selector=feature_selector, train=False)
    test_dataset = EnhancedScheduleDataset(test_data, test_targets, feature_scaler=feature_scaler, 
                                           target_scaler=target_scaler, feature_selector=feature_selector, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    input_size = train_dataset.X.shape[1]
    print(f"Input dimension after feature selection: {input_size}")
    
    models = []
    for i in range(3):
        hidden_sizes_options = [[512, 256, 128, 64], [384, 256, 128, 64], [512, 384, 256, 128, 64]]
        dropout_options = [0.3, 0.4, 0.35]
        
        model = ImprovedExecutionTimePredictor(input_size, hidden_sizes=hidden_sizes_options[i], 
                                               dropout=dropout_options[i]).to(device)
        print(f"Model {i+1} has {sum(p.numel() for p in model.parameters()):,} parameters")
        
        criterion_options = [nn.MSELoss(), nn.SmoothL1Loss(beta=0.5), 
                             lambda pred, target: torch.mean((1 + torch.log1p(torch.abs(pred - target))) * torch.square(pred - target))]
        criterion = criterion_options[i]
        
        print(f"\nTraining model {i+1}...")
        model, _, _ = train_model(model, train_loader, val_loader, criterion, num_epochs=num_epochs, patience=20)
        models.append(model)
    
    ensemble = ExecutionTimeEnsemble(models, feature_scaler, target_scaler)
    
    print("\nEvaluating ensemble on test set...")
    sample_results, metrics, metrics_by_range, predictions, true_values = evaluate_model(
        models[0], test_dataset, target_scaler=target_scaler)
    
    ensemble_preds = ensemble.predict(test_dataset.X)
    ensemble_mae = mean_absolute_error(true_values, ensemble_preds)
    ensemble_rmse = np.sqrt(mean_squared_error(true_values, ensemble_preds))
    ensemble_r2 = r2_score(true_values, ensemble_preds)
    
    print("\nSample Predictions (First Model):")
    for result in sample_results:
        print(f"Predicted: {result['predicted']:.2f}ms, True: {result['true']:.2f}ms, "
              f"Error: {result['abs_error']:.2f}ms ({result['error_pct']:.2f}%)")
    
    print("\nOverall Metrics (First Model):")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nMetrics by Execution Time Range (First Model):")
    for range_name, range_metrics in metrics_by_range.items():
        print(f"{range_name} ({range_metrics['count']} samples):")
        for m, v in range_metrics.items():
            if m != 'count':
                print(f"  {m.upper()}: {v:.4f}")
    
    print("\nEnsemble Metrics:")
    print(f"MAE: {ensemble_mae:.4f}")
    print(f"RMSE: {ensemble_rmse:.4f}")
    print(f"R2: {ensemble_r2:.4f}")
    
    print("\nTop 10 Feature Importances (First Model):")
    importance = analyze_feature_importance(models[0], train_dataset.selected_feature_names)
    for name, weight in importance[:10]:
        print(f"{name}: {weight:.6f}")
    
    plt.figure(figsize=(10, 5))
    plt.scatter(true_values, predictions, alpha=0.5, label='First Model')
    plt.scatter(true_values, ensemble_preds, alpha=0.5, label='Ensemble')
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--')
    plt.xlabel('True Execution Time (ms)')
    plt.ylabel('Predicted Execution Time (ms)')
    plt.title('Predicted vs True Execution Times')
    plt.legend()
    plt.savefig('prediction_scatter.png')
    plt.close()

if __name__ == "__main__":
    main()
