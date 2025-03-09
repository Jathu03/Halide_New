import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import random

def get_execution_time(schedule_data):
    if "execution_times" in schedule_data:
        exec_times = schedule_data["execution_times"]
        filtered_times = [t for t in exec_times if t > 0]
        return float(np.median(filtered_times)) if filtered_times else None
    print("Warning: No valid execution times found in schedule")
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
        
        loop_ranges = []
        for it in iterators.values():
            lower = it.get("lower_bound")
            upper = it.get("upper_bound")
            if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                loop_ranges.append(upper - lower)
        
        base_features = {
            'memory_size': prog_annot.get("memory_size", 0),
            'iterator_count': len(iterators),
            'max_depth_iterators': max((len(it.get("child_iterators", [])) for it in iterators.values()), default=0),
            'computation_count': len(computations),
            'reduction_count': sum(1 for comp in computations.values() if comp.get("comp_is_reduction", False)),
            'access_count': sum(len(comp.get("accesses", [])) for comp in computations.values()),
            'avg_loop_range': float(np.mean(loop_ranges)) if loop_ranges else 0,
            'memory_per_computation': prog_annot.get("memory_size", 0) / max(len(computations), 1),
            'loop_range_std': float(np.std(loop_ranges)) if loop_ranges else 0,
            'access_per_iterator': sum(len(comp.get("accesses", [])) for comp in computations.values()) / max(len(iterators), 1)
        }
        base_features['avg_access_per_comp'] = base_features['access_count'] / max(base_features['computation_count'], 1)
        
        schedules = func_data["schedules_list"]
        for idx, schedule in enumerate(schedules):
            execution_time = get_execution_time(schedule)
            if execution_time is None or execution_time <= 0:
                continue
            
            features = base_features.copy()
            features['execution_time'] = execution_time
            features['log_execution_time'] = np.log1p(execution_time)
            
            tiling_factors, unroll_factors, parallel_factors = [], [], []
            comp_features = []
            
            for comp_key, comp_data in schedule.items():
                if isinstance(comp_data, dict):
                    unroll_factor = comp_data.get("unrolling_factor")
                    comp_dict = {
                        'tiling': 1 if comp_data.get("tiling", {}) else 0,
                        'unroll': 1 if (unroll_factor is not None and isinstance(unroll_factor, (int, float)) and unroll_factor > 0) else 0,
                        'parallel': 1 if comp_data.get("parallelized_dim", "") else 0,
                        'transform_count': len(comp_data.get("transformations_list", []))
                    }
                    if "tiling" in comp_data and comp_data["tiling"]:
                        tiling_factors.extend(comp_data["tiling"].get("tiling_factors", []))
                    if unroll_factor is not None and isinstance(unroll_factor, (int, float)):
                        unroll_factors.append(unroll_factor)
                    if "parallelized_dim" in comp_data:
                        parallel_factors.append(1)
                    comp_features.append(comp_dict)
            
            features['seq_tiling'] = [cf['tiling'] for cf in comp_features]
            features['seq_unroll'] = [cf['unroll'] for cf in comp_features]
            features['seq_parallel'] = [cf['parallel'] for cf in comp_features]
            features['seq_transform_count'] = [cf['transform_count'] for cf in comp_features]
            
            features['tiling_count'] = sum(1 for comp in schedule.values() if isinstance(comp, dict) and comp.get("tiling", {}))
            features['unroll_count'] = len(unroll_factors)
            features['parallel_count'] = len(parallel_factors)
            features['total_transformation_count'] = sum(
                len(comp.get("transformations_list", [])) for comp in schedule.values() if isinstance(comp, dict)
            )
            features['avg_tiling_factor'] = float(np.mean(tiling_factors)) if tiling_factors else 0
            features['avg_unroll_factor'] = float(np.mean(unroll_factors)) if unroll_factors else 0
            features['tiling_depth'] = max(
                (comp["tiling"]["tiling_depth"] for comp in schedule.values() 
                 if isinstance(comp, dict) and comp.get("tiling", {}).get("tiling_depth")), default=0
            )
            
            if "tree_structure" in schedule and "roots" in schedule["tree_structure"]:
                roots = schedule["tree_structure"]["roots"]
                features['root_count'] = len(roots)
                features['max_tree_depth'] = max(
                    (1 + max((len(child.get("child_list", [])) for child in root.get("child_list", [])), default=0) 
                     for root in roots), default=0
                )
            
            features['memory_access_ratio'] = features['memory_size'] * features['access_count']
            features['comp_transform_interaction'] = features['computation_count'] * features['total_transformation_count']
            features['tiling_parallel_product'] = features['tiling_count'] * features['parallel_count']
            
            all_features.append(features)
    
    return all_features if all_features else None

def process_directory(directory_path):
    all_features, file_names = [], []
    
    json_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.json')])
    if not json_files:
        print(f"Error: No JSON files found in {directory_path}")
        return None, None, None
    
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

def clean_and_transform_features(train_features, test_features):
    all_features_df = pd.DataFrame(train_features + test_features)
    all_features_df = all_features_df.fillna(0)
    
    constant_columns = [col for col in all_features_df.columns 
                        if col not in ['execution_time', 'log_execution_time', 'seq_tiling', 'seq_unroll', 'seq_parallel', 'seq_transform_count']
                        and all_features_df[col].nunique() == 1]
    all_features_df = all_features_df.drop(columns=constant_columns)
    print(f"Dropped {len(constant_columns)} constant columns")
    
    for col in all_features_df.columns:
        if col not in ['execution_time', 'log_execution_time', 'seq_tiling', 'seq_unroll', 'seq_parallel', 'seq_transform_count']:
            if all_features_df[col].min() >= 0 and all_features_df[col].max() > 0:
                all_features_df[f'{col}_log'] = np.log1p(all_features_df[col])
    
    train_size = len(train_features)
    train_df = all_features_df.iloc[:train_size]
    test_df = all_features_df.iloc[train_size:]
    
    return train_df, test_df

def prepare_data_for_model(train_features, test_features):
    train_df, test_df = clean_and_transform_features(train_features, test_features)
    
    # Separate sequence features and scalar features
    seq_cols = ['seq_tiling', 'seq_unroll', 'seq_parallel', 'seq_transform_count']
    scalar_cols = [col for col in train_df.columns if col not in seq_cols + ['execution_time', 'log_execution_time']]
    
    # Pad sequence features to max length
    max_seq_len = max(
        max(train_df[col].apply(len).max(), test_df[col].apply(len).max()) for col in seq_cols
    )
    
    def pad_sequence(seq, max_len):
        padded = np.zeros(max_len)
        padded[:len(seq)] = seq[:max_len]
        return padded
    
    X_train_seq = np.stack([
        np.stack([pad_sequence(row[col], max_seq_len) for col in seq_cols], axis=1)
        for _, row in train_df.iterrows()
    ])
    X_test_seq = np.stack([
        np.stack([pad_sequence(row[col], max_seq_len) for col in seq_cols], axis=1)
        for _, row in test_df.iterrows()
    ])
    
    # Scale scalar features
    scaler_X = RobustScaler()
    X_train_scalar = scaler_X.fit_transform(train_df[scalar_cols])
    X_test_scalar = scaler_X.transform(test_df[scalar_cols])
    
    # Broadcast scalar features to match sequence length
    X_train_scalar_expanded = np.tile(X_train_scalar[:, np.newaxis, :], (1, max_seq_len, 1))
    X_test_scalar_expanded = np.tile(X_test_scalar[:, np.newaxis, :], (1, max_seq_len, 1))
    
    # Combine sequence and scalar features
    X_train = np.concatenate([X_train_seq, X_train_scalar_expanded], axis=2)
    X_test = np.concatenate([X_test_seq, X_test_scalar_expanded], axis=2)
    
    y_train = train_df['log_execution_time'].values.reshape(-1, 1)
    y_test = test_df['log_execution_time'].values.reshape(-1, 1)
    
    scaler_y = RobustScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Input sequence length: {max_seq_len}, Feature dimension: {X_train.shape[2]}")
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_y, max_seq_len, X_train.shape[2]

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=128, n_heads=4, n_layers=2, dropout=0.3):
        super(TransformerPredictor, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=512, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        self.fc1 = nn.Linear(d_model, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc_out = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_proj(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling over sequence
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.fc_out(x)
        return x

def custom_loss(y_pred, y_true, alpha=0.7):
    epsilon = 1e-8
    rel_error = torch.abs((y_pred - y_true) / (y_true.abs() + epsilon))
    mse_loss = nn.MSELoss()(y_pred, y_true)
    return alpha * torch.mean(rel_error) + (1 - alpha) * mse_loss

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=1000, patience=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch+1}")
                return None, None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(test_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
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
    return train_losses, val_losses

def evaluate_model(model, X_test, y_test, y_scaler, file_names_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    X_test = X_test.to(device)
    with torch.no_grad():
        y_pred_scaled = model(X_test)
    
    y_pred_scaled = y_pred_scaled.cpu().numpy()
    y_test = y_test.cpu().numpy()
    
    y_test_transformed = y_scaler.inverse_transform(y_test)
    y_pred_transformed = y_scaler.inverse_transform(y_pred_scaled)
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
    
    X_train, y_train, X_test, y_test, y_scaler, seq_len, input_dim = prepare_data_for_model(train_features, test_features)
    
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32)
    
    model = TransformerPredictor(
        input_dim=input_dim,
        seq_len=seq_len,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.3
    )
    
    criterion = custom_loss
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print("Building and training Transformer model...")
    train_losses, val_losses = train_model(model, train_loader, test_loader, criterion, optimizer)
    
    if train_losses is None or val_losses is None:
        print("Training failed due to NaN losses")
        return None
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(model, X_test, y_test, y_scaler, test_file_names)
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "Tiramisu"
    result = main(main_dir)
    if result is not None:
        model, y_scaler, y_test_actual, y_pred_actual = result
        print("\nTransformer model training and prediction completed!")
        torch.save(model.state_dict(), 'transformer_model.pth')
    else:
        print("\nModel training failed!")
