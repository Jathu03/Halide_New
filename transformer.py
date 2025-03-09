import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import CyclicLR
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

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
        
        loop_ranges = []
        valid_skew_values = []
        for it in iterators.values():
            lower = it.get("lower_bound")
            upper = it.get("upper_bound")
            try:
                lower = float(lower) if lower is not None and str(lower).replace('.', '').replace('-', '').isdigit() else 0.0
                upper = float(upper) if upper is not None and str(upper).replace('.', '').replace('-', '').isdigit() else 0.0
                if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                    range_val = upper - lower
                    loop_ranges.append(range_val)
                    valid_skew_values.append(range_val ** 3)
            except (ValueError, TypeError):
                continue
        
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
            'access_per_iterator': sum(len(comp.get("accesses", [])) for comp in computations.values()) / max(len(iterators), 1),
            'loop_range_skew': float(np.mean(valid_skew_values)) if valid_skew_values else 0
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
            
            tiling_factors = []
            unroll_factors = []
            parallel_factors = []
            
            for comp_key, comp_data in schedule.items():
                if isinstance(comp_data, dict):
                    if "tiling" in comp_data and comp_data["tiling"]:
                        tiling_factors.extend(comp_data["tiling"].get("tiling_factors", []))
                    if "unrolling_factor" in comp_data:
                        unroll_factor = comp_data["unrolling_factor"]
                        if unroll_factor is not None and isinstance(unroll_factor, (int, float)):
                            unroll_factors.append(unroll_factor)
                    if "parallelized_dim" in comp_data:
                        parallel_factors.append(1)
                    
                    features[f'{comp_key}_transformation_count'] = len(comp_data.get("transformations_list", []))
                    features[f'{comp_key}_tiling'] = 1 if comp_data.get("tiling", {}) else 0
                    unroll_val = comp_data.get("unrolling_factor")
                    features[f'{comp_key}_unrolled'] = 1 if (unroll_val is not None and isinstance(unroll_val, (int, float)) and unroll_val > 0) else 0
                    features[f'{comp_key}_parallelized'] = 1 if comp_data.get("parallelized_dim", "") else 0
            
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
                features['avg_children_per_root'] = np.mean(
                    [len(root.get("child_list", [])) for root in roots]
                ) if roots else 0
            
            features['comp_depth_interaction'] = features['computation_count'] * features['max_tree_depth']
            features['tiling_parallel_interaction'] = features['tiling_count'] * features['parallel_count']
            features['memory_per_access'] = features['memory_size'] / max(features['access_count'], 1)
            features['transformations_per_comp'] = features['total_transformation_count'] / max(features['computation_count'], 1)
            exec_times = [f['execution_time'] for f in all_features if f['execution_time'] > 0]
            mean_exec = np.mean(exec_times) if exec_times else 0
            std_exec = np.std(exec_times) if exec_times else 1
            features['execution_time_norm'] = (execution_time - mean_exec) / max(std_exec, 1e-8) if std_exec > 0 else 0
            
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

def clean_and_transform_features(train_features, test_features):
    all_features_df = pd.DataFrame(train_features + test_features)
    all_features_df = all_features_df.fillna(0)
    
    constant_columns = [col for col in all_features_df.columns 
                        if col not in ['execution_time', 'log_execution_time'] and all_features_df[col].nunique() == 1]
    all_features_df = all_features_df.drop(columns=constant_columns)
    print(f"Dropped {len(constant_columns)} constant columns")
    
    for col in all_features_df.columns:
        if col not in ['execution_time', 'log_execution_time'] and all_features_df[col].min() >= 0 and all_features_df[col].max() > 0:
            all_features_df[f'{col}_log'] = np.log1p(all_features_df[col].replace([np.inf, -np.inf], 0))
    
    numeric_cols = all_features_df.select_dtypes(include=['number']).columns
    all_features_df[numeric_cols] = all_features_df[numeric_cols].replace([np.inf, -np.inf], np.finfo(np.float64).max)
    
    scaler = RobustScaler()
    all_features_df[numeric_cols] = scaler.fit_transform(all_features_df[numeric_cols])
    
    vt = VarianceThreshold(threshold=0.01)
    transformed_data = vt.fit_transform(all_features_df[numeric_cols])
    selected_feature_indices = vt.get_support(indices=True)
    selected_numeric_cols = numeric_cols[selected_feature_indices]
    all_features_df = pd.DataFrame(transformed_data, columns=selected_numeric_cols, index=all_features_df.index)
    
    all_features_df = all_features_df.loc[:, ~all_features_df.columns.duplicated()]
    
    train_size = len(train_features)
    train_df = all_features_df.iloc[:train_size]
    test_df = all_features_df.iloc[train_size:]
    
    return train_df, test_df

def prepare_data_for_model(train_features, test_features):
    train_df, test_df = clean_and_transform_features(train_features, test_features)
    
    y_train = train_df['log_execution_time'].values.reshape(-1, 1)
    y_test = test_df['log_execution_time'].values.reshape(-1, 1)
    X_train_df = train_df.drop(['execution_time', 'log_execution_time'], axis=1)
    X_test_df = test_df.drop(['execution_time', 'log_execution_time'], axis=1)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train_df)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test_df)
    y_test_scaled = scaler_y.transform(y_test)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Input feature dimension: {X_train_scaled.shape[1]}")
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_y, X_train_scaled.shape[1]

class EnhancedLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], output_size=1, dropout_rate=0.5):
        super(EnhancedLSTMModel, self).__init__()
        
        # Increased num_layers to 2 to allow dropout
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.ln1 = nn.LayerNorm(hidden_sizes[0] * 2)
        self.lstm2 = nn.LSTM(hidden_sizes[0]*2, hidden_sizes[1], num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.ln2 = nn.LayerNorm(hidden_sizes[1] * 2)
        self.lstm3 = nn.LSTM(hidden_sizes[1]*2, hidden_sizes[2], num_layers=2, batch_first=True, dropout=0.2)
        self.ln3 = nn.LayerNorm(hidden_sizes[2])
        self.attention = nn.Linear(hidden_sizes[2], 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(hidden_sizes[2], hidden_sizes[2]//2)
        self.bn1 = nn.BatchNorm1d(hidden_sizes[2]//2)
        self.fc2 = nn.Linear(hidden_sizes[2]//2, output_size)
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def attention_net(self, lstm_output):
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context = torch.bmm(attn_weights.transpose(1, 2), lstm_output).squeeze(1)
        return context
    
    def forward(self, x):
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.ln1(lstm_out)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out = self.ln2(lstm_out)
        lstm_out, _ = self.lstm3(lstm_out)
        lstm_out = self.ln3(lstm_out)
        attn_out = self.attention_net(lstm_out)
        out = self.dropout(attn_out)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.fc2(out)
        return out

def custom_loss(y_pred, y_true):
    epsilon = 1e-8
    rel_error = torch.abs((y_pred - y_true) / (y_true.abs() + epsilon))
    return torch.mean(rel_error) + 0.5 * nn.MSELoss()(y_pred, y_true)

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=64, val_split=0.1):
    dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    n_samples = len(dataset)
    indices = list(range(n_samples))
    val_size = int(np.floor(val_split * n_samples))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs=500, patience=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Set cycle_momentum=False to work with AdamW
    scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, step_size_up=10, mode='triangular', cycle_momentum=False)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    learning_rates = []
    
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
                return None, None, None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.sampler.indices)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.sampler.indices)
        val_losses.append(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}')
        
        if val_loss < best_val_loss and not np.isnan(val_loss):
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            torch.save(model.state_dict(), 'best_model.pth')
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

def plot_metrics(train_losses, val_losses, learning_rates, y_test_actual, y_pred_actual):
    epochs = range(1, len(train_losses) + 1)
    
    window_size = 5
    train_losses_smooth = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
    val_losses_smooth = np.convolve(val_losses, np.ones(window_size)/window_size, mode='valid')
    epochs_smooth = epochs[:len(train_losses_smooth)]
    
    plt.figure(figsize=(12, 7))
    plt.plot(epochs_smooth, train_losses_smooth, label='Training Loss (Smoothed)', color='blue', linewidth=2)
    plt.plot(epochs_smooth, val_losses_smooth, label='Validation Loss (Smoothed)', color='orange', linewidth=2)
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', alpha=0.3)
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange', alpha=0.3)
    plt.axvline(x=epochs[np.argmin(val_losses)], color='red', linestyle='--', label='Best Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.close()
    print("Loss plot saved as 'loss_plot.png'")
    
    plt.figure(figsize=(12, 7))
    plt.plot(epochs, learning_rates, label='Learning Rate', color='green', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Over Epochs')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig('lr_plot.png')
    plt.close()
    print("Learning rate plot saved as 'lr_plot.png'")
    
    if y_test_actual is not None and y_pred_actual is not None:
        residuals = y_test_actual - y_pred_actual
        plt.figure(figsize=(12, 7))
        plt.scatter(range(len(residuals)), residuals, color='purple', alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Test Sample Index')
        plt.ylabel('Residual (Actual - Predicted)')
        plt.title('Residual Plot for Prediction Errors')
        plt.grid(True)
        plt.savefig('residual_plot.png')
        plt.close()
        print("Residual plot saved as 'residual_plot.png'")

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
    error_percentages = abs(y_test_actual - y_pred_actual) / y_test_actual * 100
    
    print("\nOverall Model Performance:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"MAPE: {mape:.2f}%")
    
    residuals = y_test_actual - y_pred_actual
    print(f"\nDiagnostics:")
    print(f"Mean Residual: {np.mean(residuals):.6f}")
    print(f"Std of Residuals: {np.std(residuals):.6f}")
    print(f"Min Error %: {np.min(error_percentages):.2f}%")
    print(f"Max Error %: {np.max(error_percentages):.2f}%")
    print(f"Median Error %: {np.median(error_percentages):.2f}%")
    
    return y_test_actual, y_pred_actual

def k_fold_cross_validate(model_class, X_train, y_train, k=3, batch_size=64, num_epochs=500, patience=100):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_losses = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        print(f'\nFold {fold + 1}/{k}')
        
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, sampler=val_sampler)
        
        model = model_class(input_size=X_train.shape[2], hidden_sizes=[512, 256, 128], output_size=1, dropout_rate=0.5)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = custom_loss
        
        _, val_losses, _ = train_model(model, train_loader, val_loader, None, criterion, optimizer, num_epochs, patience)
        fold_losses.append(min(val_losses) if val_losses else float('inf'))
    
    print(f'\nCross-validation results: Mean Val Loss = {np.mean(fold_losses):.6f}, Std Val Loss = {np.std(fold_losses):.6f}')
    return np.mean(fold_losses)

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
    
    X_train, y_train, X_test, y_test, y_scaler, input_size = prepare_data_for_model(train_features, test_features)
    
    print("\nPerforming 5-fold cross-validation...")
    k_fold_cross_validate(EnhancedLSTMModel, X_train, y_train)
    
    train_loader, val_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=64, val_split=0.25)
    
    model = EnhancedLSTMModel(input_size=input_size, hidden_sizes=[512, 256, 128], output_size=1, dropout_rate=0.5)
    
    criterion = custom_loss
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print("Building and training Enhanced LSTM model...")
    train_losses, val_losses, learning_rates = train_model(model, train_loader, val_loader, test_loader, criterion, optimizer)
    
    if train_losses is None or val_losses is None or learning_rates is None:
        print("Training failed due to NaN losses")
        return None
    
    plot_metrics(train_losses, val_losses, learning_rates, None, None)
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(model, X_test, y_test, y_scaler, test_file_names)
    plot_metrics(train_losses, val_losses, learning_rates, y_test_actual, y_pred_actual)
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "Tiramisu"
    result = main(main_dir)
    if result is not None:
        model, y_scaler, y_test_actual, y_pred_actual = result
        print("\nEnhanced model training and prediction completed!")
    else:
        print("\nModel training failed!")
