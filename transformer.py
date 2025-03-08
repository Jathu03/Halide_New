import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random

def get_execution_time(schedule_data):
    if "execution_times" in schedule_data:
        exec_times = schedule_data["execution_times"]
        # Filter out outliers using IQR method
        if len(exec_times) > 3:
            q1, q3 = np.percentile(exec_times, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_times = [t for t in exec_times if lower_bound <= t <= upper_bound]
            if filtered_times:
                return float(np.median(filtered_times))
        return float(np.median(exec_times))
    print("Warning: No execution times found in schedule")
    return None

def extract_advanced_features(iterators, computations):
    """Extract more advanced features from program structure"""
    features = {}
    
    if iterators:
        max_depth = 0
        iterator_nesting = {}
        for it_id, it in iterators.items():
            parent = it.get("parent_iterator")
            depth = 0
            current = parent
            while current:
                depth += 1
                current = iterators.get(current, {}).get("parent_iterator")
            iterator_nesting[it_id] = depth
            max_depth = max(max_depth, depth)
        
        features['max_iterator_nesting'] = max_depth
        features['avg_iterator_nesting'] = sum(iterator_nesting.values()) / len(iterator_nesting)
    
    if computations:
        access_patterns = []
        reduction_depths = []
        for comp_id, comp in computations.items():
            accesses = comp.get("accesses", [])
            access_patterns.extend([len(access.get("access_pattern", [])) for access in accesses])
            if comp.get("comp_is_reduction", False):
                parent_it = comp.get("iterator_name")
                depth = 0
                current = parent_it
                while current and current in iterators:
                    depth += 1
                    current = iterators.get(current, {}).get("parent_iterator")
                reduction_depths.append(depth)
        
        features['max_access_pattern_length'] = max(access_patterns) if access_patterns else 0
        features['avg_access_pattern_length'] = sum(access_patterns) / len(access_patterns) if access_patterns else 0
        features['max_reduction_depth'] = max(reduction_depths) if reduction_depths else 0
    
    return features

def extract_schedule_features(schedule):
    """Extract detailed features specific to a schedule"""
    features = {}
    
    transformation_types = {}
    for comp_key, comp_data in schedule.items():
        if isinstance(comp_data, dict) and "transformations_list" in comp_data:
            for transform in comp_data["transformations_list"]:
                if isinstance(transform, dict) and "type" in transform:
                    t_type = transform["type"]
                    transformation_types[t_type] = transformation_types.get(t_type, 0) + 1
    
    for t_type, count in transformation_types.items():
        features[f'transform_{t_type}_count'] = count
    
    if any(isinstance(comp, dict) and comp.get("tiling") for comp in schedule.values()):
        tiling_factors = []
        for comp_key, comp_data in schedule.items():
            if isinstance(comp_data, dict) and "tiling" in comp_data and comp_data["tiling"]:
                factors = comp_data["tiling"].get("tiling_factors", [])
                tiling_factors.extend(factors)
                for i, factor in enumerate(factors[:3]):
                    features[f'{comp_key}_tiling_factor_{i}'] = factor
        
        features['min_tiling_factor'] = min(tiling_factors) if tiling_factors else 0
        features['max_tiling_factor'] = max(tiling_factors) if tiling_factors else 0
        features['tiling_factor_product'] = np.prod(tiling_factors) if tiling_factors else 0
    
    if "tree_structure" in schedule and "roots" in schedule["tree_structure"]:
        roots = schedule["tree_structure"]["roots"]
        tree_depths = []
        node_counts = []
        
        for root in roots:
            depth = 1
            nodes = 1
            if "child_list" in root:
                queue = [(child, 2) for child in root.get("child_list", [])]
                while queue:
                    node, level = queue.pop(0)
                    depth = max(depth, level)
                    nodes += 1
                    if "child_list" in node:
                        queue.extend([(child, level + 1) for child in node.get("child_list", [])])
            
            tree_depths.append(depth)
            node_counts.append(nodes)
        
        features['max_tree_depth'] = max(tree_depths)
        features['total_tree_nodes'] = sum(node_counts)
        features['tree_complexity'] = sum(tree_depths) / len(tree_depths) if tree_depths else 0
    
    return features

def extract_features_from_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None
    
    all_features = []
    function_metadata = {}
    
    for func_id, func_data in data.items():
        if "program_annotation" not in func_data or "schedules_list" not in func_data:
            print(f"Warning: Missing required fields in {file_path} for {func_id}")
            continue
        
        prog_annot = func_data["program_annotation"]
        iterators = prog_annot.get("iterators", {})
        computations = prog_annot.get("computations", {})
        
        loop_ranges = []
        max_loop_range = 0
        for it in iterators.values():
            lower = it.get("lower_bound")
            upper = it.get("upper_bound")
            if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                loop_range = upper - lower
                loop_ranges.append(loop_range)
                max_loop_range = max(max_loop_range, loop_range)
        
        base_features = {
            'memory_size': prog_annot.get("memory_size", 0),
            'iterator_count': len(iterators),
            'max_depth_iterators': max(
                (len(it.get("child_iterators", [])) for it in iterators.values()), default=0
            ),
            'computation_count': len(computations),
            'reduction_count': sum(1 for comp in computations.values() if comp.get("comp_is_reduction", False)),
            'access_count': sum(len(comp.get("accesses", [])) for comp in computations.values()),
            'avg_loop_range': float(np.mean(loop_ranges)) if loop_ranges else 0,
            'max_loop_range': max_loop_range,
            'loop_range_product': float(np.prod(loop_ranges)) if loop_ranges else 0
        }
        
        advanced_features = extract_advanced_features(iterators, computations)
        base_features.update(advanced_features)
        
        base_features['avg_access_per_comp'] = base_features['access_count'] / base_features['computation_count'] if base_features['computation_count'] > 0 else 0
        
        function_metadata[func_id] = {
            'memory_size': base_features['memory_size'],
            'iterator_count': base_features['iterator_count'],
            'computation_count': base_features['computation_count']
        }
        
        schedules = func_data["schedules_list"]
        for idx, schedule in enumerate(schedules):
            execution_time = get_execution_time(schedule)
            if execution_time is None:
                continue
            
            features = base_features.copy()
            features['execution_time'] = execution_time
            features['schedule_id'] = idx
            features['func_id'] = func_id
            
            schedule_features = extract_schedule_features(schedule)
            features.update(schedule_features)
            
            tiling_factors = []
            total_transformations = 0
            for comp_key, comp_data in schedule.items():
                if isinstance(comp_data, dict):
                    if "tiling" in comp_data and comp_data["tiling"]:
                        tiling_factors.extend(comp_data["tiling"].get("tiling_factors", []))
                    transformations = len(comp_data.get("transformations_list", []))
                    features[f'{comp_key}_transformation_count'] = transformations
                    features[f'{comp_key}_tiling'] = 1 if comp_data.get("tiling", {}) else 0
                    total_transformations += transformations
            
            features['tiling_count'] = sum(1 for comp in schedule.values() if isinstance(comp, dict) and comp.get("tiling", {}))
            features['total_transformation_count'] = total_transformations
            features['avg_tiling_factor'] = float(np.mean(tiling_factors)) if tiling_factors else 0
            features['tiling_depth'] = max(
                (comp["tiling"]["tiling_depth"] for comp in schedule.values() 
                 if isinstance(comp, dict) and comp.get("tiling", {}).get("tiling_depth")), default=0
            )
            
            features['file_path'] = file_path
            features['file_name'] = os.path.basename(file_path)
            
            all_features.append(features)
    
    return all_features, function_metadata

def process_directory(directory_path, test_size=10, random_split=True):
    all_features = []
    file_names = []
    function_metadata = {}
    
    json_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.json')])
    
    if len(json_files) < 2:
        print(f"Error: Expected at least 2 files in {directory_path}, found {len(json_files)}")
        return None, None, None, None
    
    for filename in json_files:
        file_path = os.path.join(directory_path, filename)
        features_list, metadata = extract_features_from_file(file_path)
        if features_list is not None:
            function_metadata.update(metadata)
            all_features.extend(features_list)
            file_names.extend([f"{filename}_schedule_{i}" for i in range(len(features_list))])
    
    if len(all_features) < test_size + 10:
        print(f"Error: Only {len(all_features)} valid schedules found in {directory_path}")
        return None, None, None, None
    
    features_df = pd.DataFrame(all_features)
    
    if random_split:
        unique_funcs = features_df['func_id'].unique()
        train_funcs, test_funcs = train_test_split(unique_funcs, test_size=min(0.2, test_size/len(features_df)))
        
        test_features_df = features_df[features_df['func_id'].isin(test_funcs)]
        train_features_df = features_df[~features_df['func_id'].isin(test_funcs)]
        
        if len(test_features_df) > test_size:
            test_features_df = test_features_df.sample(test_size, random_state=42)
        elif len(test_features_df) < test_size and len(train_features_df) > 0:
            additional_samples = train_features_df.sample(min(test_size - len(test_features_df), len(train_features_df)), random_state=42)
            test_features_df = pd.concat([test_features_df, additional_samples])
            train_features_df = train_features_df.drop(additional_samples.index)
    else:
        train_features_df = features_df.iloc[:-test_size]
        test_features_df = features_df.iloc[-test_size:]
    
    train_features = train_features_df.to_dict('records')
    test_features = test_features_df.to_dict('records')
    test_file_names = [f"{row['file_name']}_func_{row['func_id']}_schedule_{row['schedule_id']}" for _, row in test_features_df.iterrows()]
    
    print(f"Processed {directory_path}: {len(train_features)} training schedules, {len(test_features)} test schedules")
    
    return train_features, test_features, test_file_names, function_metadata

def clean_and_transform_features(train_features, test_features):
    all_features_df = pd.DataFrame(train_features + test_features)
    all_features_df = all_features_df.fillna(0)
    
    all_features_df.columns = [str(col) for col in all_features_df.columns]
    
    meta_columns = ['func_id', 'schedule_id', 'file_path', 'file_name']
    numeric_features_df = all_features_df.drop(columns=[col for col in meta_columns if col in all_features_df.columns])
    
    constant_columns = [col for col in numeric_features_df.columns 
                        if col != 'execution_time' and numeric_features_df[col].nunique() <= 1]
    all_features_df = all_features_df.drop(columns=constant_columns)
    print(f"Dropped {len(constant_columns)} constant columns")
    
    all_features_df['execution_time_log'] = np.log1p(all_features_df['execution_time'])
    
    if 'memory_size' in all_features_df.columns and 'computation_count' in all_features_df.columns:
        all_features_df['memory_per_computation'] = all_features_df['memory_size'] / (all_features_df['computation_count'] + 1e-6)
    
    if 'total_transformation_count' in all_features_df.columns and 'computation_count' in all_features_df.columns:
        all_features_df['transformations_per_comp'] = all_features_df['total_transformation_count'] / (all_features_df['computation_count'] + 1e-6)
    
    numeric_cols = all_features_df.select_dtypes(include=['number']).columns
    model_features_df = all_features_df[numeric_cols]
    
    train_size = len(train_features)
    train_df = model_features_df.iloc[:train_size]
    test_df = model_features_df.iloc[train_size:]
    
    return train_df, test_df, all_features_df.iloc[:train_size], all_features_df.iloc[train_size:]

def prepare_data_for_model(train_features, test_features):
    train_df, test_df, train_meta_df, test_meta_df = clean_and_transform_features(train_features, test_features)
    
    y_train_actual = train_df['execution_time'].values
    y_test_actual = test_df['execution_time'].values
    
    y_train = train_df['execution_time_log'].values.reshape(-1, 1)
    y_test = test_df['execution_time_log'].values.reshape(-1, 1)
    
    X_train_df = train_df.drop(['execution_time', 'execution_time_log'], axis=1, errors='ignore')
    X_test_df = test_df.drop(['execution_time', 'execution_time_log'], axis=1, errors='ignore')
    
    common_cols = set(X_train_df.columns).intersection(set(X_test_df.columns))
    X_train_df = X_train_df[list(common_cols)]
    X_test_df = X_test_df[list(common_cols)]
    
    print(f"Using {len(common_cols)} features for modeling")
    
    scaler_X = RobustScaler()
    scaler_y = PowerTransformer(method='yeo-johnson')
    
    X_train_scaled = scaler_X.fit_transform(X_train_df)
    y_train_scaled = scaler_y.fit_transform(y_train)
    X_test_scaled = scaler_X.transform(X_test_df)
    y_test_scaled = scaler_y.transform(y_test)
    
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    print(f"Input feature dimension: {X_train_scaled.shape[1]}")
    
    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
            X_train_tensor, X_test_tensor, scaler_y, X_train_scaled.shape[1],
            y_train_actual, y_test_actual, train_meta_df, test_meta_df, list(X_train_df.columns))

class HybridAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=1, dropout_rate=0.3):
        super(HybridAttentionModel, self).__init__()
        
        # LSTM layer for sequential analysis
        self.lstm = nn.LSTM(input_size, hidden_sizes[0], batch_first=True, bidirectional=True)
        self.lstm_attention = nn.Linear(hidden_sizes[0] * 2, 1)
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Fully connected layers
        combined_size = hidden_sizes[0] * 2 + 64
        self.fc1 = nn.Linear(combined_size, hidden_sizes[1])
        self.bn1 = nn.BatchNorm1d(hidden_sizes[1])
        self.fc2 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[2])
        self.fc3 = nn.Linear(hidden_sizes[2], output_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.silu = nn.SiLU()
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape input for LSTM if needed
        if x.dim() == 2:
            # Add sequence dimension
            x_lstm = x.unsqueeze(1)
        else:
            x_lstm = x
        
        # LSTM processing
        lstm_out, _ = self.lstm(x_lstm)
        attention_weights = torch.softmax(self.lstm_attention(lstm_out), dim=1)
        lstm_attn_out = torch.sum(attention_weights * lstm_out, dim=1)
        
        # CNN processing - ensure correct input shape
        if x.dim() == 2:
            x_cnn = x.unsqueeze(1)  # Add channel dimension [batch, 1, features]
        else:
            x_cnn = x.view(batch_size, 1, -1)  # Flatten sequence dimension
        
        cnn_out = self.leaky_relu(self.conv1(x_cnn))
        cnn_out = self.leaky_relu(self.conv2(cnn_out))
        cnn_out = self.max_pool(cnn_out).squeeze(2)
        
        # Combine outputs
        combined = torch.cat((lstm_attn_out, cnn_out), dim=1)
        combined = self.dropout(combined)
        
        # Fully connected layers
        fc_out = self.silu(self.bn1(self.fc1(combined)))
        fc_out = self.dropout(fc_out)
        fc_out = self.silu(self.bn2(self.fc2(fc_out)))
        fc_out = self.dropout(fc_out)
        output = self.fc3(fc_out)
        
        return output

class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32], output_size=1, dropout_rate=0.3):
        super(MLPModel, self).__init__()
        
        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes
        
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            self.layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            self.layers.append(nn.SiLU())
            self.layers.append(nn.Dropout(dropout_rate))
        
        self.output_layer = nn.Linear(layer_sizes[-1], output_size)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        
        for layer in self.layers:
            x = layer(x)
        
        return self.output_layer(x)

def custom_loss(y_pred, y_true, alpha=0.85):
    mse_loss = torch.mean(torch.square(y_pred - y_true))
    huber_loss = torch.nn.functional.smooth_l1_loss(y_pred, y_true)
    return alpha * mse_loss + (1 - alpha) * huber_loss

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16):
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=300, patience=40, model_name="Model"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True, min_lr=1e-6)
    
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
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    return train_losses[:epoch], val_losses[:epoch]
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
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, {model_name} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if val_loss < best_val_loss * 0.995 and not np.isnan(val_loss):
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping {model_name} after {epoch+1} epochs')
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break
        
        if epoch > num_epochs * 0.7 and optimizer.param_groups[0]['lr'] > 1e-5:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.95
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return train_losses, val_losses

def train_random_forest(X_train, y_train, X_test, y_test, feature_names):
    if isinstance(X_train, torch.Tensor):
        X_train = X_train.numpy()
    if isinstance(y_train, torch.Tensor):
        y_train = y_train.numpy()
    if isinstance(X_test, torch.Tensor):
        X_test = X_test.numpy()
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.numpy()
    
    if y_train.ndim > 1:
        y_train = y_train.squeeze()
    if y_test.ndim > 1:
        y_test = y_test.squeeze()
    
    print("Training Random Forest model...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    feature_importances = rf_model.feature_importances_
    feature_importance_dict = dict(zip(feature_names, feature_importances))
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 important features:")
    for feature, importance in sorted_features[:10]:
        print(f"{feature}: {importance:.4f}")
    
    y_pred = rf_model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Random Forest Validation Metrics - MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
    
    return rf_model, y_pred

def ensemble_predictions(models, X_test, weights=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_preds = []
    
    for model in models:
        if isinstance(model, torch.nn.Module):
            model.eval()
            model.to(device)
            with torch.no_grad():
                if isinstance(X_test, torch.Tensor):
                    X_test_device = X_test.to(device)
                    preds = model(X_test_device).cpu().numpy()
                else:
                    X_test_tensor = torch.FloatTensor(X_test).to(device)
                    preds = model(X_test_tensor).cpu().numpy()
        else:
            if isinstance(X_test, torch.Tensor):
                preds = model.predict(X_test.cpu().numpy())
            else:
                preds = model.predict(X_test)
        all_preds.append(preds)
    
    if weights is None:
        weights = [1/len(models)] * len(models)
    
    weights = np.array(weights) / sum(weights)
    
    # Make sure predictions have consistent shapes
    reshaped_preds = []
    for pred in all_preds:
        if pred.ndim > 1:
            reshaped_preds.append(pred.reshape(-1))
        else:
            reshaped_preds.append(pred)
    
    ensemble_preds = np.average(np.column_stack(reshaped_preds), axis=1, weights=weights)
    
    return ensemble_preds.reshape(-1, 1)

def evaluate_model(predictions, y_test, y_scaler, file_names_test, y_test_actual, model_name="Model"):
    if predictions.ndim > 1:
        predictions = predictions.squeeze()
    y_test_scaled = y_test.squeeze()
    
    y_pred_transformed = y_scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test_transformed = y_scaler.inverse_transform(y_test_scaled.reshape(-1, 1))
    
    y_pred_actual = np.expm1(np.clip(y_pred_transformed, 0, None))
    y_test_actual_pred = np.expm1(y_test_transformed)
    
    y_test_actual = y_test_actual.reshape(-1, 1)
    
    # Calculate error metrics
    mse = mean_squared_error(y_test_actual, y_pred_actual)
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual.ravel(), y_pred_actual.ravel())
    
    # Calculate relative error metrics
    relative_errors = np.abs(y_pred_actual - y_test_actual) / (y_test_actual + 1e-10)
    mean_relative_error = np.mean(relative_errors)
    median_relative_error = np.median(relative_errors)
    
    print(f"\n{model_name} Results:")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    print(f"Mean Relative Error: {mean_relative_error:.4f}, Median Relative Error: {median_relative_error:.4f}")
    
    # Detailed prediction analysis
    results = []
    for i in range(len(file_names_test)):
        result = {
            'file_name': file_names_test[i],
            'actual': float(y_test_actual[i]),
            'predicted': float(y_pred_actual[i]),
            'error': float(y_pred_actual[i] - y_test_actual[i]),
            'relative_error': float((y_pred_actual[i] - y_test_actual[i]) / (y_test_actual[i] + 1e-10)),
            'absolute_error': float(abs(y_pred_actual[i] - y_test_actual[i])),
            'percent_error': float(abs(y_pred_actual[i] - y_test_actual[i]) / (y_test_actual[i] + 1e-10) * 100)
        }
        results.append(result)
    
    # Sort results by relative error
    sorted_results = sorted(results, key=lambda x: x['relative_error'], reverse=True)
    
    print("\nTop 5 worst predictions (by relative error):")
    for i in range(min(5, len(sorted_results))):
        r = sorted_results[i]
        print(f"{r['file_name']}: Actual={r['actual']:.4f}, Pred={r['predicted']:.4f}, Error={r['error']:.4f} ({r['percent_error']:.2f}%)")
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'mean_relative_error': mean_relative_error,
        'median_relative_error': median_relative_error,
        'predictions': y_pred_actual,
        'actuals': y_test_actual,
        'detailed_results': results
    }

def plot_results(train_losses, val_losses, y_actual, y_pred, results, save_path=None):
    plt.figure(figsize=(16, 12))
    
    # Plot training curve
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot actual vs predicted values
    plt.subplot(2, 2, 2)
    plt.scatter(y_actual, y_pred, alpha=0.5)
    min_val = min(np.min(y_actual), np.min(y_pred))
    max_val = max(np.max(y_actual), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title('Actual vs Predicted Execution Time')
    plt.xlabel('Actual Execution Time')
    plt.ylabel('Predicted Execution Time')
    plt.grid(True)
    
    # Plot error distribution
    errors = y_pred.flatten() - y_actual.flatten()
    plt.subplot(2, 2, 3)
    plt.hist(errors, bins=30, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Plot relative error distribution
    rel_errors = np.abs(errors) / (y_actual.flatten() + 1e-10) * 100
    plt.subplot(2, 2, 4)
    plt.hist(rel_errors, bins=30, alpha=0.7)
    plt.title('Relative Error Distribution')
    plt.xlabel('Relative Error (%)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def save_model_results(models, results, feature_names, directory="model_results"):
    os.makedirs(directory, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(directory, "metrics.json"), 'w') as f:
        json.dump({
            'mse': results['mse'],
            'mae': results['mae'],
            'r2': results['r2'],
            'mean_relative_error': results['mean_relative_error'],
            'median_relative_error': results['median_relative_error']
        }, f, indent=4)
    
    # Save detailed prediction results
    with open(os.path.join(directory, "predictions.json"), 'w') as f:
        json.dump(results['detailed_results'], f, indent=4)
    
    # Save feature importance if RF model is available
    for model in models:
        if hasattr(model, 'feature_importances_'):
            feature_imp = dict(zip(feature_names, model.feature_importances_))
            sorted_features = {k: v for k, v in sorted(feature_imp.items(), key=lambda item: item[1], reverse=True)}
            
            with open(os.path.join(directory, "feature_importance.json"), 'w') as f:
                json.dump(sorted_features, f, indent=4)
    
    # Save models if they are PyTorch models
    for i, model in enumerate(models):
        if isinstance(model, nn.Module):
            torch.save(model.state_dict(), os.path.join(directory, f"model_{i}.pt"))
    
    print(f"Results saved to {directory}")

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(data_dir, output_dir="output", test_size=15):
    set_seed(42)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Process data
    train_features, test_features, test_file_names, function_metadata = process_directory(data_dir, test_size)
    
    if train_features is None:
        print("Error processing data. Exiting.")
        return
    
    # Prepare data for modeling
    (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
     X_train, X_test, y_scaler, input_dim, y_train_actual, y_test_actual,
     train_meta_df, test_meta_df, feature_names) = prepare_data_for_model(train_features, test_features)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)
    
    # Define models
    models = []
    
    # MLP model
    mlp_model = MLPModel(
        input_size=input_dim,
        hidden_sizes=[256, 128, 64, 32],
        output_size=1,
        dropout_rate=0.3
    )
    optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Hybrid model
    hybrid_model = HybridAttentionModel(
        input_size=input_dim,
        hidden_sizes=[128, 64, 32],
        output_size=1,
        dropout_rate=0.3
    )
    optimizer_hybrid = optim.Adam(hybrid_model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Define loss function
    criterion = custom_loss
    
    # Train models
    mlp_train_losses, mlp_val_losses = train_model(
        mlp_model, train_loader, test_loader, criterion, optimizer_mlp,
        num_epochs=300, patience=40, model_name="MLP"
    )
    
    hybrid_train_losses, hybrid_val_losses = train_model(
        hybrid_model, train_loader, test_loader, criterion, optimizer_hybrid,
        num_epochs=300, patience=40, model_name="Hybrid"
    )
    
    # Train Random Forest model
    rf_model, rf_preds = train_random_forest(X_train, y_train_tensor, X_test, y_test_tensor, feature_names)
    
    # Add models to ensemble
    models.append(mlp_model)
    models.append(hybrid_model)
    models.append(rf_model)
    
    # Make ensemble predictions
    ensemble_weights = [0.3, 0.3, 0.4]  # Weights for MLP, Hybrid, RF
    ensemble_preds = ensemble_predictions(models, X_test_tensor, weights=ensemble_weights)
    
    # Evaluate final model
    results = evaluate_model(
        ensemble_preds, y_test_tensor, y_scaler, test_file_names, 
        y_test_actual, model_name="Ensemble"
    )
    
    # Plot results
    plot_results(
        mlp_train_losses, mlp_val_losses, 
        results['actuals'], results['predictions'], 
        results, save_path=os.path.join(output_dir, "results_plot.png")
    )
    
    # Save results
    save_model_results(models, results, feature_names, directory=output_dir)
    
    return models, results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train execution time prediction models")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing JSON data files")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--test_size", type=int, default=15, help="Number of samples to use for testing")
    
    args = parser.parse_args()
    
    main(args.data_dir, args.output_dir, args.test_size)
