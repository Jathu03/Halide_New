import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_regression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import random
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

class FeatureExtractor:
    """Enhanced feature extraction with better error handling and more comprehensive features"""
    
    @staticmethod
    def get_execution_time(file_path):
        """Extract execution time with robust error handling"""
        try:
            with open(file_path, 'rb') as f:
                raw_content = f.read()
                content = raw_content.decode('utf-8', errors='replace').replace('\0', '')
                data = json.loads(content)
            
            if 'programming_details' not in data:
                print(f"Warning: 'programming_details' key not found in {file_path}")
                return None
            
            # Try multiple ways to find execution time
            execution_time = None
            
            # Method 1: Check scheduling_data
            if 'scheduling_data' in data:
                for item in data["scheduling_data"]:
                    if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
                        execution_time = item.get('value')
                        if execution_time is not None:
                            return float(execution_time)
            
            # Method 2: Check last item in schedules
            if 'Schedules' in data.get('programming_details', {}):
                schedules = data['programming_details']['Schedules']
                if schedules and isinstance(schedules[-1], dict):
                    execution_time = schedules[-1].get('value')
                    if execution_time is not None:
                        return float(execution_time)
            
            # Method 3: Check for benchmark results
            if 'benchmark_results' in data:
                if isinstance(data['benchmark_results'], list) and data['benchmark_results']:
                    return float(data['benchmark_results'][0].get('time', 0))
            
            print(f"Warning: Could not find execution time in {file_path}")
            return None
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    @staticmethod
    def extract_operation_features(nodes):
        """Extract detailed operation features from nodes"""
        op_counts = defaultdict(int)
        op_types = set()
        compute_intensity = 0
        memory_ops = 0
        control_ops = 0
        
        for node in nodes:
            if 'Details' not in node or 'Op histogram' not in node['Details']:
                continue
                
            for op_line in node['Details']['Op histogram']:
                parts = op_line.strip().split(':')
                if len(parts) == 2:
                    op_name = parts[0].strip().lower()
                    op_count = int(parts[1].strip())
                    
                    op_counts[op_name] += op_count
                    op_types.add(op_name)
                    
                    # Categorize operations
                    if 'load' in op_name or 'store' in op_name:
                        memory_ops += op_count
                    elif 'if' in op_name or 'select' in op_name:
                        control_ops += op_count
                    else:
                        compute_intensity += op_count
        
        return {
            'op_counts': dict(op_counts),
            'unique_op_types': len(op_types),
            'compute_intensity': compute_intensity,
            'memory_ops': memory_ops,
            'control_ops': control_ops,
            'compute_to_memory_ratio': compute_intensity / (memory_ops + 1e-8)
        }

    @staticmethod
    def extract_scheduling_features(scheduling_data):
        """Extract comprehensive scheduling features"""
        if not scheduling_data:
            return {}
            
        features = {
            'total_bytes': 0,
            'total_vectors': 0,
            'total_parallelism': 0,
            'total_working_set': 0,
            'num_productions': 0,
            'num_realizations': 0,
            'scheduling_stages': len(scheduling_data)
        }
        
        # Initialize lists to track distributions
        bytes_list = []
        parallelism_list = []
        working_set_list = []
        
        for sched in scheduling_data:
            if not isinstance(sched, dict):
                continue
                
            # Sum features
            features['total_bytes'] += sched.get('bytes_at_production', 0)
            features['total_vectors'] += sched.get('num_vectors', 0)
            features['total_parallelism'] += sched.get('inner_parallelism', 0) * sched.get('outer_parallelism', 1)
            features['total_working_set'] += sched.get('working_set', 0)
            features['num_productions'] += sched.get('num_productions', 0)
            features['num_realizations'] += sched.get('num_realizations', 0)
            
            # Track distributions
            bytes_list.append(sched.get('bytes_at_production', 0))
            parallelism_list.append(sched.get('inner_parallelism', 0) * sched.get('outer_parallelism', 1))
            working_set_list.append(sched.get('working_set', 0))
        
        # Add distribution statistics
        if bytes_list:
            features['bytes_mean'] = np.mean(bytes_list)
            features['bytes_std'] = np.std(bytes_list)
            features['bytes_max'] = max(bytes_list)
            
        if parallelism_list:
            features['parallelism_mean'] = np.mean(parallelism_list)
            features['parallelism_std'] = np.std(parallelism_list)
            features['parallelism_max'] = max(parallelism_list)
            
        if working_set_list:
            features['working_set_mean'] = np.mean(working_set_list)
            features['working_set_std'] = np.std(working_set_list)
            
        # Add derived features
        if features['total_bytes'] > 0:
            features['bytes_per_vector'] = features['total_bytes'] / (features['total_vectors'] + 1e-8)
            features['memory_pressure'] = features['total_working_set'] / (features['total_bytes'] + 1e-8)
            features['parallelism_per_byte'] = features['total_parallelism'] / (features['total_bytes'] + 1e-8)
            
        return features

    @staticmethod
    def extract_features_from_file(file_path):
        """Main feature extraction method"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            execution_time = FeatureExtractor.get_execution_time(file_path)
            if execution_time is None or execution_time <= 0:
                return None
                
            programming_details = data.get('programming_details', {})
            nodes = programming_details.get('Nodes', [])
            edges = programming_details.get('Edges', [])
            scheduling_data = data.get('scheduling_data', programming_details.get('Schedules', []))
            
            # Extract operation features
            op_features = FeatureExtractor.extract_operation_features(nodes)
            
            # Extract scheduling features
            sched_features = FeatureExtractor.extract_scheduling_features(scheduling_data)
            
            # Build final feature set
            features = {
                'execution_time': execution_time,
                'nodes_count': len(nodes),
                'edges_count': len(edges),
                'node_edge_ratio': len(nodes) / (len(edges) + 1e-8),
                **op_features,
                **sched_features
            }
            
            # Add operation counts as individual features
            for op, count in op_features['op_counts'].items():
                features[f'op_{op}'] = count
                
            return features
            
        except Exception as e:
            print(f"Error extracting features from {file_path}: {str(e)}")
            return None

class DataProcessor:
    """Handles data loading, preprocessing, and feature engineering"""
    
    @staticmethod
    def process_directory(directory_path):
        """Process all JSON files in a directory"""
        all_features = []
        file_names = []
        
        for filename in sorted(os.listdir(directory_path)):
            if not filename.endswith('.json'):
                continue
                
            file_path = os.path.join(directory_path, filename)
            features = FeatureExtractor.extract_features_from_file(file_path)
            
            if features is not None:
                all_features.append(features)
                file_names.append(filename)
                
        return all_features, file_names

    @staticmethod
    def process_main_directory(main_dir, test_size=50, min_samples=100):
        """Process all subdirectories in main directory"""
        all_features = []
        all_file_names = []
        
        subdirs = sorted([d for d in os.listdir(main_dir) 
                         if os.path.isdir(os.path.join(main_dir, d))])
        
        if not subdirs:
            raise ValueError(f"No subdirectories found in {main_dir}")
            
        for subdir in subdirs:
            subdir_path = os.path.join(main_dir, subdir)
            features, file_names = DataProcessor.process_directory(subdir_path)
            
            if not features:
                print(f"Skipping {subdir} - no valid data")
                continue
                
            all_features.extend(features)
            all_file_names.extend([os.path.join(subdir, f) for f in file_names])
            print(f"Processed {subdir}: {len(features)} files")
            
        if len(all_features) < min_samples:
            raise ValueError(f"Insufficient data: {len(all_features)} samples (need at least {min_samples})")
            
        # Shuffle while maintaining reproducibility
        combined = list(zip(all_features, all_file_names))
        random.Random(42).shuffle(combined)
        all_features, all_file_names = zip(*combined)
        
        # Split data
        train_features = all_features[:-test_size]
        test_features = all_features[-test_size:]
        train_file_names = all_file_names[:-test_size]
        test_file_names = all_file_names[-test_size:]
        
        print(f"\nData Summary:")
        print(f"Total files: {len(all_features)}")
        print(f"Training files: {len(train_features)}")
        print(f"Testing files: {len(test_features)}")
        
        return train_features, test_features, list(test_file_names)

    @staticmethod
    def clean_and_transform_features(train_features, test_features):
        """Clean, transform, and select features"""
        # Convert to DataFrame
        train_df = pd.DataFrame(train_features)
        test_df = pd.DataFrame(test_features)
        
        # Drop constant features
        constant_cols = [col for col in train_df.columns 
                        if col != 'execution_time' and train_df[col].nunique() == 1]
        train_df = train_df.drop(columns=constant_cols)
        test_df = test_df.drop(columns=constant_cols)
        
        # Drop highly correlated features
        corr_matrix = train_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        train_df = train_df.drop(columns=to_drop)
        test_df = test_df.drop(columns=to_drop)
        
        # Log transform execution time and add as target
        train_df['log_execution_time'] = np.log1p(train_df['execution_time'])
        test_df['log_execution_time'] = np.log1p(test_df['execution_time'])
        
        # Select numeric features only
        numeric_cols = train_df.select_dtypes(include=['number']).columns.tolist()
        train_df = train_df[numeric_cols]
        test_df = test_df[numeric_cols]
        
        # Feature selection using SelectKBest
        X_train = train_df.drop(['execution_time', 'log_execution_time'], axis=1)
        y_train = train_df['log_execution_time']
        
        selector = SelectKBest(f_regression, k=min(30, X_train.shape[1]))
        selector.fit(X_train, y_train)
        selected_cols = X_train.columns[selector.get_support()]
        
        # Apply selection
        X_train = X_train[selected_cols]
        X_test = test_df[selected_cols]
        y_train = train_df['log_execution_time']
        y_test = test_df['log_execution_time']
        
        # Power transform for heavy-tailed features
        pt = PowerTransformer()
        X_train_transformed = pt.fit_transform(X_train)
        X_test_transformed = pt.transform(X_test)
        
        # Scale features
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train_transformed)
        X_test_scaled = scaler_X.transform(X_test_transformed)
        
        # Scale target
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)
        y_train_tensor = torch.FloatTensor(y_train_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
        y_test_tensor = torch.FloatTensor(y_test_scaled)
        
        print(f"\nFeature Engineering Summary:")
        print(f"Selected {len(selected_cols)} features:")
        print(selected_cols.tolist())
        print(f"Input feature dimension: {X_train_scaled.shape[1]}")
        
        return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
                scaler_y, X_train_scaled.shape[1], True)

class HierarchicalAttentionModel(nn.Module):
    """Enhanced model with hierarchical attention and residual connections"""
    
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], output_size=1, 
                 dropout_rate=0.4, num_heads=4):
        super(HierarchicalAttentionModel, self).__init__()
        
        # Feature-level attention
        self.feature_attention = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], input_size),
            nn.Softmax(dim=-1)
        )
        
        # LSTM layers with skip connections
        self.lstm_layers = nn.ModuleList()
        self.lstm_norms = nn.ModuleList()
        
        for i in range(len(hidden_sizes)):
            input_dim = input_size if i == 0 else hidden_sizes[i-1]
            self.lstm_layers.append(
                nn.LSTM(input_dim, hidden_sizes[i], batch_first=True, bidirectional=True)
            )
            self.lstm_norms.append(nn.LayerNorm(hidden_sizes[i] * 2))
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_sizes[-1] * 2, 
            num_heads=num_heads,
            dropout=dropout_rate
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_sizes[-1] * 2, hidden_sizes[-1] * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_sizes[-1] * 4, hidden_sizes[-1] * 2),
            nn.Dropout(dropout_rate)
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_sizes[-1] * 2, hidden_sizes[-1]),
            nn.GELU(),
            nn.Linear(hidden_sizes[-1], output_size)
        )
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Feature attention
        attn_weights = self.feature_attention(x.squeeze(1))
        x = x * attn_weights.unsqueeze(1)
        
        # LSTM processing with skip connections
        lstm_out = x
        for i, (lstm, norm) in enumerate(zip(self.lstm_layers, self.lstm_norms)):
            residual = lstm_out
            lstm_out, _ = lstm(lstm_out)
            lstm_out = norm(lstm_out)
            
            # Skip connection if dimensions match
            if i > 0 and residual.size(-1) == lstm_out.size(-1):
                lstm_out = lstm_out + residual
                
            lstm_out = self.dropout(lstm_out)
        
        # Permute for attention (seq_len, batch_size, embed_dim)
        lstm_out = lstm_out.permute(1, 0, 2)
        
        # Multi-head attention
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_output = self.dropout(attn_output)
        
        # Feed-forward network with residual
        ffn_output = self.ffn(attn_output)
        ffn_output = ffn_output + attn_output  # Residual connection
        
        # Average over sequence dimension
        pooled = torch.mean(ffn_output, dim=0)
        
        # Final output
        output = self.output_layers(pooled)
        
        return output

class ModelTrainer:
    """Handles model training and evaluation"""
    
    @staticmethod
    def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32):
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )
        
        return train_loader, test_loader
    
    @staticmethod
    def train_model(model, train_loader, test_loader, criterion, optimizer, 
                   num_epochs=200, patience=25):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nTraining on {device}")
        model.to(device)
        
        # Mixed precision training
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        
        # Learning rate schedulers
        lr_scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        cos_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs//4, eta_min=1e-6)
        
        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(num_epochs):
            start_time = time.time()
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item() * inputs.size(0)
            
            # Validation phase
            val_loss = ModelTrainer.evaluate_model(model, test_loader, criterion, device)
            
            # Update learning rate
            lr_scheduler.step(val_loss)
            cos_scheduler.step()
            
            # Record history
            train_loss /= len(train_loader.dataset)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict().copy()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    model.load_state_dict(best_model_state)
                    break
            
            # Print progress
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        return history
    
    @staticmethod
    def evaluate_model(model, data_loader, criterion, device):
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
        
        return val_loss / len(data_loader.dataset)
    
    @staticmethod
    def evaluate_predictions(model, X_test, y_test, y_scaler, file_names_test):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        X_test = X_test.to(device)
        with torch.no_grad():
            y_pred_scaled = model(X_test)
        
        # Inverse scaling
        y_pred = y_scaler.inverse_transform(y_pred_scaled.cpu().numpy())
        y_test = y_scaler.inverse_transform(y_test.cpu().numpy())
        
        # Convert from log scale to original
        y_pred_actual = np.expm1(y_pred)
        y_test_actual = np.expm1(y_test)
        
        # Calculate errors
        errors = np.abs(y_test_actual - y_pred_actual)
        relative_errors = errors / (y_test_actual + 1e-8)
        
        # Group by subfolder
        results = defaultdict(list)
        for i, file_path in enumerate(file_names_test):
            subfolder = file_path.split('/')[0]
            results[subfolder].append({
                'file': file_path,
                'actual': y_test_actual[i][0],
                'predicted': y_pred_actual[i][0],
                'error': errors[i][0],
                'relative_error': relative_errors[i][0]
            })
        
        # Print detailed results
        for subfolder, sub_results in results.items():
            print(f"\nResults for {subfolder}:")
            avg_error = np.mean([r['relative_error'] for r in sub_results])
            print(f"  Average relative error: {avg_error:.1%}")
            
            for r in sub_results[:3]:  # Print first 3 examples
                print(f"  {r['file']}")
                print(f"    Actual: {r['actual']:.2f} ms")
                print(f"    Predicted: {r['predicted']:.2f} ms")
                print(f"    Error: {r['error']:.2f} ms ({r['relative_error']:.1%})")
        
        # Calculate overall metrics
        mse = np.mean(errors**2)
        rmse = np.sqrt(mse)
        mae = np.mean(errors)
        mape = np.mean(relative_errors) * 100
        
        print("\nOverall Model Performance:")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        return y_test_actual, y_pred_actual

def main(main_dir="synthetic_data"):
    print("Starting Halide Cost Model Training")
    print("=" * 50)
    
    # Step 1: Data Processing
    print("\nProcessing data...")
    train_features, test_features, test_file_names = DataProcessor.process_main_directory(main_dir)
    
    # Step 2: Feature Engineering
    print("\nEngineering features...")
    (X_train, y_train, X_test, y_test, 
     y_scaler, input_size, _) = DataProcessor.clean_and_transform_features(
        train_features, test_features
    )
    
    # Step 3: Create Data Loaders
    print("\nCreating data loaders...")
    train_loader, test_loader = ModelTrainer.create_data_loaders(
        X_train, y_train, X_test, y_test, batch_size=32
    )
    
    # Step 4: Initialize Model
    print("\nInitializing model...")
    model = HierarchicalAttentionModel(
        input_size=input_size,
        hidden_sizes=[256, 128, 64],
        dropout_rate=0.3,
        num_heads=4
    )
    print(model)
    
    # Step 5: Training Setup
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=1e-3, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    
    # Step 6: Train Model
    print("\nTraining model...")
    history = ModelTrainer.train_model(
        model, train_loader, test_loader, criterion, optimizer,
        num_epochs=200, patience=25
    )
    
    # Step 7: Evaluate Model
    print("\nEvaluating model...")
    y_test_actual, y_pred_actual = ModelTrainer.evaluate_predictions(
        model, X_test, y_test, y_scaler, test_file_names
    )
    
    # Step 8: Save Model
    print("\nSaving model...")
    try:
        # Trace with example input
        example_input = torch.randn(1, 1, input_size).to(next(model.parameters()).device)
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save("halide_cost_model.pt")
        print("Model saved as 'halide_cost_model.pt'")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
    
    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main()
