import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import OrderedDict, Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

# Recursive function to flatten JSON data and extract features
def flatten_json(data, parent_key=''):
    features = {}
    
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}_{key}" if parent_key else key
            # Recursively flatten nested structures
            nested_features = flatten_json(value, new_key)
            features.update(nested_features)
    elif isinstance(data, list):
        # For lists, extract length and aggregate statistics
        if parent_key:
            features[f"{parent_key}_length"] = len(data)
            
            # If list is not empty and contains simple types, extract statistics
            if data and all(not isinstance(x, (dict, list)) for x in data):
                numeric_values = []
                for item in data:
                    try:
                        numeric_values.append(float(item))
                    except (ValueError, TypeError):
                        pass
                
                if numeric_values:
                    features[f"{parent_key}_mean"] = np.mean(numeric_values)
                    features[f"{parent_key}_std"] = np.std(numeric_values)
                    features[f"{parent_key}_max"] = np.max(numeric_values)
                    features[f"{parent_key}_min"] = np.min(numeric_values)
            
            # Recursively process list items (sample up to 5 to avoid explosion)
            for i, value in enumerate(data[:5]):
                nested_features = flatten_json(value, f"{parent_key}_{i}")
                features.update(nested_features)
    else:
        # Extract value directly for simple types
        try:
            features[parent_key] = float(data)
        except (ValueError, TypeError):
            # For non-numeric values, use simple encoding
            if isinstance(data, str):
                # Use a hash for strings, normalized
                features[parent_key] = hash(str(data)) % 1000 / 1000.0
            else:
                features[parent_key] = 0.0
    
    return features

# Extract target variable (total_execution_time_ms)
def extract_target(data):
    for item in data.get("scheduling_data", []):
        if item.get("name") == "total_execution_time_ms":
            return item["value"]
    return None

# Extract key statistics and features from scheduling data
def extract_important_features(data):
    features = {}
    
    # Extract direct features known to be important for performance
    if "hardware" in data:
        hw = data["hardware"]
        features["num_cores"] = hw.get("num_cores", 0)
        features["has_gpu"] = 1.0 if hw.get("has_gpu", False) else 0.0
        features["memory_gb"] = hw.get("memory_gb", 0)
    
    # Extract program complexity metrics
    if "program" in data:
        prog = data["program"]
        features["num_statements"] = prog.get("num_statements", 0)
        features["loop_count"] = prog.get("loop_count", 0)
        features["recursion_depth"] = prog.get("recursion_depth", 0)
    
    # Extract optimization flags
    if "compiler_options" in data:
        opt = data["compiler_options"]
        features["optimization_level"] = opt.get("optimization_level", 0)
        features["vectorization"] = 1.0 if opt.get("vectorization", False) else 0.0
        features["parallelization"] = 1.0 if opt.get("parallelization", False) else 0.0
    
    # Extract scheduling data metrics
    if "scheduling_data" in data:
        sched_data = data["scheduling_data"]
        for item in sched_data:
            if isinstance(item, dict) and "name" in item and "value" in item:
                if item["name"] != "total_execution_time_ms":  # Don't include target
                    features[f"sched_{item['name']}"] = item["value"]
    
    return features

# Custom Dataset with enhanced feature extraction
class EnhancedScheduleDataset(Dataset):
    def __init__(self, json_data_list, targets=None, feature_scaler=None, target_scaler=None, train=True):
        self.json_data = json_data_list
        
        # Extract features from JSON data
        print("Extracting features...")
        all_features = []
        for data in json_data_list:
            # Get important features
            direct_features = extract_important_features(data)
            
            # Also get flattened features for completeness
            flat_features = flatten_json(data)
            
            # Combine both
            combined_features = {**flat_features, **direct_features}
            all_features.append(combined_features)
        
        # Create unified feature set
        print("Creating feature matrix...")
        self.feature_names = set()
        for features in all_features:
            self.feature_names.update(features.keys())
        
        self.feature_names = sorted(list(self.feature_names))
        print(f"Total features extracted: {len(self.feature_names)}")
        
        # Convert to feature matrix
        X = np.zeros((len(all_features), len(self.feature_names)))
        for i, features in enumerate(all_features):
            for j, name in enumerate(self.feature_names):
                X[i, j] = features.get(name, 0.0)
        
        # Scale features
        if train:
            self.feature_scaler = StandardScaler()
            self.X = self.feature_scaler.fit_transform(X)
        else:
            self.X = feature_scaler.transform(X)
            self.feature_scaler = feature_scaler
        
        # Handle targets
        if targets is not None:
            if train:
                self.target_scaler = StandardScaler()
                self.y = self.target_scaler.fit_transform(np.array(targets).reshape(-1, 1)).flatten()
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
            target = self.targets[idx]
            return features, target
        else:
            return features

# Define a simple but effective MLP model
class ExecutionTimePredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout=0.3):
        super(ExecutionTimePredictor, self).__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# Load and preprocess all data
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
                
                if target is not None:
                    all_json_data.append(data)
                    all_targets.append(target)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return all_json_data, all_targets

# Training function with early stopping and lr scheduling
def train_model(model, train_loader, val_loader, criterion, num_epochs=100, patience=15):
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # For tracking training progress
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            print(f"Saving best model with validation loss: {best_val_loss:.6f}")
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.close()
    
    return model, train_losses, val_losses

# Prediction and evaluation function
def evaluate_model(model, test_dataset, target_scaler=None, num_samples=None):
    model.eval()
    
    if num_samples is None or num_samples > len(test_dataset):
        num_samples = len(test_dataset)
    
    test_loader = DataLoader(test_dataset, batch_size=num_samples, shuffle=False)
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            # Denormalize predictions and true values
            norm_preds = outputs.cpu().numpy().reshape(-1, 1)
            norm_trues = y_batch.cpu().numpy().reshape(-1, 1)
            
            predictions = target_scaler.inverse_transform(norm_preds).flatten()
            true_values = target_scaler.inverse_transform(norm_trues).flatten()
            
            break  # Only need one batch since we loaded all samples
    
    # Calculate metrics
    abs_errors = np.abs(predictions - true_values)
    rel_errors = abs_errors / np.maximum(true_values, 1e-10) * 100  # Avoid division by zero
    
    mse = np.mean(np.square(predictions - true_values))
    mae = np.mean(abs_errors)
    mape = np.mean(rel_errors)
    rmse = np.sqrt(mse)
    
    # Return sample predictions and overall metrics
    sample_results = []
    for i in range(min(10, num_samples)):
        sample_results.append({
            'predicted': predictions[i],
            'true': true_values[i],
            'error_pct': rel_errors[i]
        })
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }
    
    return sample_results, metrics, predictions, true_values

# Feature importance analysis
def analyze_feature_importance(model, feature_names):
    # Extract weights from the first layer
    first_layer = next(layer for layer in model.model if isinstance(layer, nn.Linear))
    weights = first_layer.weight.data.abs().mean(dim=0).cpu().numpy()
    
    # Create features importance dictionary
    importance = {name: weight for name, weight in zip(feature_names, weights)}
    
    # Sort by importance
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_importance

# Main execution
def main():
    # Hyperparameters
    batch_size = 32
    num_epochs = 200
    
    # Load data
    folder_path = 'synthetic_data'
    print("Loading data...")
    all_json_data, all_targets = load_data(folder_path)
    
    if not all_json_data:
        print("No valid data found.")
        return
    
    print(f"Loaded {len(all_json_data)} samples.")
    
    # Look at the distribution of targets
    plt.figure(figsize=(10, 5))
    plt.hist(all_targets, bins=50)
    plt.xlabel('Execution Time (ms)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Execution Times')
    plt.savefig('target_distribution.png')
    plt.close()
    
    # Apply log transformation if data is skewed
    log_transform = False
    if np.mean(all_targets) > 10 * np.median(all_targets):
        print("Applying log transformation to targets due to skewed distribution")
        all_targets = np.log1p(all_targets)
        log_transform = True
    
    # Split into train, validation, and test sets
    train_data, test_data, train_targets, test_targets = train_test_split(
        all_json_data, all_targets, test_size=0.2, random_state=42
    )
    
    train_data, val_data, train_targets, val_targets = train_test_split(
        train_data, train_targets, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2 of original
    )
    
    print(f"Train set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    # Create datasets with normalization
    print("Creating datasets...")
    train_dataset = EnhancedScheduleDataset(train_data, train_targets, train=True)
    feature_scaler = train_dataset.feature_scaler
    target_scaler = train_dataset.target_scaler
    
    val_dataset = EnhancedScheduleDataset(
        val_data, val_targets, 
        feature_scaler=feature_scaler, 
        target_scaler=target_scaler, 
        train=False
    )
    
    test_dataset = EnhancedScheduleDataset(
        test_data, test_targets, 
        feature_scaler=feature_scaler, 
        target_scaler=target_scaler, 
        train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Get input dimension
    input_size = train_dataset.X.shape[1]
    print(f"Input dimension: {input_size}")
    
    # Initialize model
    model = ExecutionTimePredictor(
        input_size, 
        hidden_sizes=[512, 256, 128, 64],  # Deeper network
        dropout=0.4  # Increased dropout for regularization
    ).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    
    # Define loss function - use Huber loss for robustness to outliers
    criterion = nn.SmoothL1Loss()
    
    # Train model
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, num_epochs=num_epochs
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    sample_results, metrics, all_preds, all_trues = evaluate_model(
        model, test_dataset, target_scaler
    )
    
    # If we applied log transform, adjust the metrics message
    transform_msg = " (log-transformed)" if log_transform else ""
    
    # Print sample results
    print(f"\nSample Predictions{transform_msg}:")
    for i, result in enumerate(sample_results, 1):
        print(f"Sample {i}:")
        print(f"  Predicted Time: {result['predicted']:.2f} ms")
        print(f"  True Time: {result['true']:.2f} ms")
        print(f"  Error Percentage: {result['error_pct']:.2f}%")
    
    # Print overall metrics
    print(f"\nOverall Metrics{transform_msg}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.2f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 7))
    plt.scatter(all_trues, all_preds, alpha=0.5)
    
    # Add diagonal line for perfect prediction
    max_val = max(np.max(all_trues), np.max(all_preds))
    min_val = min(np.min(all_trues), np.min(all_preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('True Execution Time (ms)')
    plt.ylabel('Predicted Execution Time (ms)')
    plt.title('Predicted vs Actual Execution Times')
    plt.savefig('predictions_vs_actual.png')
    plt.close()
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    feature_importance = analyze_feature_importance(model, train_dataset.feature_names)
    
    print("\nTop 20 most important features:")
    for feature, importance in feature_importance[:20]:
        print(f"  {feature}: {importance:.4f}")
    
    # Plot feature importance (top 30)
    plt.figure(figsize=(12, 8))
    top_features = feature_importance[:30]
    features, importances = zip(*top_features)
    
    plt.barh(range(len(features)), importances, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance')
    plt.title('Top 30 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

if __name__ == "__main__":
    main()
