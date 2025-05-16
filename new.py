import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Try importing torch, with helpful error message if not available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, random_split
except ImportError:
    print("PyTorch is not installed. Please install it using:")
    print("pip install torch torchvision")
    print("or visit https://pytorch.org/get-started/locally/ for installation instructions specific to your system.")
    exit(1)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class GraphDataset(Dataset):
    def __init__(self, features, execution_times):
        self.features = features
        self.execution_times = execution_times
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.execution_times[idx]

class LSTMExecutionTimePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMExecutionTimePredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # We use the last time step output for prediction
        out = self.fc(out[:, -1, :])
        return out.squeeze()

def examine_json_structure(file_path):
    """
    Examine the structure of a JSON file to help debug extraction issues
    """
    try:
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        
        # Check for key structures
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
        
        # Check node structure if nodes exist
        node_structure = {}
        if num_nodes > 0:
            if has_without_extern:
                first_node = without_extern["nodes"][0]
            else:
                first_node = json_data["nodes"][0]
                
            node_structure = {key: type(value).__name__ for key, value in first_node.items()}
            
            # Check stages structure
            if "stages" in first_node:
                if isinstance(first_node["stages"], list):
                    node_structure["stages"] = f"list[{len(first_node['stages'])}]"
                    if first_node["stages"]:
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
    """
    Extract relevant features from the JSON data
    """
    # Check if the data is under 'without_extern' key
    if "without_extern" in json_data:
        json_data = json_data["without_extern"]
    
    # Debug: Print JSON structure
    if debug:
        print("JSON keys:", json_data.keys())
    
    # Extract global features if available
    global_features = json_data.get("global_features", {})
    if debug and not global_features:
        print("Warning: No global_features found in JSON")
    
    execution_time = global_features.get("execution_time_ms", 0)
    if debug:
        print(f"Execution time: {execution_time}")
    
    # Skip files with invalid execution times
    if execution_time <= 0:
        if debug:
            print(f"Skipping file due to invalid execution time: {execution_time}")
        return None, None
    
    # Process nodes
    nodes = json_data.get("nodes", [])
    if debug:
        print(f"Number of nodes: {len(nodes)}")
        if nodes:
            print(f"First node keys: {nodes[0].keys() if isinstance(nodes[0], dict) else 'Not a dict'}")
    
    node_features = []
    
    for i, node in enumerate(nodes):
        # Skip if node is not a dictionary
        if not isinstance(node, dict):
            continue
            
        node_feature_vector = []
        
        # Basic node properties
        node_feature_vector.append(1 if node.get("input", False) else 0)
        node_feature_vector.append(1 if node.get("output", False) else 0)
        node_feature_vector.append(1 if node.get("pointwise", False) else 0)
        node_feature_vector.append(1 if node.get("boundary_condition", False) else 0)
        node_feature_vector.append(1 if node.get("wrapper", False) else 0)
        
        # Extract schedule features
        stages = node.get("stages", [])
        if stages:
            # Check if stages is a list or dictionary
            if isinstance(stages, list) and stages:
                first_stage = stages[0]
            elif isinstance(stages, dict):
                first_stage = stages
            else:
                first_stage = {}
                
            pipeline_features = first_stage.get("pipeline_features", {})
            schedule_features = pipeline_features.get("schedule_features", {})
            
            if debug and i == 0:  # Only print for the first node to avoid clutter
                print(f"Stage type: {type(stages)}")
                print(f"Pipeline features keys: {pipeline_features.keys() if pipeline_features else 'None'}")
                print(f"Schedule features keys: {schedule_features.keys() if schedule_features else 'None'}")
            
            # Select important numerical features
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
                    node_feature_vector.append(value)
                except (ValueError, TypeError):
                    if debug:
                        print(f"Warning: Could not convert {feature} to float")
                    node_feature_vector.append(0.0)
            
            # Extract operation histogram information
            op_histogram = pipeline_features.get("op_histogram", {})
            float_ops = op_histogram.get("Float", {})
            
            if debug and i == 0:
                print(f"Op histogram keys: {op_histogram.keys() if op_histogram else 'None'}")
                print(f"Float ops keys: {float_ops.keys() if float_ops else 'None'}")
            
            # Count total operations
            try:
                total_ops = sum(float_ops.values())
                node_feature_vector.append(total_ops)
            except:
                node_feature_vector.append(0.0)
                if debug:
                    print("Warning: Could not calculate total_ops")
            
            # Count specific operation types
            try:
                compute_ops = float_ops.get("Add", 0) + float_ops.get("Sub", 0) + float_ops.get("Mul", 0) + float_ops.get("Div", 0)
                node_feature_vector.append(compute_ops)
            except:
                node_feature_vector.append(0.0)
                if debug:
                    print("Warning: Could not calculate compute_ops")
            
            try:
                memory_ops = float_ops.get("Variable", 0) + float_ops.get("Param", 0) + float_ops.get("ImageCall", 0)
                node_feature_vector.append(memory_ops)
            except:
                node_feature_vector.append(0.0)
                if debug:
                    print("Warning: Could not calculate memory_ops")
            
            try:
                control_ops = float_ops.get("Select", 0) + float_ops.get("Let", 0) + float_ops.get("FuncCall", 0)
                node_feature_vector.append(control_ops)
            except:
                node_feature_vector.append(0.0)
                if debug:
                    print("Warning: Could not calculate control_ops")
        
        # If we don't have stages, add zeros
        else:
            node_feature_vector.extend([0.0] * (15 + 4))  # 15 schedule features + 4 operation counts
        
        node_features.append(node_feature_vector)
    
    # If we have no valid nodes, return None
    if not node_features:
        if debug:
            print("No valid nodes found in JSON data")
        return None, None
    
    # Ensure all nodes have the same feature length by padding with zeros if needed
    max_feature_len = max(len(f) for f in node_features) if node_features else 0
    if debug:
        print(f"Max feature length: {max_feature_len}")
        feature_lengths = [len(f) for f in node_features]
        if len(set(feature_lengths)) > 1:
            print(f"Warning: Inconsistent feature lengths: {feature_lengths}")
    
    padded_features = [f + [0] * (max_feature_len - len(f)) for f in node_features]
    
    # Convert to numpy array and ensure consistent shape
    features = np.array(padded_features, dtype=np.float32)
    if debug:
        print(f"Features shape: {features.shape}")
    
    return features, execution_time

def process_data_directory(root_dir, debug=False, max_files=None):
    """
    Process all JSON files in the directory structure
    """
    all_features = []
    all_execution_times = []
    file_paths = []
    
    skipped_format = 0
    skipped_execution = 0
    skipped_other = 0
    processed = 0
    
    # First, find all JSON files
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
            # Check if the directory exists
            if not os.path.exists(root_dir):
                print(f"Directory {root_dir} does not exist!")
            else:
                # List contents of directory
                print(f"Contents of {root_dir}:")
                for item in os.listdir(root_dir):
                    print(f"  {item}")
    
    # Process files with progress bar
    for file_path in tqdm(all_json_files, desc="Processing JSON files"):
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            if debug and processed < 2:  # Only examine structure for first few files
                print(f"\nExamining structure of {file_path}:")
                structure = examine_json_structure(file_path)
                for key, value in structure.items():
                    print(f"  {key}: {value}")
            
            features, execution_time = extract_features_from_json(json_data, debug=(debug and processed < 2))
            
            if features is None:
                if execution_time is None:
                    # Check why it was skipped
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
    """
    Pad sequences to the same length
    """
    if not sequences:
        return np.array([])
        
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            # Pad with zeros
            padded = np.pad(seq, ((0, max_length - len(seq)), (0, 0)), 'constant')
        else:
            # Truncate
            padded = seq[:max_length]
        padded_sequences.append(padded)
    
    return np.array(padded_sequences)

def train_model(train_loader, val_loader, model, criterion, optimizer, device, num_epochs=50, patience=10):
    """
    Train the LSTM model
    """
    model.to(device)
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    return model, train_losses, val_losses

def evaluate_model(test_loader, model, device):
    """
    Evaluate the model on test data
    """
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
    
    # Calculate error metrics
    absolute_errors = np.abs(predictions - actuals)
    percentage_errors = (absolute_errors / actuals) * 100
    
    mean_absolute_error = np.mean(absolute_errors)
    mean_percentage_error = np.mean(percentage_errors)
    median_percentage_error = np.median(percentage_errors)
    
    print(f"Mean Absolute Error: {mean_absolute_error:.4f}")
    print(f"Mean Percentage Error: {mean_percentage_error:.2f}%")
    print(f"Median Percentage Error: {median_percentage_error:.2f}%")
    
    return predictions, actuals, percentage_errors

def main(debug=True, max_files=None):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Process data
    root_dir = "Graph_Output"
    all_features, all_execution_times, file_paths = process_data_directory(root_dir, debug=debug, max_files=max_files)
    
    if not all_features:
        print("No valid data found. Exiting.")
        return
    
    # Pad sequences to the same length
    padded_features = pad_sequences(all_features)
    print(f"Padded features shape: {padded_features.shape}")
    
    # Convert to numpy arrays
    execution_times = np.array(all_execution_times, dtype=np.float32)
    print(f"Execution times shape: {execution_times.shape}")
    print(f"Execution times range: {execution_times.min()} to {execution_times.max()}")
    
    # Log transform execution times (often helps with skewed distributions)
    log_execution_times = np.log1p(execution_times)
    
    # Split data into train, validation, and test sets
    test_size = min(20, len(padded_features) // 5)  # Use at most 20 samples or 20% for testing
    
    X_train_val, X_test, y_train_val, y_test, paths_train_val, paths_test = train_test_split(
        padded_features, log_execution_times, file_paths, test_size=test_size, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Normalize features
    feature_dim = X_train.shape[2]
    
    # Reshape for scaling
    X_train_reshaped = X_train.reshape(-1, feature_dim)
    X_val_reshaped = X_val.reshape(-1, feature_dim)
    X_test_reshaped = X_test.reshape(-1, feature_dim)
    
    scaler = StandardScaler()
    X_train_reshaped = scaler.fit_transform(X_train_reshaped)
    X_val_reshaped = scaler.transform(X_val_reshaped)
    X_test_reshaped = scaler.transform(X_test_reshaped)
    
    # Reshape back
    X_train = X_train_reshaped.reshape(X_train.shape)
    X_val = X_val_reshaped.reshape(X_val.shape)
    X_test = X_test_reshaped.reshape(X_test.shape)
    
    # Create datasets and dataloaders
    train_dataset = GraphDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = GraphDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    test_dataset = GraphDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    batch_size = min(32, len(train_dataset))  # Ensure batch size isn't larger than dataset
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    input_size = feature_dim
    hidden_size = 128
    num_layers = 2
    
    model = LSTMExecutionTimePredictor(input_size, hidden_size, num_layers)
    print(f"Model input size: {input_size}")
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    model, train_losses, val_losses = train_model(
        train_loader, val_loader, model, criterion, optimizer, device, num_epochs=100, patience=15
    )
    
    # Evaluate on test set
    predictions, actuals, percentage_errors = evaluate_model(test_loader, model, device)
    
    # Convert log predictions back to original scale
    predictions_original = np.expm1(predictions)
    actuals_original = np.expm1(actuals)
    
    # Print test file results
    print("\nTest File Predictions:")
    for i, (pred, actual, error, path) in enumerate(zip(predictions_original, actuals_original, percentage_errors, paths_test)):
        print(f"{i+1}. {os.path.basename(path)}: Predicted={pred:.2f}ms, Actual={actual:.2f}ms, Error={error:.2f}%")
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.scatter(actuals_original, predictions_original, alpha=0.5)
    plt.plot([min(actuals_original), max(actuals_original)], [min(actuals_original), max(actuals_original)], 'r--')
    plt.xlabel('Actual Execution Time (ms)')
    plt.ylabel('Predicted Execution Time (ms)')
    plt.title('Prediction vs Actual')
    
    plt.tight_layout()
    plt.savefig('execution_time_prediction_results.png')
    plt.show()

if __name__ == "__main__":
    # Run with debug mode and limit to 100 files for initial testing
    main(debug=True, max_files=100)
