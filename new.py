import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.preprocessing import MinMaxScaler
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("halide_predictor.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Constants for representation
MAX_NODES = 20  # Max number of nodes (computations)
MAX_LOOPS = 5   # Max loop depth
MAX_TRANSFORMS = 4  # Max number of transformations per computation
MAX_TAGS = 8    # Size of transformation tag vector

# Load JSON data from a file
def load_data(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in file: {file_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        return None

# Extract execution time from different possible JSON structures
def extract_execution_time(program_dict):
    """Extract execution time from various JSON structures that might be present"""
    # Case 1: Direct total_execution_time_ms field
    if isinstance(program_dict, dict) and "total_execution_time_ms" in program_dict:
        return float(program_dict["total_execution_time_ms"])
    
    # Case 2: In a list of dictionaries with "name" and "value" fields
    if isinstance(program_dict, list):
        for item in program_dict:
            if isinstance(item, dict):
                # Check for "name" and "value" structure
                if item.get("name") == "total_execution_time_ms" and "value" in item:
                    return float(item["value"])
                # Check for direct "value" field for execution time
                if "value" in item and isinstance(item["value"], (int, float)):
                    if "execution_time" in str(item).lower() or "runtime" in str(item).lower():
                        return float(item["value"])
                # Also check for execution_time or similar fields
                if "execution_time" in item:
                    # Handle case where execution_time is directly the value
                    if isinstance(item["execution_time"], (int, float)):
                        return float(item["execution_time"])
                    # Handle case where execution_time contains a value field
                    elif isinstance(item["execution_time"], dict) and "value" in item["execution_time"]:
                        return float(item["execution_time"]["value"])
                if "Details" in item and "execution_time" in item["Details"]:
                    if isinstance(item["Details"]["execution_time"], (int, float)):
                        return float(item["Details"]["execution_time"])
                    elif isinstance(item["Details"]["execution_time"], dict) and "value" in item["Details"]["execution_time"]:
                        return float(item["Details"]["execution_time"]["value"])
    
    # Case 3: In a nested structure
    if isinstance(program_dict, dict):
        # Check in performance_metrics
        if "performance_metrics" in program_dict:
            metrics = program_dict["performance_metrics"]
            if isinstance(metrics, dict):
                if "execution_time_ms" in metrics:
                    return float(metrics["execution_time_ms"])
                # Check if metrics contains a value field
                if "value" in metrics:
                    return float(metrics["value"])
            
        # Check for nested values
        for key, value in program_dict.items():
            if isinstance(value, dict) and "value" in value:
                if "time" in key.lower() or "execution" in key.lower() or "runtime" in key.lower():
                    return float(value["value"])
    
    # Case 4: Check for alternative field names
    for field in ["execution_time_ms", "runtime_ms", "execution_time", "runtime"]:
        if isinstance(program_dict, dict) and field in program_dict:
            if isinstance(program_dict[field], (int, float)):
                return float(program_dict[field])
            elif isinstance(program_dict[field], dict) and "value" in program_dict[field]:
                return float(program_dict[field]["value"])
    
    # Case 5: Recursive search for "value" in nested structures
    def recursive_search(obj, key_hint=""):
        if isinstance(obj, dict):
            # If we find a structure with a "value" field and the parent key suggests time
            if "value" in obj and isinstance(obj["value"], (int, float)) and (
                "time" in key_hint.lower() or 
                "execution" in key_hint.lower() or 
                "runtime" in key_hint.lower()):
                return float(obj["value"])
            
            # Search in all dictionary items
            for k, v in obj.items():
                result = recursive_search(v, k)
                if result is not None:
                    return result
        elif isinstance(obj, list):
            for item in obj:
                result = recursive_search(item, key_hint)
                if result is not None:
                    return result
        return None
    
    return recursive_search(program_dict)

# Create a template for Halide program representation
def get_halide_representation_template(program_dict):
    if program_dict is None:
        logger.error("Invalid program_dict (None) passed to template generator")
        return [], {}, {}
    
    # Handle both list and dict structures
    nodes = None
    edges = None
    
    try:
        if isinstance(program_dict, list):
            for item in program_dict:
                if isinstance(item, dict) and "programming_details" in item:
                    nodes = item["programming_details"].get("Nodes", [])
                    edges = item["programming_details"].get("Edges", [])
                    break
            if nodes is None:
                raise ValueError("No 'programming_details' found in JSON list")
        else:
            if "programming_details" not in program_dict:
                raise ValueError("No 'programming_details' found in JSON dict")
            nodes = program_dict["programming_details"].get("Nodes", [])
            edges = program_dict["programming_details"].get("Edges", [])
        
        if not nodes or not edges:
            raise ValueError("Empty 'Nodes' or 'Edges' in programming_details")

        node_dict = {node["Name"]: node["Details"] for node in nodes}
        comps_repr_templates = []
        comps_indices_dict = {}
        comps_placeholders_indices_dict = {}

        for comp_idx, node_name in enumerate(node_dict.keys()):
            node = node_dict[node_name]
            op_hist = {}
            
            # Process op histogram
            if "Op histogram" in node:
                for entry in node["Op histogram"]:
                    parts = entry.split(':')
                    if len(parts) == 2:
                        key, value = parts[0].strip(), int(parts[1].strip().split()[0])
                        op_hist[key] = value

            comp_repr = [
                op_hist.get("Add", 0),
                op_hist.get("Mul", 0),
                op_hist.get("Div", 0),
                op_hist.get("Min", 0),
                op_hist.get("Max", 0),
                op_hist.get("FuncCall", 0),
                len([e for e in edges if e["To"] == node_name or e["To"].startswith(node_name)]),  # Inputs
                1 if any(e["To"] == f"{node_name}.update(0)" for e in edges) else 0  # Reduction
            ]

            loop_repr = []
            c_code = f"C{comp_idx}"
            for loop_idx in range(MAX_LOOPS):
                l_code = f"{c_code}-L{loop_idx}"
                loop_repr.extend([
                    f"{l_code}-Parallel",
                    f"{l_code}-Tile",
                    f"{l_code}-TileFactor",
                    f"{l_code}-Vectorize",
                    f"{l_code}-VectorSize",
                    f"{l_code}-Unroll",
                    f"{l_code}-UnrollFactor"
                ])
            comp_repr.extend(loop_repr)

            comp_repr.append(f"{c_code}-TransformTagsStart")
            comp_repr.extend(["T"] * (MAX_TRANSFORMS * MAX_TAGS - 2))
            comp_repr.append(f"{c_code}-TransformTagsEnd")

            comps_repr_templates.append(comp_repr)
            comps_indices_dict[node_name] = comp_idx
            for j, element in enumerate(comp_repr):
                if isinstance(element, str):
                    comps_placeholders_indices_dict[element] = (comp_idx, j)

        return comps_repr_templates, comps_indices_dict, comps_placeholders_indices_dict
    
    except Exception as e:
        logger.error(f"Error creating template: {str(e)}")
        return [], {}, {}

# Fill the template with schedule-specific features
def get_halide_schedule_representation(program_dict, comps_repr_templates, comps_indices_dict, comps_placeholders_indices_dict):
    if not comps_repr_templates:
        return None, None
        
    try:
        # Extract execution time
        exec_time = extract_execution_time(program_dict)
        if exec_time is None:
            logger.warning(f"No execution time found in program data")
            return None, None

        # Handle both list and dict structures for scheduling features
        nodes = None
        sched_dict = {}
        
        if isinstance(program_dict, list):
            for item in program_dict:
                if isinstance(item, dict):
                    if "programming_details" in item:
                        nodes = item["programming_details"].get("Nodes", [])
                    elif "Name" in item and "scheduling_feature" in item.get("Details", {}):
                        sched_dict[item["Name"]] = item["Details"]["scheduling_feature"]
        else:
            nodes = program_dict.get("programming_details", {}).get("Nodes", [])
            # Try to find scheduling features in different possible locations
            if "scheduling_features" in program_dict:
                sf = program_dict["scheduling_features"]
                if isinstance(sf, dict):
                    for node_name, features in sf.items():
                        sched_dict[node_name] = features
        
        # Create node dictionary if nodes were found
        node_dict = {}
        if nodes:
            node_dict = {node["Name"]: node.get("Details", {}) for node in nodes}

        # Create a deep copy of the templates to avoid modifying the originals
        comps_repr = [list(template) for template in comps_repr_templates]

        # Process each computation
        for comp_idx, node_name in enumerate(comps_indices_dict.keys()):
            sched = sched_dict.get(node_name, {})
            c_code = f"C{comp_idx}"

            for loop_idx in range(min(MAX_LOOPS, len(sched.get("loops", [])))):
                l_code = f"{c_code}-L{loop_idx}"
                
                # Get placeholder indices
                if f"{l_code}-Parallel" in comps_placeholders_indices_dict:
                    idx = comps_placeholders_indices_dict[f"{l_code}-Parallel"][1]
                    comps_repr[comp_idx][idx] = sched.get("inner_parallelism", 1.0) > 1.0
                
                if f"{l_code}-Tile" in comps_placeholders_indices_dict:
                    idx = comps_placeholders_indices_dict[f"{l_code}-Tile"][1]
                    comps_repr[comp_idx][idx] = 1 if sched.get("unrolled_loop_extent", 1.0) > 1.0 else 0
                
                if f"{l_code}-TileFactor" in comps_placeholders_indices_dict:
                    idx = comps_placeholders_indices_dict[f"{l_code}-TileFactor"][1]
                    comps_repr[comp_idx][idx] = sched.get("unrolled_loop_extent", 1.0)
                
                if f"{l_code}-Vectorize" in comps_placeholders_indices_dict:
                    idx = comps_placeholders_indices_dict[f"{l_code}-Vectorize"][1]
                    comps_repr[comp_idx][idx] = 1 if sched.get("vector_size", 16.0) > 16.0 else 0
                
                if f"{l_code}-VectorSize" in comps_placeholders_indices_dict:
                    idx = comps_placeholders_indices_dict[f"{l_code}-VectorSize"][1]
                    comps_repr[comp_idx][idx] = sched.get("vector_size", 16.0)
                
                if f"{l_code}-Unroll" in comps_placeholders_indices_dict:
                    idx = comps_placeholders_indices_dict[f"{l_code}-Unroll"][1]
                    comps_repr[comp_idx][idx] = 1 if sched.get("unrolled_loop_extent", 1.0) > 1.0 else 0
                
                if f"{l_code}-UnrollFactor" in comps_placeholders_indices_dict:
                    idx = comps_placeholders_indices_dict[f"{l_code}-UnrollFactor"][1]
                    comps_repr[comp_idx][idx] = sched.get("unrolled_loop_extent", 1.0)

            # Handle transform tags if needed
            tags = [0] * (MAX_TRANSFORMS * MAX_TAGS)
            if f"{c_code}-TransformTagsStart" in comps_placeholders_indices_dict and f"{c_code}-TransformTagsEnd" in comps_placeholders_indices_dict:
                tags_start = comps_placeholders_indices_dict[f"{c_code}-TransformTagsStart"][1]
                tags_end = comps_placeholders_indices_dict[f"{c_code}-TransformTagsEnd"][1]
                # Fill with tags if available in the schedule
                if "transform_tags" in sched:
                    # Process transform tags (implementation depends on your data format)
                    pass
                comps_repr[comp_idx][tags_start:tags_end + 1] = tags

        # Convert to numerical values and handle strings
        padded_comps = []
        for comp in comps_repr:
            padded_comps.append([float(x) if not isinstance(x, str) else 0.0 for x in comp])
        
        # Pad to ensure consistent dimensions
        if padded_comps:
            if len(padded_comps) < MAX_NODES:
                padded_comps.extend([[0.0] * len(padded_comps[0])] * (MAX_NODES - len(padded_comps)))
            elif len(padded_comps) > MAX_NODES:
                padded_comps = padded_comps[:MAX_NODES]
            
            return torch.FloatTensor(padded_comps).unsqueeze(0), float(exec_time)
        else:
            return None, None
            
    except Exception as e:
        logger.error(f"Error creating schedule representation: {str(e)}")
        return None, None

# Load and preprocess Halide dataset
def load_halide_dataset(data_dir="Output_Programs"):
    X_data = []
    y_data = []
    valid_files = 0
    total_files = 0
    skipped_folders = 0
    processed_folders = 0

    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} does not exist")
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    # Create a dictionary to store execution times for each program
    programs_data = {}

    for program_folder in os.listdir(data_dir):
        program_path = os.path.join(data_dir, program_folder)
        if os.path.isdir(program_path):
            program_times = []
            program_reprs = []
            valid_files_in_folder = 0
            
            for filename in os.listdir(program_path):
                if filename.endswith(".json"):
                    total_files += 1
                    file_path = os.path.join(program_path, filename)
                    try:
                        program_dict = load_data(file_path)
                        if program_dict is None:
                            continue
                            
                        templates, indices_dict, placeholders_dict = get_halide_representation_template(program_dict)
                        if not templates:
                            logger.warning(f"Empty templates for {file_path}")
                            continue
                            
                        comps_tensor, exec_time = get_halide_schedule_representation(program_dict, templates, indices_dict, placeholders_dict)
                        if comps_tensor is not None and exec_time is not None:
                            program_reprs.append(comps_tensor.squeeze(0).numpy())
                            program_times.append(exec_time)
                            valid_files += 1
                            valid_files_in_folder += 1
                        else:
                            logger.warning(f"No execution time found in {file_path}")
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {str(e)}")
                        continue
            
            # Only calculate speedup if we have at least 2 valid schedules for this program
            if valid_files_in_folder >= 2:
                baseline_time = max(program_times)  # Max time as baseline
                programs_data[program_folder] = {
                    "representations": program_reprs,
                    "times": program_times,
                    "baseline": baseline_time
                }
                processed_folders += 1
            else:
                skipped_folders += 1
                logger.warning(f"Skipping {program_folder}: only {valid_files_in_folder} valid files (need at least 2)")

    # Calculate speedups and collect data
    total_samples = 0
    for program, data in programs_data.items():
        baseline_time = data["baseline"]
        for i, (repr_data, time) in enumerate(zip(data["representations"], data["times"])):
            speedup = baseline_time / time
            X_data.append(repr_data)
            y_data.append(speedup)
            total_samples += 1

    logger.info(f"Total files processed: {total_files}")
    logger.info(f"Valid files with execution times: {valid_files}")
    logger.info(f"Processed program folders: {processed_folders}")
    logger.info(f"Skipped program folders: {skipped_folders}")
    logger.info(f"Total samples for model training: {total_samples}")

    if not X_data:
        logger.error("No valid data loaded - check your JSON files structure")
        raise ValueError("No valid data loaded from dataset folder")
    
    X_data = np.array(X_data)  # Shape: (samples, MAX_NODES, features)
    y_data = np.array(y_data).reshape(-1, 1)  # Shape: (samples, 1)

    # Normalize
    scaler_X = MinMaxScaler()
    X_flat = X_data.reshape(-1, X_data.shape[-1])
    X_normalized = scaler_X.fit_transform(X_flat).reshape(X_data.shape)
    scaler_y = MinMaxScaler()
    y_normalized = scaler_y.fit_transform(y_data)

    logger.info(f"Dataset shape: X={X_normalized.shape}, y={y_normalized.shape}")
    return X_normalized, y_normalized, scaler_X, scaler_y

# LSTM Model
class LSTMSpeedupPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMSpeedupPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Training function
def train_model(model, X_train, y_train, epochs=100, batch_size=8):
    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

    best_loss = float('inf')
    early_stop_count = 0
    early_stop_patience = 15

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stop_count = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_halide_predictor.pth')
        else:
            early_stop_count += 1
        
        if early_stop_count >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load the best model
    model.load_state_dict(torch.load('best_halide_predictor.pth'))
    return model

# Predict speedup for test data
def predict_halide_speedup(data_dir="Output_Programs"):
    try:
        # Load and preprocess data
        X_data, y_data, scaler_X, scaler_y = load_halide_dataset(data_dir)
        logger.info(f"Loaded {X_data.shape[0]} samples with shape {X_data.shape}")

        if len(X_data) < 10:  # Require at least 10 samples for meaningful training
            logger.error("Insufficient data for training (need at least 10 samples)")
            return

        # Train-test split (80% train, 20% test)
        split_idx = int(0.8 * len(X_data))
        X_train, X_test = X_data[:split_idx], X_data[split_idx:]
        y_train, y_test = y_data[:split_idx], y_data[split_idx:]
        logger.info(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

        # Initialize and train model
        input_size = X_data.shape[2]
        model = LSTMSpeedupPredictor(input_size).to(device)
        model = train_model(model, X_train, y_train, epochs=100)

        # Predict on test set
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            y_test_tensor = torch.FloatTensor(y_test).to(device)
            y_pred = model(X_test_tensor)
            test_loss = nn.MSELoss()(y_pred, y_test_tensor)
            logger.info(f"Test Loss (Normalized): {test_loss.item():.4f}")

            # Denormalize predictions and actual values
            y_pred_denorm = scaler_y.inverse_transform(y_pred.cpu().numpy())
            y_test_denorm = scaler_y.inverse_transform(y_test)
            rmse = np.sqrt(np.mean((y_pred_denorm - y_test_denorm) ** 2))
            logger.info(f"Test RMSE (Speedup): {rmse:.2f}")
            
            # Calculate R² score
            y_mean = np.mean(y_test_denorm)
            ss_tot = np.sum((y_test_denorm - y_mean) ** 2)
            ss_res = np.sum((y_test_denorm - y_pred_denorm) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            logger.info(f"R² Score: {r2:.4f}")

            # Compute speedups for test data
            logger.info("\nSpeedup Predictions for Test Data:")
            for i in range(min(10, len(y_test_denorm))):  # Show first 10 test samples
                actual_speedup = y_test_denorm[i][0]
                pred_speedup = y_pred_denorm[i][0]
                error_pct = abs(actual_speedup - pred_speedup) / actual_speedup * 100
                logger.info(f"Test Sample {i+1}: Actual Speedup: {actual_speedup:.2f}x, "
                      f"Predicted Speedup: {pred_speedup:.2f}x, Error: {error_pct:.2f}%")
                
        # Save the final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'input_size': input_size,
            'max_nodes': MAX_NODES
        }, 'halide_speedup_predictor.pth')
        logger.info("Model saved as 'halide_speedup_predictor.pth'")
        
    except Exception as e:
        logger.error(f"Error in predict_halide_speedup: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    logger.info("Starting Halide Speedup Prediction")
    data_dir = "Output_Programs"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    logger.info(f"Using data directory: {data_dir}")
    predict_halide_speedup(data_dir=data_dir)
