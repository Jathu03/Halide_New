import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import random
from pathlib import Path

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Define fixed features (same as C++)
FIXED_FEATURES = [
    "cache_hits", "cache_misses", "execution_time_ms", "sched_num_realizations",
    "sched_num_productions", "sched_points_computed_total", "sched_innermost_loop_extent",
    "sched_inner_parallelism", "sched_outer_parallelism", "sched_bytes_at_realization",
    "sched_bytes_at_production", "sched_bytes_at_root", "sched_unique_bytes_read_per_realization",
    "sched_working_set", "sched_vector_size", "sched_num_vectors", "sched_num_scalars",
    "sched_bytes_at_task", "sched_working_set_at_task", "sched_working_set_at_production",
    "sched_working_set_at_realization", "sched_working_set_at_root", "total_parallelism",
    "scheduling_count", "total_bytes_at_production", "total_vectors", "computation_efficiency",
    "memory_pressure", "memory_utilization_ratio", "bytes_processing_rate", "bytes_per_parallelism",
    "bytes_per_vector", "nodes_count", "edges_count", "node_edge_ratio", "nodes_per_schedule",
    "op_diversity",
    "op_add", "op_sub", "op_mul", "op_div", "op_mod", "op_eq", "op_ne", "op_lt", "op_le",
    "op_or", "op_and", "op_not", "op_min", "op_max", "op_constant", "op_variable",
    "op_funccall", "op_imagecall", "op_externcall", "op_let", "op_param",
    "memory_transpose_0", "memory_transpose_1", "memory_transpose_2", "memory_transpose_3",
    "memory_slice_0", "memory_slice_1", "memory_slice_2", "memory_slice_3",
    "memory_broadcast_0", "memory_broadcast_1", "memory_broadcast_2", "memory_broadcast_3",
    "memory_pointwise_0", "memory_pointwise_1", "memory_pointwise_2", "memory_pointwise_3"
]

# Define low-importance features to drop (same as C++)
LOW_IMPORTANCE_FEATURES = [
    "op_cast", "op_selfcall", "memory_pointwise_1", "memory_transpose_1", "memory_broadcast_1",
    "memory_slice_1", "op_select", "op_not", "op_and", "op_ne", "op_mod", "memory_pointwise_2",
    "memory_broadcast_2", "memory_slice_2", "memory_transpose_2", "op_externcall", "op_imagecall",
    "op_param", "memory_pointwise_3", "memory_transpose_3", "op_sub", "memory_pointwise_0", "op_let"
]

# Define skewed features for log transformation
SKEWED_FEATURES = [
    "cache_hits", "bytes_processing_rate", "sched_bytes_at_task", "computation_efficiency"
]

def extract_features(json_data):
    features = {}

    # Extract global features
    for child in json_data.get("children", []):
        if child.get("name") == "Global Features":
            features["cache_hits"] = child.get("cache_hits", 0.0)
            features["cache_misses"] = child.get("cache_misses", 0.0)
            features["execution_time_ms"] = child.get("execution_time_ms", 0.0)
            break

    # Extract op_histogram features
    op_histogram = {}
    for node in json_data.get("children", []):
        if "op_histogram" in node:
            for op, count in node["op_histogram"].items():
                op_lower = op.lower()
                op_histogram[op_lower] = op_histogram.get(op_lower, 0) + float(count)
    for op, count in op_histogram.items():
        features[f"op_{op}"] = count

    # Extract memory patterns
    memory_patterns = {}
    for node in json_data.get("children", []):
        if "memory_patterns" in node:
            for pattern, values in node["memory_patterns"].items():
                curr_values = [0.0] * 4
                for i, val in enumerate(values[:4]):
                    curr_values[i] = float(val)
                if pattern not in memory_patterns:
                    memory_patterns[pattern] = [0.0] * 4
                for i in range(4):
                    memory_patterns[pattern][i] += curr_values[i]
    for pattern, values in memory_patterns.items():
        pattern_lower = pattern.lower()
        for i, val in enumerate(values):
            features[f"memory_{pattern_lower}_{i}"] = val

    # Extract scheduling features
    scheduling_keys = [
        "num_realizations", "num_productions", "points_computed_total", "innermost_loop_extent",
        "inner_parallelism", "outer_parallelism", "bytes_at_realization", "bytes_at_production",
        "bytes_at_root", "unique_bytes_read_per_realization", "working_set", "vector_size",
        "num_vectors", "num_scalars", "bytes_at_task", "working_set_at_task", "working_set_at_production",
        "working_set_at_realization", "working_set_at_root"
    ]
    scheduling_sums = {key: 0.0 for key in scheduling_keys}
    node_count = 0
    for node in json_data.get("children", []):
        if "scheduling" in node:
            node_count += 1
            for key in scheduling_keys:
                scheduling_sums[key] += float(node["scheduling"].get(key, 0.0))
    for key in scheduling_keys:
        if key in ["inner_parallelism", "outer_parallelism"]:
            features[f"sched_{key}"] = scheduling_sums[key] / node_count if node_count > 0 else 0.0
        else:
            features[f"sched_{key}"] = scheduling_sums[key]

    # Derived features
    features["total_parallelism"] = features["sched_inner_parallelism"] + features["sched_outer_parallelism"]
    features["scheduling_count"] = features["sched_num_realizations"] + features["sched_num_productions"]
    features["total_bytes_at_production"] = features["sched_bytes_at_production"]
    features["total_vectors"] = features["sched_num_vectors"]
    features["computation_efficiency"] = (
        features["sched_points_computed_total"] / features["sched_bytes_at_realization"]
        if features["sched_bytes_at_realization"] != 0 else 0.0
    )
    features["memory_pressure"] = (
        features["sched_working_set"] / features["sched_bytes_at_root"]
        if features["sched_bytes_at_root"] != 0 else 0.0
    )
    features["memory_utilization_ratio"] = (
        features["sched_unique_bytes_read_per_realization"] / features["sched_bytes_at_task"]
        if features["sched_bytes_at_task"] != 0 else 0.0
    )
    features["bytes_processing_rate"] = (
        features["sched_bytes_at_realization"] / features["execution_time_ms"]
        if features["execution_time_ms"] != 0 else 0.0
    )
    features["bytes_per_parallelism"] = (
        features["sched_bytes_at_task"] / features["total_parallelism"]
        if features["total_parallelism"] != 0 else 0.0
    )
    features["bytes_per_vector"] = (
        features["sched_bytes_at_realization"] / features["sched_num_vectors"]
        if features["sched_num_vectors"] != 0 else 0.0
    )
    nodes_count = len(json_data.get("children", []))
    edges_count = sum(len(node.get("children", [])) for node in json_data.get("children", []))
    features["nodes_count"] = nodes_count
    features["edges_count"] = edges_count
    features["node_edge_ratio"] = nodes_count / (edges_count + 1) if edges_count + 1 != 0 else 0.0
    features["nodes_per_schedule"] = (
        nodes_count / features["scheduling_count"] if features["scheduling_count"] != 0 else 0.0
    )
    features["op_diversity"] = sum(1 for key, val in features.items() if key.startswith("op_") and val > 0)

    # Create fixed-length feature vector
    fixed_features = {key: features.get(key, 0.0) for key in FIXED_FEATURES}
    return fixed_features

def load_json_files(main_dir):
    file_paths = []
    for root, _, files in os.walk(main_dir):
        for file in files:
            if file == "tree_representation.json":
                file_paths.append(os.path.join(root, file))
    return file_paths

def prepare_data_for_model(main_dir):
    file_paths = load_json_files(main_dir)
    if not file_paths:
        print(f"No tree_representation.json files found in {main_dir}")
        return None, None, None, None, None

    invalid_files = []
    all_features = []
    file_names = []

    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            features = extract_features(json_data)
            if features["execution_time_ms"] <= 0 or not np.isfinite(features["execution_time_ms"]):
                print(f"Invalid execution time in {file_path}")
                invalid_files.append(file_path)
                continue
            all_features.append(features)
            file_names.append(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            invalid_files.append(file_path)

    if not all_features:
        print("No valid data after processing")
        return None, None, None, None, None

    # Save invalid files log
    with open(os.path.join(main_dir, "invalid_files_log.txt"), "w") as f:
        f.write("Files with invalid execution times or errors (skipped):\n")
        for file in invalid_files:
            f.write(f"{file}\n")

    # Create DataFrame
    df = pd.DataFrame(all_features)

    # Log transform skewed features
    for feature in SKEWED_FEATURES:
        if feature in df.columns:
            df[f"log_{feature}"] = np.log1p(df[feature])
            df = df.drop(columns=[feature])

    # Drop low-importance features
    features_to_drop = [f for f in LOW_IMPORTANCE_FEATURES if f in df.columns]
    df = df.drop(columns=features_to_drop)

    # Separate features and target
    X = df.drop(columns=["execution_time_ms"])
    y = df["execution_time_ms"]

    # Scale features and target
    scaler_X = RobustScaler()
    X_scaled = scaler_X.fit_transform(X)
    scaler_y = RobustScaler()
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()

    # Convert to tensors
    sequence_length = 3  # Same as C++
    seq_input_size = X_scaled.shape[1]
    scalar_input_size = seq_input_size

    sequences = []
    for i in range(len(X_scaled)):
        seq = np.tile(X_scaled[i], (sequence_length, 1))  # Repeat for sequence_length
        sequences.append(seq)
    sequences = np.array(sequences)
    scalar_inputs = X_scaled
    y = np.log1p(y)  # Log transform target

    # Convert to tensors
    sequences_tensor = torch.tensor(sequences, dtype=torch.float32)
    scalar_tensor = torch.tensor(scalar_inputs, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return sequences_tensor, scalar_tensor, y_tensor, file_names, scaler_y, scaler_X

class SimpleLSTMModel(nn.Module):
    def __init__(self, seq_input_size, scalar_input_size, hidden_size=128, num_layers=2, num_heads=4):
        super(SimpleLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM for sequence input
        self.lstm = nn.LSTM(seq_input_size, hidden_size, num_layers, batch_first=True)
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size + scalar_input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, seq_input, scalar_input):
        # LSTM forward
        batch_size = seq_input.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(seq_input.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(seq_input.device)
        lstm_out, _ = self.lstm(seq_input, (h0, c0))

        # Attention
        lstm_out = lstm_out.transpose(0, 1)  # [seq_len, batch, hidden]
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.transpose(0, 1)  # [batch, seq_len, hidden]
        attn_out = attn_out[:, -1, :]  # Take last timestep

        # Concatenate with scalar input
        combined = torch.cat((attn_out, scalar_input), dim=1)
        out = self.fc1(combined)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

def custom_loss(y_pred, y_true, feature_indices, feature_importances):
    mse_loss = nn.MSELoss()(y_pred, y_true)
    return mse_loss  # Simplified; add feature-based loss if needed

def train_model(model, train_loader, val_loader, criterion, optimizer, feature_indices, feature_importances, num_epochs=1000, patience=50, accumulation_steps=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for i, (seq_inputs, scalar_inputs, targets) in enumerate(train_loader):
            seq_inputs, scalar_inputs, targets = seq_inputs.to(device), scalar_inputs.to(device), targets.to(device)
            outputs = model(seq_inputs, scalar_inputs)
            loss = criterion(outputs.squeeze(), targets, feature_indices, feature_importances)
            loss = loss / accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            train_loss += loss.item() * accumulation_steps
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for seq_inputs, scalar_inputs, targets in val_loader:
                seq_inputs, scalar_inputs, targets = seq_inputs.to(device), scalar_inputs.to(device), targets.to(device)
                outputs = model(seq_inputs, scalar_inputs)
                loss = criterion(outputs.squeeze(), targets, feature_indices, feature_importances)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return train_loss, val_loss

def evaluate_model(model, test_sequences, test_scalar, y_test, y_scaler, test_file_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    y_pred = []
    with torch.no_grad():
        for i in range(len(test_sequences)):
            seq_input = test_sequences[i:i+1].to(device)
            scalar_input = test_scalar[i:i+1].to(device)
            output = model(seq_input, scalar_input)
            y_pred.append(output.item())
    y_pred = np.array(y_pred)
    y_pred_actual = np.expm1(y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten())
    y_test_actual = np.expm1(y_scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten())
    return y_test_actual, y_pred_actual

def main(main_dir):
    # Load and prepare data
    sequences_tensor, scalar_tensor, y_tensor, file_names, y_scaler, scaler_X_scalar = prepare_data_for_model(main_dir)
    if sequences_tensor is None:
        return None

    # Split data
    indices = list(range(len(sequences_tensor)))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    train_sequences = sequences_tensor[train_idx]
    train_scalar = scalar_tensor[train_idx]
    train_y = y_tensor[train_idx]
    test_sequences = sequences_tensor[test_idx]
    test_scalar = scalar_tensor[test_idx]
    test_y = y_tensor[test_idx]
    test_file_names = [file_names[i] for i in test_idx]

    # Create data loaders
    train_dataset = TensorDataset(train_sequences, train_scalar, train_y)
    test_dataset = TensorDataset(test_sequences, test_scalar, test_y)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model
    seq_input_size = sequences_tensor.shape[2]
    scalar_input_size = scalar_tensor.shape[1]
    model = SimpleLSTMModel(seq_input_size, scalar_input_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    feature_indices = list(range(scalar_input_size))  # Placeholder
    feature_importances = torch.ones(scalar_input_size)  # Placeholder
    custom_loss_fn = custom_loss

    # Train model
    print("Building and training Simple LSTM model...")
    train_losses, val_losses = train_model(
        model, train_loader, test_loader,
        custom_loss_fn, optimizer, feature_indices, feature_importances,
        num_epochs=1000, patience=50, accumulation_steps=2
    )

    if train_losses is None or val_losses is None:
        print("Training failed due to invalid values")
        return None

    # Save the trained model for LibTorch
    model.eval()
    try:
        # Use realistic example inputs from test_loader
        example_seq, example_scalar = next(iter(test_loader))[0:2]
        example_seq = example_seq[:1].to(torch.float32)  # Shape: [1, seq_len, features]
        example_scalar = example_scalar[:1].to(torch.float32)  # Shape: [1, scalar_features]

        # Trace the model
        traced_model = torch.jit.trace(model, (example_seq, example_scalar))
        traced_model.save("model.pt")
        print("Model saved to model.pt using torch.jit.trace")
    except Exception as e:
        print(f"Tracing failed: {e}")
        try:
            # Fallback to torch.jit.script
            scripted_model = torch.jit.script(model)
            scripted_model.save("model.pt")
            print("Model saved to model.pt using torch.jit.script")
        except Exception as e:
            print(f"Scripting also failed: {e}")
            return None

    # Save scaler parameters for C++
    scaler_params = {
        "X_scalar_center": scaler_X_scalar.center_.tolist(),
        "X_scalar_scale": scaler_X_scalar.scale_.tolist(),
        "y_center": y_scaler.center_.tolist(),
        "y_scale": y_scaler.scale_.tolist()
    }
    with open("scaler_params.json", "w") as f:
        json.dump(scaler_params, f)
    print("Scaler parameters saved to scaler_params.json")

    # Evaluate model
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(
        model, test_sequences, test_scalar, test_y,
        y_scaler, test_file_names
    )

    print(f"\nSummary for Comparison:")
    print(f"Model: SimpleLSTM")

    return model, y_scaler, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "Tree_Output"
    main(main_dir)
