import json
import pickle
import os
import torch
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
MAX_NODES = 50
MAX_EDGES = 50
NODE_FEATURE_DIM = 67  # 24 ops + 8 access patterns + 35 scheduling
EDGE_FEATURE_DIM = 80  # 16 footprint + 64 Jacobian
DATA_DIR = "synthetic_data"
OUTPUT_PATH = "representation_halide_v2.pkl"
DEFAULT_EXEC_TIME = 1.0  # Fallback for debugging

def initialize_operation_histogram():
    """Initialize a dictionary for operation types."""
    return defaultdict(lambda: 0, {
        "add": 0, "sub": 0, "mul": 0, "div": 0, "mod": 0,
        "min": 0, "max": 0, "and": 0, "or": 0, "xor": 0,
        "not": 0, "shift_left": 0, "shift_right": 0,
        "equal": 0, "not_equal": 0, "less": 0, "less_equal": 0,
        "greater": 0, "greater_equal": 0, "select": 0,
        "load": 0, "store": 0, "call": 0, "cast": 0
    })

def initialize_scheduling_features():
    """Initialize scheduling-related features."""
    return {
        "parallelized": 0,
        "vectorized": 0,
        "unrolled": 0,
        "tiled": 0,
        "fused": 0,
        "reordered": 0,
        "split": 0,
        "inlined": 0,
        "memoized": 0,
        "loop_nest_depth": 0,
        "num_parallel_tasks": 0,
        "tile_size_x": 0,
        "tile_size_y": 0,
        "vector_width": 0,
        "unroll_factor": 0,
        "split_factors": [0] * 10,
        "reorder_indices": [0] * 5,
        "fusion_levels": [0] * 5
    })

def extract_features_from_node(node):
    """Extract features for a single node."""
    op_histogram = initialize_operation_histogram()
    access_patterns = [0] * 8
    scheduling_features = initialize_scheduling_features()
    
    # Operation features
    op_type = node.get("operation", "unknown")
    if op_type in op_histogram:
        op_histogram[op_type] = 1
    else:
        logging.debug(f"Unknown operation type: {op_type}")
    
    # Access patterns
    memory_access = node.get("memory_access", {})
    try:
        access_patterns[0] = float(memory_access.get("read_count", 0))
        access_patterns[1] = float(memory_access.get("write_count", 0))
        access_patterns[2] = float(memory_access.get("stride_x", 0))
        access_patterns[3] = float(memory_access.get("stride_y", 0))
        access_patterns[4] = float(memory_access.get("extent_x", 0))
        access_patterns[5] = float(memory_access.get("extent_y", 0))
        access_patterns[6] = float(memory_access.get("is_contiguous", 0))
        access_patterns[7] = float(memory_access.get("is_aligned", 0))
    except (TypeError, ValueError) as e:
        logging.debug(f"Invalid memory_access in node: {e}")
    
    # Scheduling features
    schedule = node.get("schedule", {})
    try:
        scheduling_features["parallelized"] = float(schedule.get("parallelized", 0))
        scheduling_features["vectorized"] = float(schedule.get("vectorized", 0))
        scheduling_features["unrolled"] = float(schedule.get("unrolled", 0))
        scheduling_features["tiled"] = float(schedule.get("tiled", 0))
        scheduling_features["fused"] = float(schedule.get("fused", 0))
        scheduling_features["reordered"] = float(schedule.get("reordered", 0))
        scheduling_features["split"] = float(schedule.get("split", 0))
        scheduling_features["inlined"] = float(schedule.get("inlined", 0))
        scheduling_features["memoized"] = float(schedule.get("memoized", 0))
        scheduling_features["loop_nest_depth"] = float(schedule.get("loop_nest_depth", 0))
        scheduling_features["num_parallel_tasks"] = float(schedule.get("num_parallel_tasks", 0))
        scheduling_features["tile_size_x"] = float(schedule.get("tile_size_x", 0))
        scheduling_features["tile_size_y"] = float(schedule.get("tile_size_y", 0))
        scheduling_features["vector_width"] = float(schedule.get("vector_width", 0))
        scheduling_features["unroll_factor"] = float(schedule.get("unroll_factor", 0))
    except (TypeError, ValueError) as e:
        logging.debug(f"Invalid schedule in node: {e}")
    
    # Combine features
    node_features = (
        [op_histogram[op] for op in sorted(op_histogram.keys())] +
        access_patterns +
        [
            scheduling_features["parallelized"],
            scheduling_features["vectorized"],
            scheduling_features["unrolled"],
            scheduling_features["tiled"],
            scheduling_features["fused"],
            scheduling_features["reordered"],
            scheduling_features["split"],
            scheduling_features["inlined"],
            scheduling_features["memoized"],
            scheduling_features["loop_nest_depth"],
            scheduling_features["num_parallel_tasks"],
            scheduling_features["tile_size_x"],
            scheduling_features["tile_size_y"],
            scheduling_features["vector_width"],
            scheduling_features["unroll_factor"],
        ] +
        scheduling_features["split_factors"] +
        scheduling_features["reorder_indices"] +
        scheduling_features["fusion_levels"]
    )
    
    return torch.tensor(node_features, dtype=torch.float32)

def extract_edge_features(edge):
    """Extract features for an edge."""
    footprint = edge.get("footprint", {})
    jacobian = edge.get("jacobian", [0] * 64)
    try:
        edge_features = (
            [
                float(footprint.get("min_x", 0)),
                float(footprint.get("max_x", 0)),
                float(footprint.get("min_y", 0)),
                float(footprint.get("max_y", 0)),
                float(footprint.get("min_z", 0)),
                float(footprint.get("max_z", 0)),
                float(footprint.get("min_t", 0)),
                float(footprint.get("max_t", 0)),
                float(footprint.get("extent_x", 0)),
                float(footprint.get("extent_y", 0)),
                float(footprint.get("extent_z", 0)),
                float(footprint.get("extent_t", 0)),
                float(footprint.get("stride_x", 0)),
                float(footprint.get("stride_y", 0)),
                float(footprint.get("stride_z", 0)),
                float(footprint.get("stride_t", 0))
            ] +
            [float(x) for x in jacobian[:64]]
        )
    except (TypeError, ValueError) as e:
        logging.debug(f"Invalid edge features: {e}")
        edge_features = [0.0] * EDGE_FEATURE_DIM
    
    return torch.tensor(edge_features, dtype=torch.float32)

def create_representation(data_dir):
    """Create dataset from JSON files in subfolders."""
    # Verify directory
    if not os.path.isdir(data_dir):
        logging.error(f"Directory {data_dir} does not exist.")
        return []
    
    json_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    
    if not json_files:
        logging.error(f"No JSON files found in {data_dir} or its subfolders.")
        return []
    
    logging.info(f"Found {len(json_files)} JSON files across subfolders in {data_dir}")
    
    dataset = []
    for file_path in json_files:
        program_id = os.path.basename(os.path.dirname(file_path))  # e.g., "program1"
        try:
            with open(file_path, "r") as f:
                prog_details = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load {file_path}: {e}")
            continue
        
        # Validate structure
        if not isinstance(prog_details, dict):
            logging.warning(f"Invalid JSON structure in {file_path}. Skipping.")
            continue
        
        # Extract execution time
        metrics = prog_details.get("Metrics", [])
        execution_time = None
        for entry in metrics:
            if entry.get("name") == "total_execution_time_ms":
                try:
                    value = float(entry.get("value", 0))
                    if value > 0.0:
                        execution_time = value
                        break
                except (TypeError, ValueError):
                    logging.debug(f"Invalid execution time in {file_path}: {entry.get('value')}")
        
        if execution_time is None:
            logging.warning(f"No valid execution time in {file_path}. Using default: {DEFAULT_EXEC_TIME}")
            execution_time = DEFAULT_EXEC_TIME
        
        # Build tree and tensors
        tree = prog_details.get("ComputationGraph", {})
        nodes = tree.get("nodes", {})
        edges = tree.get("edges", [])
        
        if not nodes:
            logging.warning(f"No nodes in {file_path}. Skipping.")
            continue
        
        node_tensor = torch.zeros((MAX_NODES, NODE_FEATURE_DIM))
        edge_tensor = torch.zeros((MAX_EDGES, EDGE_FEATURE_DIM))
        
        # Process nodes
        node_count = 0
        for node_id, node_data in nodes.items():
            if node_count >= MAX_NODES:
                break
            try:
                node_tensor[node_count] = extract_features_from_node(node_data)
                node_count += 1
            except Exception as e:
                logging.debug(f"Failed to process node {node_id} in {file_path}: {e}")
        
        # Process edges
        edge_count = 0
        for edge in edges:
            if edge_count >= MAX_EDGES:
                break
            try:
                edge_tensor[edge_count] = extract_edge_features(edge)
                edge_count += 1
            except Exception as e:
                logging.debug(f"Failed to process edge in {file_path}: {e}")
        
        # Log sample features
        if node_count > 0:
            logging.debug(f"Sample node features from {file_path}: {node_tensor[0][:10].tolist()}")
        if edge_count > 0:
            logging.debug(f"Sample edge features from {file_path}: {edge_tensor[0][:10].tolist()}")
        
        # Store representation
        dataset.append({
            "tree": tree,
            "node_tensor": node_tensor,
            "edge_tensor": edge_tensor,
            "execution_time": torch.tensor(execution_time, dtype=torch.float32),
            "file_path": file_path,
            "program_id": program_id
        })
    
    # Validate dataset
    if not dataset:
        logging.error("No valid samples created. Check JSON files for valid 'Metrics' and 'ComputationGraph'.")
        return []
    
    exec_times = torch.tensor([d["execution_time"].item() for d in dataset])
    node_features = torch.stack([d["node_tensor"] for d in dataset])
    edge_features = torch.stack([d["edge_tensor"] for d in dataset])
    
    logging.info(f"Dataset stats - Samples: {len(dataset)}, "
                 f"Exec time Mean: {exec_times.mean():.4f}, Std: {exec_times.std():.4f}, "
                 f"Min: {exec_times.min():.4f}, Max: {exec_times.max():.4f}")
    logging.info(f"Node feature variance (first 5): {node_features.std(dim=(0, 1))[:5].tolist()}")
    logging.info(f"Edge feature variance (first 5): {edge_features.std(dim=(0, 1))[:5].tolist()}")
    
    return dataset

def inspect_json(data_dir):
    """Inspect a few JSON files for debugging."""
    json_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    
    for file_path in json_files[:3]:  # Inspect first 3 files
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            metrics = data.get("Metrics", [])
            exec_time = next((entry["value"] for entry in metrics if entry.get("name") == "total_execution_time_ms"), "Missing")
            nodes = data.get("ComputationGraph", {}).get("nodes", {})
            logging.info(f"{file_path} - Exec time: {exec_time}, Nodes: {len(nodes)}")
        except Exception as e:
            logging.error(f"Failed to inspect {file_path}: {e}")

def main():
    inspect_json(DATA_DIR)
    dataset = create_representation(DATA_DIR)
    if dataset:
        with open(OUTPUT_PATH, "wb") as f:
            pickle.dump(dataset, f)
        logging.info(f"Saved dataset to {OUTPUT_PATH}")
    else:
        logging.error("Dataset creation failed.")

if __name__ == "__main__":
    main()
