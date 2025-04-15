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
NODE_FEATURE_DIM = 70  # Expanded: 24 ops + 8 access patterns + 35 scheduling + 3 graph props
EDGE_FEATURE_DIM = 80  # 16 footprint + 64 Jacobian
DATA_DIR = "synthetic_data"
OUTPUT_PATH = "representation_halide_v2.pkl"

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
    }

def compute_graph_properties(tree, node_id):
    """Compute graph properties: degree, depth, is_leaf."""
    degree = len(tree.get("dependencies", []))
    depth = 0
    current = node_id
    while current in tree.get("parent", {}):
        depth += 1
        current = tree["parent"][current]
    is_leaf = 1 if degree == 0 else 0
    return degree, depth, is_leaf

def extract_features_from_node(node, tree, node_id):
    """Extract features for a single node."""
    op_histogram = initialize_operation_histogram()
    access_patterns = [0] * 8
    scheduling_features = initialize_scheduling_features()
    
    # Operation features
    op_type = node.get("operation", "unknown")
    if op_type in op_histogram:
        op_histogram[op_type] = 1
    
    # Access patterns (simplified)
    memory_access = node.get("memory_access", {})
    access_patterns[0] = memory_access.get("read_count", 0)
    access_patterns[1] = memory_access.get("write_count", 0)
    access_patterns[2] = memory_access.get("stride_x", 0)
    access_patterns[3] = memory_access.get("stride_y", 0)
    access_patterns[4] = memory_access.get("extent_x", 0)
    access_patterns[5] = memory_access.get("extent_y", 0)
    access_patterns[6] = memory_access.get("is_contiguous", 0)
    access_patterns[7] = memory_access.get("is_aligned", 0)
    
    # Scheduling features
    schedule = node.get("schedule", {})
    scheduling_features["parallelized"] = schedule.get("parallelized", 0)
    scheduling_features["vectorized"] = schedule.get("vectorized", 0)
    scheduling_features["unrolled"] = schedule.get("unrolled", 0)
    scheduling_features["tiled"] = schedule.get("tiled", 0)
    scheduling_features["fused"] = schedule.get("fused", 0)
    scheduling_features["reordered"] = schedule.get("reordered", 0)
    scheduling_features["split"] = schedule.get("split", 0)
    scheduling_features["inlined"] = schedule.get("inlined", 0)
    scheduling_features["memoized"] = schedule.get("memoized", 0)
    scheduling_features["loop_nest_depth"] = schedule.get("loop_nest_depth", 0)
    scheduling_features["num_parallel_tasks"] = schedule.get("num_parallel_tasks", 0)
    scheduling_features["tile_size_x"] = schedule.get("tile_size_x", 0)
    scheduling_features["tile_size_y"] = schedule.get("tile_size_y", 0)
    scheduling_features["vector_width"] = schedule.get("vector_width", 0)
    scheduling_features["unroll_factor"] = schedule.get("unroll_factor", 0)
    
    # Graph properties
    degree, depth, is_leaf = compute_graph_properties(tree, node_id)
    
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
        scheduling_features["fusion_levels"] +
        [degree, depth, is_leaf]
    )
    
    return torch.tensor(node_features, dtype=torch.float32)

def extract_edge_features(edge):
    """Extract features for an edge."""
    footprint = edge.get("footprint", {})
    jacobian = edge.get("jacobian", [0] * 64)
    edge_features = (
        [
            footprint.get("min_x", 0),
            footprint.get("max_x", 0),
            footprint.get("min_y", 0),
            footprint.get("max_y", 0),
            footprint.get("min_z", 0),
            footprint.get("max_z", 0),
            footprint.get("min_t", 0),
            footprint.get("max_t", 0),
            footprint.get("extent_x", 0),
            footprint.get("extent_y", 0),
            footprint.get("extent_z", 0),
            footprint.get("extent_t", 0),
            footprint.get("stride_x", 0),
            footprint.get("stride_y", 0),
            footprint.get("stride_z", 0),
            footprint.get("stride_t", 0)
        ] +
        jacobian[:64]
    )
    return torch.tensor(edge_features, dtype=torch.float32)

def create_representation(data_dir):
    """Create dataset from JSON files."""
    dataset = []
    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(data_dir, filename)
        try:
            with open(file_path, "r") as f:
                prog_details = json.load(f)
        except Exception as e:
            logging.error(f"Failed to load {file_path}: {e}")
            continue
        
        # Extract execution time
        execution_time = next(
            (float(entry["value"]) for entry in prog_details.get("Metrics", [])
             if entry["name"] == "total_execution_time_ms" and float(entry["value"]) > 0.0),
            None
        )
        if execution_time is None:
            logging.warning(f"No valid execution time in {file_path}. Skipping.")
            continue
        
        # Build tree and tensors
        tree = prog_details.get("ComputationGraph", {})
        nodes = tree.get("nodes", {})
        edges = tree.get("edges", [])
        
        node_tensor = torch.zeros((MAX_NODES, NODE_FEATURE_DIM))
        edge_tensor = torch.zeros((MAX_EDGES, EDGE_FEATURE_DIM))
        
        # Process nodes
        for i, (node_id, node_data) in enumerate(nodes.items()):
            if i >= MAX_NODES:
                break
            node_tensor[i] = extract_features_from_node(node_data, tree, node_id)
        
        # Process edges
        for i, edge in enumerate(edges):
            if i >= MAX_EDGES:
                break
            edge_tensor[i] = extract_edge_features(edge)
        
        # Store representation
        dataset.append({
            "tree": tree,
            "node_tensor": node_tensor,
            "edge_tensor": edge_tensor,
            "execution_time": torch.tensor(execution_time, dtype=torch.float32),
            "file_path": file_path
        })
    
    # Validate dataset
    if not dataset:
        logging.error("No valid samples created. Check JSON files.")
        return []
    
    exec_times = torch.tensor([d["execution_time"].item() for d in dataset])
    logging.info(f"Dataset stats - Samples: {len(dataset)}, Exec time Mean: {exec_times.mean():.4f}, "
                 f"Std: {exec_times.std():.4f}, Min: {exec_times.min():.4f}, Max: {exec_times.max():.4f}")
    
    return dataset

def main():
    dataset = create_representation(DATA_DIR)
    if dataset:
        with open(OUTPUT_PATH, "wb") as f:
            pickle.dump(dataset, f)
        logging.info(f"Saved dataset to {OUTPUT_PATH}")
    else:
        logging.error("Dataset creation failed.")

if __name__ == "__main__":
    main()
