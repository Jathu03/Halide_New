import os
import json
import torch
import numpy as np
from copy import deepcopy
from collections import defaultdict
import pickle
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants for padding
MAX_NODES = 50
MAX_EDGES = 50
MAX_DIMS = 8
MAX_OPS = 24
MAX_ACCESS_PATTERNS = 8
MAX_SCHED_FEATURES = 35
MAX_FOOTPRINT_LEN = 16
MAX_JACOBIAN_SIZE = 64

# Device for tensors
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_footprint(footprint):
    """
    Parse footprint into a numerical vector of min/max bounds.
    Returns a vector of length MAX_FOOTPRINT_LEN.
    """
    bounds = [0.0] * MAX_FOOTPRINT_LEN
    for i, entry in enumerate(footprint[:MAX_FOOTPRINT_LEN]):
        try:
            # Extract the value after the colon
            value = entry.split(":")[-1].strip()
            value = value.replace("(", "").replace(")", "")
            
            # Handle numerical values, fractions, or expressions
            if re.match(r'^-?\d*\.?\d+$', value):  # Integer or float
                bounds[i] = float(value)
            elif "/" in value:  # Fraction
                num, denom = value.split("/")
                bounds[i] = float(num.strip()) / float(denom.strip())
            elif value in ["0", "1"]:  # Single digits
                bounds[i] = float(value)
            else:
                # Non-numeric or symbolic value (e.g., 'all_r$1._0.max')
                logging.warning(f"Non-numeric footprint entry '{entry}', assigning 0.0")
                bounds[i] = 0.0
        except Exception as e:
            logging.warning(f"Failed to parse footprint entry '{entry}': {e}")
            bounds[i] = 0.0
    return bounds

def parse_jacobian(jacobian):
    """
    Parse load Jacobian into a flattened vector.
    Returns a vector of length MAX_JACOBIAN_SIZE.
    """
    flat_jacobian = [0.0] * MAX_JACOBIAN_SIZE
    try:
        for i, row in enumerate(jacobian[:MAX_DIMS]):
            values = row.strip().split()
            for j, val in enumerate(values[:MAX_DIMS]):
                idx = i * MAX_DIMS + j
                if idx >= MAX_JACOBIAN_SIZE:
                    logging.warning(f"Jacobian index {idx} exceeds MAX_JACOBIAN_SIZE {MAX_JACOBIAN_SIZE}")
                    continue
                try:
                    if "/" in val:
                        num, denom = val.split("/")
                        flat_jacobian[idx] = float(num) / float(denom)
                    else:
                        flat_jacobian[idx] = float(val)
                except Exception as e:
                    logging.warning(f"Failed to parse Jacobian value '{val}': {e}")
                    flat_jacobian[idx] = 0.0
    except Exception as e:
        logging.error(f"Error parsing Jacobian {jacobian}: {e}")
    return flat_jacobian

def parse_op_histogram(op_hist):
    """
    Parse operation histogram into a vector of counts.
    Returns a vector of length MAX_OPS.
    """
    op_counts = [0.0] * MAX_OPS
    op_names = [
        "Constant", "Cast", "Variable", "Param", "Add", "Sub", "Mod", "Mul", "Div",
        "Min", "Max", "EQ", "NE", "LT", "LE", "And", "Or", "Not", "Select",
        "ImageCall", "FuncCall", "SelfCall", "ExternCall", "Let"
    ]
    for entry in op_hist:
        try:
            op_name, count = entry.split(":")
            op_name = op_name.strip()
            count = float(count.strip())
            if op_name in op_names:
                idx = op_names.index(op_name)
                op_counts[idx] = count
        except Exception as e:
            logging.warning(f"Failed to parse op_histogram entry '{entry}': {e}")
    return op_counts

def parse_memory_access(access_patterns):
    """
    Parse memory access patterns into a vector.
    Returns a vector of length MAX_ACCESS_PATTERNS.
    """
    access_vec = [0.0] * MAX_ACCESS_PATTERNS
    pattern_names = ["Pointwise", "Transpose", "Broadcast", "Slice"]
    for entry in access_patterns:
        try:
            pattern, values = entry.split(":")
            pattern = pattern.strip()
            values = [float(v) for v in values.strip().split()]
            if pattern in pattern_names:
                idx = pattern_names.index(pattern)
                access_vec[idx] = sum(values) / len(values) if values else 0.0
        except Exception as e:
            logging.warning(f"Failed to parse memory access entry '{entry}': {e}")
    return access_vec

def parse_scheduling_features(sched_features):
    """
    Parse scheduling features into a vector.
    Returns a vector of length MAX_SCHED_FEATURES.
    """
    feature_vec = [0.0] * MAX_SCHED_FEATURES
    feature_names = [
        "allocation_bytes_read_per_realization", "bytes_at_production", "bytes_at_realization",
        "bytes_at_root", "bytes_at_task", "inlined_calls", "inner_parallelism",
        "innermost_bytes_at_production", "innermost_bytes_at_realization",
        "innermost_bytes_at_root", "innermost_bytes_at_task", "innermost_loop_extent",
        "innermost_pure_loop_extent", "native_vector_size", "num_productions",
        "num_realizations", "num_scalars", "num_vectors", "outer_parallelism",
        "points_computed_minimum", "points_computed_per_production",
        "points_computed_per_realization", "points_computed_total", "scalar_loads_per_scalar",
        "scalar_loads_per_vector", "unique_bytes_read_per_realization",
        "unique_bytes_read_per_task", "unique_bytes_read_per_vector",
        "unique_lines_read_per_realization", "unique_lines_read_per_task",
        "unique_lines_read_per_vector", "unrolled_loop_extent", "vector_loads_per_vector",
        "vector_size", "working_set"
    ]
    for i, fname in enumerate(feature_names):
        try:
            if fname in sched_features:
                feature_vec[i] = float(sched_features[fname])
        except Exception as e:
            logging.warning(f"Failed to parse scheduling feature '{fname}': {e}")
    return feature_vec

def build_graph_tree(nodes, edges):
    """
    Build a tree-like representation of the graph based on dependencies.
    """
    node_indices = {node["Name"]: i for i, node in enumerate(nodes)}
    tree = {"nodes": [], "children": defaultdict(list)}
    
    for node in nodes:
        tree["nodes"].append({
            "name": node["Name"],
            "index": node_indices[node["Name"]],
            "has_computation": "update" not in node["Name"]
        })
    
    for edge in edges:
        from_node = edge["From"]
        to_node = edge["To"]
        if from_node in node_indices and to_node in node_indices:
            tree["children"][node_indices[from_node]].append(node_indices[to_node])
    
    return tree, node_indices

def get_node_representation(node, node_indices):
    """
    Create a feature vector for a node.
    """
    details = node.get("Details", {})
    op_hist = parse_op_histogram(details.get("Op histogram", []))
    mem_access = parse_memory_access(details.get("Memory access patterns", []))
    sched_features = parse_scheduling_features(details.get("scheduling_feature", {}))
    
    node_vec = op_hist + mem_access + sched_features
    return node_vec

def get_edge_representation(edge):
    """
    Create a feature vector for an edge.
    """
    details = edge.get("Details", {})
    footprint = parse_footprint(details.get("Footprint", []))
    jacobian = parse_jacobian(details.get("Load Jacobians", []))
    
    edge_vec = footprint + jacobian
    return edge_vec

def create_representation(json_data, file_path):
    """
    Create a representation for a single JSON file.
    """
    try:
        prog_details = json_data.get("programming_details", {})
        nodes = prog_details.get("Nodes", []) + [
            {"Name": entry["Name"], "Details": entry["Details"]}
            for entry in prog_details.get("Schedule", [])
        ]
        edges = prog_details.get("Edges", [])
        execution_time = next(
            (float(entry["value"]) for entry in prog_details.get("Metrics", [])
             if entry["name"] == "total_execution_time_ms"),
            0.0
        )
        
        tree, node_indices = build_graph_tree(nodes, edges)
        
        node_features = [get_node_representation(node, node_indices) for node in nodes]
        while len(node_features) < MAX_NODES:
            node_features.append([0.0] * (MAX_OPS + MAX_ACCESS_PATTERNS + MAX_SCHED_FEATURES))
        node_features = node_features[:MAX_NODES]
        node_tensor = torch.tensor(node_features, dtype=torch.float32, device=DEVICE)
        
        edge_features = [get_edge_representation(edge) for edge in edges]
        while len(edge_features) < MAX_EDGES:
            edge_features.append([0.0] * (MAX_FOOTPRINT_LEN + MAX_JACOBIAN_SIZE))
        edge_features = edge_features[:MAX_EDGES]
        edge_tensor = torch.tensor(edge_features, dtype=torch.float32, device=DEVICE)
        
        exec_time_tensor = torch.tensor([execution_time], dtype=torch.float32, device=DEVICE)
        
        return {
            "tree": tree,
            "node_tensor": node_tensor,
            "edge_tensor": edge_tensor,
            "execution_time": exec_time_tensor,
            "file_path": file_path
        }
    except Exception as e:
        logging.error(f"Failed to create representation for {file_path}: {e}")
        return None

def process_directory(root_dir):
    """
    Process all JSON files in the synthetic_data directory.
    """
    dataset = []
    for program_folder in os.listdir(root_dir):
        program_path = os.path.join(root_dir, program_folder)
        if not os.path.isdir(program_path):
            continue
        for file_name in os.listdir(program_path):
            if not file_name.endswith(".json"):
                continue
            file_path = os.path.join(program_path, file_name)
            try:
                with open(file_path, "r") as f:
                    json_data = json.load(f)
                representation = create_representation(json_data, file_path)
                if representation is not None:
                    dataset.append(representation)
                else:
                    logging.warning(f"Skipped {file_path} due to processing error")
            except Exception as e:
                logging.error(f"Error reading {file_path}: {e}")
    return dataset

def save_dataset(dataset, output_path):
    """
    Save the dataset to a file.
    """
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)
    logging.info(f"Dataset saved to {output_path}")

def main():
    root_dir = "synthetic_data"
    output_path = "representation_halide.pkl"
    
    if not os.path.exists(root_dir):
        logging.error(f"Directory {root_dir} does not exist.")
        return
    
    dataset = process_directory(root_dir)
    save_dataset(dataset, output_path)
    
    logging.info(f"Processed {len(dataset)} files.")
    for rep in dataset[:5]:
        logging.info(f"File: {rep['file_path']}, Nodes: {rep['node_tensor'].shape}, "
                     f"Edges: {rep['edge_tensor'].shape}, Exec Time: {rep['execution_time'].item()}")

if __name__ == "__main__":
    main()
