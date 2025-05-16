import os
import json
import numpy as np
from pathlib import Path
import uuid

def extract_node_features(node):
    """
    Extract features from a node, including its properties and stage features.
    Returns a flattened feature vector.
    """
    features = []
    
    # Basic node properties
    features.append(1 if node.get('input', False) else 0)
    features.append(1 if node.get('output', False) else 0)
    features.append(1 if node.get('pointwise', False) else 0)
    features.append(1 if node.get('boundary_condition', False) else 0)
    features.append(1 if node.get('wrapper', False) else 0)
    
    # Region computed/required (number of dimensions)
    features.append(len(node.get('region_computed', [])))
    features.append(len(node.get('region_required', [])))
    
    # Stage features
    for stage in node.get('stages', []):
        # Loop features
        features.append(len(stage.get('loops', [])))
        
        # Pipeline features: memory access patterns
        for dtype in ['Float', 'UInt32']:
            for pattern in ['Broadcast', 'Pointwise', 'Slice', 'Transpose']:
                mem_pattern = stage.get('pipeline_features', {}).get('memory_access_patterns', {}).get(dtype, {}).get(pattern, [0]*4)
                features.extend(mem_pattern)
        
        # Pipeline features: operation histogram
        for dtype in ['Float', 'UInt32']:
            op_hist = stage.get('pipeline_features', {}).get('op_histogram', {}).get(dtype, {})
            for op in ['Constant', 'Cast', 'Variable', 'Param', 'Add', 'Sub', 'Mod', 'Mul', 'Div', 
                      'Min', 'Max', 'EQ', 'NE', 'LT', 'LE', 'And', 'Or', 'Not', 'Select', 
                      'ImageCall', 'FuncCall', 'SelfCall', 'ExternCall', 'Let']:
                features.append(op_hist.get(op, 0))
        
        # Schedule features
        sched_features = stage.get('schedule_features', {})
        for key in ['allocation_bytes_read_per_realization', 'bytes_at_production', 'bytes_at_realization',
                   'bytes_at_root', 'bytes_at_task', 'inlined_calls', 'inner_parallelism',
                   'innermost_bytes_at_production', 'innermost_bytes_at_realization', 'innermost_bytes_at_root',
                   'innermost_bytes_at_task', 'innermost_loop_extent', 'innermost_pure_loop_extent',
                   'native_vector_size', 'num_productions', 'num_realizations', 'num_scalars', 'num_vectors',
                   'outer_parallelism', 'points_computed_minimum', 'points_computed_per_production',
                   'points_computed_per_realization', 'points_computed_total', 'scalar_loads_per_scalar',
                   'scalar_loads_per_vector', 'unique_bytes_read_per_realization', 'unique_bytes_read_per_task',
                   'unique_bytes_read_per_vector', 'unique_lines_read_per_realization', 'unique_lines_read_per_task',
                   'unique_lines_read_per_vector', 'unrolled_loop_extent', 'vector_loads_per_vector', 'vector_size',
                   'working_set', 'working_set_at_production', 'working_set_at_realization', 'working_set_at_root',
                   'working_set_at_task']:
            features.append(sched_features.get(key, 0.0))
    
    return features

def extract_edge_features(edge):
    """
    Extract features from an edge, including bounds and load jacobians.
    Returns a flattened feature vector.
    """
    features = []
    
    # Number of bounds
    features.append(len(edge.get('bounds', [])))
    
    # Load jacobians
    for jacobian in edge.get('load_jacobians', []):
        features.append(jacobian.get('count', 0))
        matrix = jacobian.get('matrix', [])
        # Flatten the matrix
        for row in matrix:
            features.extend(row)
        # Pad to a fixed size (assume max 5x5 matrix for consistency)
        max_matrix_elements = 25
        while len(features) % max_matrix_elements != 0:
            features.append(0.0)
    
    return features

def process_json_file(file_path):
    """
    Process a single JSON file and extract features.
    Returns a sequence of node features, edge features, and execution time.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Get execution time
        execution_time = data.get('without_extern', {}).get('global_features', {}).get('execution_time_ms', 0)
        if execution_time <= 0:
            return None  # Skip files with invalid execution time
        
        # Extract node features
        nodes = data.get('without_extern', {}).get('nodes', [])
        node_features = [extract_node_features(node) for node in nodes]
        
        # Extract edge features
        edges = data.get('without_extern', {}).get('edges', [])
        edge_features = [extract_edge_features(edge) for edge in edges]
        
        # If no features, skip the file
        if not node_features and not edge_features:
            print(f"Skipping {file_path}: No nodes or edges found.")
            return None
        
        # Combine all features to find the maximum length
        all_features = node_features + edge_features
        max_feature_len = max(len(f) for f in all_features) if all_features else 0
        
        # Pad all feature vectors to the maximum length
        node_features = [f + [0.0] * (max_feature_len - len(f)) for f in node_features]
        edge_features = [f + [0.0] * (max_feature_len - len(f)) for f in edge_features]
        
        # Combine node and edge features into a sequence
        sequence = node_features + edge_features
        
        # Pad sequence to a fixed length (e.g., max 100 steps)
        max_sequence_len = 100
        if len(sequence) < max_sequence_len:
            sequence.extend([[0.0] * max_feature_len] * (max_sequence_len - len(sequence)))
        elif len(sequence) > max_sequence_len:
            sequence = sequence[:max_sequence_len]
        
        return np.array(sequence, dtype=np.float32), np.array([execution_time], dtype=np.float32)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_dataset(graph_output_dir, output_file):
    """
    Create a dataset from all JSON files in the Graph_Output directory.
    Saves the dataset as a .npz file with sequences and execution times.
    """
    sequences = []
    execution_times = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(graph_output_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                result = process_json_file(file_path)
                if result is not None:
                    sequence, exec_time = result
                    sequences.append(sequence)
                    execution_times.append(exec_time)
    
    # Convert to NumPy arrays
    if not sequences:
        print("No valid data found. Dataset creation aborted.")
        return
    
    sequences = np.array(sequences)
    execution_times = np.array(execution_times)
    
    # Save dataset
    np.savez(output_file, sequences=sequences, execution_times=execution_times)
    print(f"Dataset saved to {output_file}")
    print(f"Number of valid samples: {len(sequences)}")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Execution times shape: {execution_times.shape}")

if __name__ == "__main__":
    # Define input and output paths
    graph_output_dir = Path("Graph_Output")
    output_file = Path("lstm_dataset.npz")
    
    # Create dataset
    create_dataset(graph_output_dir, output_file)
