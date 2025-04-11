import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import re

def extract_numerical_features(data: Dict[str, Any]) -> np.ndarray:
    """
    Extract numerical features from a dictionary, handling lists and nested structures.
    Returns a flattened numpy array of numerical values.
    """
    numerical_features = []

    def parse_value(value: Any):
        if isinstance(value, (int, float)):
            numerical_features.append(float(value))
        elif isinstance(value, list):
            for item in value:
                parse_value(item)
        elif isinstance(value, dict):
            for k, v in value.items():
                parse_value(v)
        elif isinstance(value, str):
            # Try to extract numbers from strings (e.g., "1/8" -> 0.125)
            try:
                if '/' in value:
                    num, denom = map(float, value.split('/'))
                    numerical_features.append(num / denom)
                else:
                    numerical_features.append(float(value))
            except (ValueError, TypeError):
                pass  # Skip non-numerical strings

    parse_value(data)
    return np.array(numerical_features, dtype=np.float32)

def process_sequential_data(data: List[str]) -> np.ndarray:
    """
    Process sequential data (e.g., memory access patterns, footprints) into a numerical sequence.
    Returns a padded numpy array for consistent length.
    """
    sequence = []
    for item in data:
        # Extract numbers from strings using regex
        numbers = re.findall(r'-?\d*\.?\d+', item)
        sequence.extend([float(n) for n in numbers])
    # Pad or truncate to a fixed length (e.g., 100) for consistency
    max_len = 100
    if len(sequence) < max_len:
        sequence.extend([0.0] * (max_len - len(sequence)))
    else:
        sequence = sequence[:max_len]
    return np.array(sequence, dtype=np.float32)

def create_graph_representation(program_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a graph representation for a single program.
    Returns a dictionary with nodes, edges, features, and target.
    """
    nodes = program_data['programming_details']['Nodes']
    edges = program_data['programming_details']['Edges']
    execution_time = next(
        (item['value'] for item in program_data['programming_details']['Nodes'] if item.get('name') == 'total_execution_time_ms'),
        None
    )

    if execution_time is None:
        raise ValueError("Execution time not found in program data")

    # Node mapping (name to index)
    node_map = {node['Name']: idx for idx, node in enumerate(nodes)}

    # Initialize graph components
    num_nodes = len(nodes)
    adj_list = [[] for _ in range(num_nodes)]  # Adjacency list for edges
    node_features = []
    node_sequences = []  # For LSTM processing
    edge_features = []
    edge_sequences = []

    # Process nodes
    for node in nodes:
        details = node['Details']
        # Extract numerical features
        numerical = extract_numerical_features(details)
        node_features.append(numerical)
        
        # Extract sequential data (e.g., memory access patterns, region computed)
        seq_data = []
        if 'Memory access patterns' in details:
            seq_data.extend(details['Memory access patterns'])
        if 'Region computed' in details:
            seq_data.extend(details['Region computed'])
        if 'Op histogram' in details:
            seq_data.extend(details['Op histogram'])
        node_sequences.append(process_sequential_data(seq_data))

    # Process edges
    for edge in edges:
        from_node = edge['From']
        to_node = edge['To']
        if from_node in node_map and to_node in node_map:
            from_idx = node_map[from_node]
            to_idx = node_map[to_node]
            adj_list[from_idx].append(to_idx)
            
            # Extract edge features
            details = edge['Details']
            numerical = extract_numerical_features(details)
            edge_features.append(numerical)
            
            # Extract sequential data (Footprint, Load Jacobians)
            seq_data = []
            if 'Footprint' in details:
                seq_data.extend(details['Footprint'])
            if 'Load Jacobians' in details:
                seq_data.extend(details['Load Jacobians'])
            edge_sequences.append(process_sequential_data(seq_data))

    # Pad node and edge features to fixed length
    max_node_feat_len = max(len(f) for f in node_features) if node_features else 1
    max_edge_feat_len = max(len(f) for f in edge_features) if edge_features else 1

    node_features = [
        np.pad(f, (0, max_node_feat_len - len(f)), mode='constant') if len(f) < max_node_feat_len else f[:max_node_feat_len]
        for f in node_features
    ]
    edge_features = [
        np.pad(f, (0, max_edge_feat_len - len(f)), mode='constant') if len(f) < max_edge_feat_len else f[:max_edge_feat_len]
        for f in edge_features
    ]

    return {
        'adj_list': adj_list,
        'node_features': np.array(node_features, dtype=np.float32),
        'node_sequences': np.array(node_sequences, dtype=np.float32),
        'edge_features': np.array(edge_features, dtype=np.float32),
        'edge_sequences': np.array(edge_sequences, dtype=np.float32),
        'execution_time': float(execution_time)
    }

def create_dataset(data_dir: str = 'synthetic_data') -> List[Dict[str, Any]]:
    """
    Create a dataset from all JSON files in the synthetic_data folder.
    Returns a list of graph representations.
    """
    data = []
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Directory {data_dir} does not exist")

    # Iterate through subfolders
    for subfolder in data_path.iterdir():
        if subfolder.is_dir():
            # Iterate through JSON files in subfolder
            for json_file in subfolder.glob('*.json'):
                try:
                    with open(json_file, 'r') as f:
                        program_data = json.load(f)
                    graph_data = create_graph_representation(program_data)
                    data.append(graph_data)
                    print(f"Processed {json_file}")
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")

    return data

# Create the dataset
if __name__ == '__main__':
    try:
        data = create_dataset('synthetic_data')
        print(f"Dataset created with {len(data)} samples")
        # Example: Print first sample structure
        if data:
            print("Sample data structure:")
            for key, value in data[0].items():
                if isinstance(value, np.ndarray):
                    print(f"{key}: shape {value.shape}")
                else:
                    print(f"{key}: {value}")
    except Exception as e:
        print(f"Failed to create dataset: {e}")
