import json
import numpy as np
from collections import defaultdict
import networkx as nx
from sklearn.preprocessing import StandardScaler
import os
import glob

def parse_json_data(json_data):
    """
    Parse the JSON data to extract program and schedule features.
    """
    # Extract execution time
    execution_time = next(
        (item['value'] for item in json_data['programming_details'] if item.get('name') == 'total_execution_time_ms'),
        None
    )
    if execution_time is None:
        return None, None, None, None  # Skip files with missing execution time
    
    # Initialize graph
    G = nx.DiGraph()
    node_features = {}
    edge_features = {}
    
    # Process nodes
    nodes = [item for item in json_data['programming_details'] if 'Name' in item]
    for node in nodes:
        name = node['Name']
        details = node['Details']
        
        # Extract node features
        feature_vector = []
        
        # Memory access patterns
        if 'Memory access patterns' in details:
            mem_patterns = details['Memory access patterns']
            for pattern in mem_patterns:
                if isinstance(pattern, str):
                    values = [float(v) for v in pattern.split() if v.replace('.', '').isdigit()]
                    feature_vector.extend(values)
        
        # Operation histogram
        if 'Op histogram' in details:
            op_hist = details['Op histogram']
            for op in op_hist:
                if isinstance(op, str):
                    value = float(op.split(':')[-1].strip())
                    feature_vector.append(value)
        
        # Scheduling features
        if 'scheduling_feature' in details:
            sched = details['scheduling_feature']
            sched_features = [
                sched.get('allocation_bytes_read_per_realization', 0.0),
                sched.get('bytes_at_production', 0.0),
                sched.get('bytes_at_realization', 0.0),
                sched.get('bytes_at_root', 0.0),
                sched.get('bytes_at_task', 0.0),
                sched.get('inlined_calls', 0.0),
                sched.get('inner_parallelism', 0.0),
                sched.get('innermost_bytes_at_production', 0.0),
                sched.get('innermost_bytes_at_realization', 0.0),
                sched.get('innermost_bytes_at_root', 0.0),
                sched.get('innermost_bytes_at_task', 0.0),
                sched.get('innermost_loop_extent', 0.0),
                sched.get('innermost_pure_loop_extent', 0.0),
                sched.get('native_vector_size', 0.0),
                sched.get('num_productions', 0.0),
                sched.get('num_realizations', 0.0),
                sched.get('num_scalars', 0.0),
                sched.get('num_vectors', 0.0),
                sched.get('outer_parallelism', 0.0),
                sched.get('points_computed_minimum', 0.0),
                sched.get('points_computed_per_production', 0.0),
                sched.get('points_computed_per_realization', 0.0),
                sched.get('points_computed_total', 0.0),
                sched.get('scalar_loads_per_scalar', 0.0),
                sched.get('scalar_loads_per_vector', 0.0),
                sched.get('unique_bytes_read_per_realization', 0.0),
                sched.get('unique_lines_read_per_realization', 0.0),
                sched.get('unrolled_loop_extent', 0.0),
                sched.get('vector_loads_per_vector', 0.0),
                sched.get('vector_size', 0.0),
                sched.get('working_set', 0.0),
                sched.get('working_set_at_production', 0.0),
                sched.get('working_set_at_realization', 0.0),
                sched.get('working_set_at_root', 0.0),
                sched.get('working_set_at_task', 0.0),
            ]
            feature_vector.extend(sched_features)
        
        node_features[name] = feature_vector
        G.add_node(name)
    
    # Process edges
    edges = json_data['programming_details'].get('Edges', [])
    for edge in edges:
        from_node = edge['From']
        to_node = edge['To']
        details = edge['Details']
        
        # Extract edge features
        edge_vector = []
        
        # Footprint
        if 'Footprint' in details:
            footprint = details['Footprint']
            for fp in footprint:
                # Extract numerical values from footprint expressions
                try:
                    # Simple parsing for min/max values
                    value = fp.split(':')[-1].strip()
                    # Handle basic arithmetic expressions
                    if '/' in value:
                        num, denom = value.split('/')
                        edge_vector.append(float(num) / float(denom))
                    elif '*' in value:
                        parts = value.split('*')
                        result = 1.0
                        for part in parts:
                            result *= float(part.strip('()'))
                        edge_vector.append(result)
                    else:
                        edge_vector.append(float(value))
                except (ValueError, ZeroDivisionError):
                    edge_vector.append(0.0)
        
        # Load Jacobians
        if 'Load Jacobians' in details:
            jacobians = details['Load Jacobians']
            for row in jacobians:
                values = [float(v) if v.replace('/', '').replace('.', '').replace('-', '').isdigit() or '/' in v else 0.0 for v in row.split()]
                edge_vector.extend(values)
        
        edge_features[(from_node, to_node)] = edge_vector
        G.add_edge(from_node, to_node)
    
    return G, node_features, edge_features, execution_time

def create_sequence_representation(G, node_features, edge_features, max_nodes):
    """
    Create a sequential representation of the DAG for LSTM input, padded to max_nodes.
    """
    # Perform topological sort to ensure consistent ordering
    topo_order = list(nx.topological_sort(G))
    
    # Initialize sequence
    sequence = []
    max_node_len = max(len(f) for f in node_features.values()) if node_features else 1
    max_edge_len = max(len(f) for f in edge_features.values()) if edge_features else 1
    
    for node in topo_order[:max_nodes]:  # Limit to max_nodes
        # Get node features
        node_vec = node_features.get(node, [0.0] * max_node_len)
        if len(node_vec) < max_node_len:
            node_vec.extend([0.0] * (max_node_len - len(node_vec)))
        
        # Get incoming edge features
        edge_vec = [0.0] * max_edge_len
        predecessors = list(G.predecessors(node))
        if predecessors:
            # Average features of incoming edges
            incoming_edges = [(pred, node) for pred in predecessors if (pred, node) in edge_features]
            if incoming_edges:
                edge_vecs = [edge_features[edge] for edge in incoming_edges]
                edge_vec = np.mean([ev + [0.0] * (max_edge_len - len(ev)) for ev in edge_vecs], axis=0).tolist()
        
        # Combine node and edge features
        combined = node_vec + edge_vec
        sequence.append(combined)
    
    # Pad sequence if fewer than max_nodes
    while len(sequence) < max_nodes:
        sequence.append([0.0] * (max_node_len + max_edge_len))
    
    return np.array(sequence)

def normalize_features(sequences):
    """
    Normalize feature sequences.
    """
    scaler = StandardScaler()
    flattened = sequences.reshape(-1, sequences.shape[-1])
    normalized = scaler.fit_transform(flattened)
    return normalized.reshape(sequences.shape), scaler

def prepare_dataset(synthetic_data_dir):
    """
    Process all JSON files in synthetic_data directory and create halide_data dataset.
    """
    dataset = []
    max_nodes = 0
    all_sequences = []
    execution_times = []
    
    # Collect all JSON files
    json_files = []
    for root, _, files in os.walk(synthetic_data_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    # Determine max_nodes across all graphs
    for json_file in json_files:
        with open(json_file, 'r') as f:
            try:
                json_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON file: {json_file}")
                continue
                
        G, _, _, exec_time = parse_json_data(json_data)
        if G is None or exec_time is None:
            print(f"Skipping file with missing data: {json_file}")
            continue
        max_nodes = max(max_nodes, G.number_of_nodes())
    
    # Process each JSON file
    for json_file in json_files:
        with open(json_file, 'r') as f:
            try:
                json_data = json.load(f)
            except json.JSONDecodeError:
                continue
                
        # Parse JSON to extract features
        G, node_features, edge_features, execution_time = parse_json_data(json_data)
        if G is None or execution_time is None:
            continue
        
        # Create sequence representation
        sequence = create_sequence_representation(G, node_features, edge_features, max_nodes)
        all_sequences.append(sequence)
        execution_times.append(execution_time)
    
    # Convert to numpy arrays
    all_sequences = np.array(all_sequences)
    
    # Normalize features
    normalized_sequences, scaler = normalize_features(all_sequences)
    
    # Create dataset
    dataset = {
        'sequences': normalized_sequences,
        'execution_times': np.array(execution_times),
        'scaler': scaler
    }
    
    # Save dataset
    np.savez('halide_data.npz', 
             sequences=dataset['sequences'], 
             execution_times=dataset['execution_times'])
    
    return dataset

# Example usage
if __name__ == "__main__":
    synthetic_data_dir = "synthetic_data"
    dataset = prepare_dataset(synthetic_data_dir)
    print(f"Dataset created with {len(dataset['execution_times'])} samples")
    print(f"Sequence shape: {dataset['sequences'].shape}")
    print(f"Execution times shape: {dataset['execution_times'].shape}")
