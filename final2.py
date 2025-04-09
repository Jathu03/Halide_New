import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from glob import glob

# Load the JSON data from a file
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# 1. Enhanced Feature Extraction
def extract_features(data):
    # Handle different JSON structures
    if isinstance(data, list) and len(data) > 0:
        data_dict = data[0]
    elif isinstance(data, dict):
        data_dict = data
    else:
        print("Error: Unexpected JSON structure. Expected a list or dict. Using empty dict as fallback.")
        data_dict = {}

    edges = data_dict.get('programming_details', {}).get('Edges', [])
    nodes = data_dict.get('programming_details', {}).get('Nodes', [])
    
    # Feature dictionaries
    edge_features = []
    node_features = []
    temporal_sequences = []
    
    # Extract Edge Features
    for edge in edges:
        footprint = edge['Details']['Footprint']
        jacobians = edge['Details']['Load Jacobians']
        
        fp_min_0 = parse_footprint(footprint[0]) if len(footprint) > 0 else 0.0
        fp_max_0 = parse_footprint(footprint[1]) if len(footprint) > 1 else 0.0
        fp_min_1 = parse_footprint(footprint[2]) if len(footprint) > 2 else 0.0
        fp_max_1 = parse_footprint(footprint[3]) if len(footprint) > 3 else 0.0
        fp_min_2 = parse_footprint(footprint[4]) if len(footprint) > 4 else 0.0
        fp_max_2 = parse_footprint(footprint[5]) if len(footprint) > 5 else 0.0
        
        def parse_jacobian(value):
            try:
                return float(eval(value.replace('_', '0')))
            except:
                return 0.0
        
        jacobian_00 = parse_jacobian(jacobians[0].split()[0]) if len(jacobians) > 0 else 0.0
        jacobian_11 = parse_jacobian(jacobians[1].split()[1]) if len(jacobians) > 1 else 0.0
        jacobian_22 = parse_jacobian(jacobians[2].split()[2]) if len(jacobians) > 2 else 0.0
        
        edge_dict = {
            'name': edge['Name'],
            'from': edge['From'],
            'to': edge['To'],
            'footprint_min_0': fp_min_0,
            'footprint_max_0': fp_max_0,
            'footprint_min_1': fp_min_1,
            'footprint_max_1': fp_max_1,
            'footprint_min_2': fp_min_2,
            'footprint_max_2': fp_max_2,
            'jacobian_00': jacobian_00,
            'jacobian_11': jacobian_11,
            'jacobian_22': jacobian_22,
        }
        edge_features.append(edge_dict)
        temporal_sequences.append(edge['Name'])
    
    # Extract Node Features
    for node in nodes:
        details = node['Details']
        op_hist = details['Op histogram']
        ops_add = 0
        ops_mul = 0
        ops_div = 0
        for line in op_hist:
            parts = line.split()
            if len(parts) >= 2:
                if parts[1] == '+:':
                    ops_add = int(parts[2])
                elif parts[1] == '*:':
                    ops_mul = int(parts[2])
                elif parts[1] == '/:':
                    ops_div = int(parts[2])
        
        if 'scheduling_feature' in details:
            sched = details['scheduling_feature']
            node_dict = {
                'name': node['Name'],
                'pointwise': sum(map(int, details['Memory access patterns'][0].split()[1:])),
                'transpose': sum(map(int, details['Memory access patterns'][1].split()[1:])),
                'broadcast': sum(map(int, details['Memory access patterns'][2].split()[1:])),
                'slice': sum(map(int, details['Memory access patterns'][3].split()[1:])),
                'ops_add': ops_add,
                'ops_mul': ops_mul,
                'ops_div': ops_div,
                'inner_parallelism': sched['inner_parallelism'],
                'outer_parallelism': sched['outer_parallelism'],
                'num_vectors': sched['num_vectors'],
                'working_set': sched['working_set'],
                'points_computed_total': sched['points_computed_total'],
            }
        else:
            node_dict = {
                'name': node['Name'],
                'pointwise': sum(map(int, details['Memory access patterns'][0].split()[1:])),
                'transpose': sum(map(int, details['Memory access patterns'][1].split()[1:])),
                'broadcast': sum(map(int, details['Memory access patterns'][2].split()[1:])),
                'slice': sum(map(int, details['Memory access patterns'][3].split()[1:])),
                'ops_add': ops_add,
                'ops_mul': ops_mul,
                'ops_div': ops_div,
            }
        node_features.append(node_dict)
    
    # Total execution time from scheduling_data
    scheduling_data = data_dict.get('scheduling_data', [])
    execution_time = 0.0
    for item in scheduling_data:
        if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
            execution_time = item.get('value', 0.0)
            break
    if execution_time == 0.0:
        print("Warning: 'total_execution_time_ms' not found in 'scheduling_data'. Using default value 0.0.")
    
    return edge_features, node_features, temporal_sequences, execution_time

# 2. Data Preprocessing Pipeline
def preprocess_data(edge_features_list, node_features_list, temporal_sequences_list):
    # Combine all edge and node features into DataFrames
    all_edge_df = pd.concat([pd.DataFrame(ef) for ef in edge_features_list], ignore_index=True) if edge_features_list else pd.DataFrame()
    all_node_df = pd.concat([pd.DataFrame(nf) for nf in node_features_list], ignore_index=True) if node_features_list else pd.DataFrame()
    
    # Encode categorical variables
    le = LabelEncoder()
    if not all_edge_df.empty and 'from' in all_edge_df.columns:
        all_edge_df['from'] = le.fit_transform(all_edge_df['from'])
        all_edge_df['to'] = le.fit_transform(all_edge_df['to'])
    if not all_node_df.empty and 'name' in all_node_df.columns:
        all_node_df['name'] = le.fit_transform(all_node_df['name'])
    
    # Normalize numerical features
    scaler = StandardScaler()
    edge_numeric_cols = ['footprint_min_0', 'footprint_max_0', 'footprint_min_1', 
                         'footprint_max_1', 'footprint_min_2', 'footprint_max_2',
                         'jacobian_00', 'jacobian_11', 'jacobian_22']
    node_numeric_cols = ['pointwise', 'transpose', 'broadcast', 'slice', 
                         'ops_add', 'ops_mul', 'ops_div']
    
    if 'inner_parallelism' in all_node_df.columns:
        node_numeric_cols.extend(['inner_parallelism', 'outer_parallelism', 
                                  'num_vectors', 'working_set', 'points_computed_total'])
    
    if not all_edge_df.empty and all(col in all_edge_df.columns for col in edge_numeric_cols):
        all_edge_df[edge_numeric_cols] = scaler.fit_transform(all_edge_df[edge_numeric_cols])
    if not all_node_df.empty and all(col in all_node_df.columns for col in node_numeric_cols):
        all_node_df[node_numeric_cols] = scaler.fit_transform(all_node_df[node_numeric_cols])
    
    # Create temporal sequence representations for each file
    sequence_data_list = []
    for temporal_sequences, edge_features in zip(temporal_sequences_list, edge_features_list):
        edge_df = pd.DataFrame(edge_features)
        if not edge_df.empty and 'from' in edge_df.columns:
            edge_df['from'] = le.fit_transform(edge_df['from'])
            edge_df['to'] = le.fit_transform(edge_df['to'])
            edge_df[edge_numeric_cols] = scaler.fit_transform(edge_df[edge_numeric_cols])
            sequence_data = []
            for seq in temporal_sequences:
                edge_idx = edge_df[edge_df['name'] == seq].index
                if not edge_idx.empty:
                    edge_row = edge_df.iloc[edge_idx[0]][edge_numeric_cols + ['from', 'to']].values
                    sequence_data.append(edge_row)
            sequence_data_list.append(np.array(sequence_data))
        else:
            sequence_data_list.append(np.array([]))
    
    return all_edge_df, all_node_df, sequence_data_list

# Helper to parse footprint expressions
def parse_footprint(footprint_str):
    try:
        value = footprint_str.split(':')[-1].strip()
        return float(eval(value))
    except:
        return 0.0

# Main execution
if __name__ == "__main__":
    # Base directory
    base_dir = "synthetic_data"
    
    # Collect all JSON files
    json_files = glob(os.path.join(base_dir, "*", "*.json"))
    print(f"Found {len(json_files)} JSON files.")
    
    # Lists to store data from all files
    all_edge_features = []
    all_node_features = []
    all_temporal_sequences = []
    all_execution_times = []
    
    # Process each file
    for file_path in json_files:
        print(f"Processing: {file_path}")
        try:
            data = load_json_data(file_path)
            edge_features, node_features, temporal_sequences, execution_time = extract_features(data)
            all_edge_features.append(edge_features)
            all_node_features.append(node_features)
            all_temporal_sequences.append(temporal_sequences)
            all_execution_times.append(execution_time)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Preprocess all data
    edge_df, node_df, sequence_data_list = preprocess_data(all_edge_features, all_node_features, all_temporal_sequences)
    execution_times = np.array(all_execution_times)
    
    # Pad sequence_data_list to have uniform shape (files × max_timesteps × features)
    max_timesteps = max(len(seq) for seq in sequence_data_list if seq.size > 0) if sequence_data_list else 0
    if max_timesteps > 0:
        padded_sequences = []
        feature_dim = sequence_data_list[0].shape[1] if sequence_data_list and sequence_data_list[0].size > 0 else 0
        for seq in sequence_data_list:
            if seq.size > 0:
                padded = np.pad(seq, ((0, max_timesteps - len(seq)), (0, 0)), mode='constant', constant_values=0)
                padded_sequences.append(padded)
            else:
                padded_sequences.append(np.zeros((max_timesteps, feature_dim)))
        all_sequence_data = np.stack(padded_sequences)
    else:
        all_sequence_data = np.array([])  # Empty if no valid sequences
    
    # Output shapes
    print("All Edge DataFrame Shape:", edge_df.shape)
    print("All Node DataFrame Shape:", node_df.shape)
    print("All Sequence Data Shape (files × timesteps × features):", all_sequence_data.shape)
    print("Execution Times Shape:", execution_times.shape)
    if all_sequence_data.size > 0:
        print("Sample Sequence Data (first file, first timestep):\n", all_sequence_data[0, 0])
    print("Sample Execution Time (first file):", execution_times[0])
    
    # Save the dataset
    output_dir = "preprocessed_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "sequence_data.npy"), all_sequence_data)
    edge_df.to_csv(os.path.join(output_dir, "edge_features.csv"), index=False)
    node_df.to_csv(os.path.join(output_dir, "node_features.csv"), index=False)
    np.save(os.path.join(output_dir, "execution_times.npy"), execution_times)
    print(f"Dataset saved to {output_dir}")
