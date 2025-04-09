import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import defaultdict
import matplotlib.pyplot as plt

# Load the JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data  # Return the full JSON object (could be a list)

# 1. Enhanced Feature Extraction
def extract_features(data):
    # Check if data is a list and access the first element if so
    if isinstance(data, list) and len(data) > 0:
        data_dict = data[0]
    else:
        print("Error: Expected a list with at least one dictionary. Using empty dict as fallback.")
        data_dict = {}

    edges = data_dict.get('programming_details', {}).get('Edges', [])
    nodes = data_dict.get('programming_details', {}).get('Nodes', [])
    
    # Feature dictionaries
    edge_features = []
    node_features = []
    temporal_sequences = []  # For temporal dependencies
    
    # Extract Edge Features
    for edge in edges:
        footprint = edge['Details']['Footprint']
        jacobians = edge['Details']['Load Jacobians']
        
        # Safely extract Footprint values with defaults
        fp_min_0 = parse_footprint(footprint[0]) if len(footprint) > 0 else 0.0
        fp_max_0 = parse_footprint(footprint[1]) if len(footprint) > 1 else 0.0
        fp_min_1 = parse_footprint(footprint[2]) if len(footprint) > 2 else 0.0
        fp_max_1 = parse_footprint(footprint[3]) if len(footprint) > 3 else 0.0
        fp_min_2 = parse_footprint(footprint[4]) if len(footprint) > 4 else 0.0
        fp_max_2 = parse_footprint(footprint[5]) if len(footprint) > 5 else 0.0
        
        # Safely extract Jacobian values with defaults, handling fractions
        def parse_jacobian(value):
            try:
                return float(eval(value.replace('_', '0')))  # Handles fractions like '1/8'
            except:
                return 0.0
        
        jacobian_00 = parse_jacobian(jacobians[0].split()[0]) if len(jacobians) > 0 else 0.0
        jacobian_11 = parse_jacobian(jacobians[1].split()[1]) if len(jacobians) > 1 else 0.0
        jacobian_22 = parse_jacobian(jacobians[2].split()[2]) if len(jacobians) > 2 else 0.0
        
        edge_dict = {
            'name': edge['Name'],
            'from': edge['From'],
            'to': edge['To'],
            # Footprint features (min/max bounds for each dimension)
            'footprint_min_0': fp_min_0,
            'footprint_max_0': fp_max_0,
            'footprint_min_1': fp_min_1,
            'footprint_max_1': fp_max_1,
            'footprint_min_2': fp_min_2,
            'footprint_max_2': fp_max_2,
            # Jacobian features (dependency scaling factors)
            'jacobian_00': jacobian_00,
            'jacobian_11': jacobian_11,
            'jacobian_22': jacobian_22,
        }
        edge_features.append(edge_dict)
        temporal_sequences.append(edge['Name'])  # Preserve edge order
    
    # Extract Node Features
    for node in nodes:
        details = node['Details']
        # Safely extract operation counts from Op histogram
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
                # Memory access patterns
                'pointwise': sum(map(int, details['Memory access patterns'][0].split()[1:])),
                'transpose': sum(map(int, details['Memory access patterns'][1].split()[1:])),
                'broadcast': sum(map(int, details['Memory access patterns'][2].split()[1:])),
                'slice': sum(map(int, details['Memory access patterns'][3].split()[1:])),
                # Operation histogram (computational complexity)
                'ops_add': ops_add,
                'ops_mul': ops_mul,
                'ops_div': ops_div,
                # Scheduling features
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
    execution_time = data_dict.get('scheduling_data', {}).get('total_execution_time_ms', 0.0)
    if execution_time == 0.0:
        print("Warning: 'total_execution_time_ms' not found in 'scheduling_data'. Using default value 0.0.")
    
    return edge_features, node_features, temporal_sequences, execution_time

# Helper to parse footprint expressions (simplified for numeric extraction)
def parse_footprint(footprint_str):
    # Extract the last part after ':' and attempt to convert to float
    try:
        value = footprint_str.split(':')[-1].strip()
        return float(eval(value))  # Caution: eval used for simplicity; refine for production
    except:
        return 0.0  # Default for complex expressions

# 2. Data Preprocessing Pipeline
def preprocess_data(edge_features, node_features, temporal_sequences):
    # Convert to DataFrames
    edge_df = pd.DataFrame(edge_features)
    node_df = pd.DataFrame(node_features)
    
    # Encode categorical variables
    le = LabelEncoder()
    edge_df['from'] = le.fit_transform(edge_df['from'])
    edge_df['to'] = le.fit_transform(edge_df['to'])
    node_df['name'] = le.fit_transform(node_df['name'])
    
    # Normalize numerical features
    scaler = StandardScaler()
    edge_numeric_cols = ['footprint_min_0', 'footprint_max_0', 'footprint_min_1', 
                         'footprint_max_1', 'footprint_min_2', 'footprint_max_2',
                         'jacobian_00', 'jacobian_11', 'jacobian_22']
    node_numeric_cols = ['pointwise', 'transpose', 'broadcast', 'slice', 
                         'ops_add', 'ops_mul', 'ops_div']
    
    # Add scheduling features if present
    if 'inner_parallelism' in node_df.columns:
        node_numeric_cols.extend(['inner_parallelism', 'outer_parallelism', 
                                  'num_vectors', 'working_set', 'points_computed_total'])
    
    edge_df[edge_numeric_cols] = scaler.fit_transform(edge_df[edge_numeric_cols])
    node_df[node_numeric_cols] = scaler.fit_transform(node_df[node_numeric_cols])
    
    # Create a temporal sequence representation
    sequence_data = []
    for seq in temporal_sequences:
        edge_idx = edge_df[edge_df['name'] == seq].index[0]
        edge_row = edge_df.iloc[edge_idx][edge_numeric_cols + ['from', 'to']].values
        sequence_data.append(edge_row)
    
    return edge_df, node_df, np.array(sequence_data)

# 3. Validate Feature Distributions
def validate_distributions(edge_df, node_df):
    # Plot distributions of key features
    plt.figure(figsize=(12, 6))
    
    # Edge features
    plt.subplot(1, 2, 1)
    edge_df[['footprint_min_0', 'jacobian_00']].hist(bins=20)
    plt.title("Edge Feature Distributions")
    
    # Node features
    plt.subplot(1, 2, 2)
    node_df[['ops_add', 'pointwise']].hist(bins=20)
    plt.title("Node Feature Distributions")
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Updated file path
    file_path = "synthetic_data/program_50001/0_15.json"
    data = load_json_data(file_path)
    
    # Extract features
    edge_features, node_features, temporal_sequences, execution_time = extract_features(data)
    
    # Preprocess data
    edge_df, node_df, sequence_data = preprocess_data(edge_features, node_features, temporal_sequences)
    
    # Validate distributions
    validate_distributions(edge_df, node_df)
    
    # Output shapes and sample
    print("Edge DataFrame Shape:", edge_df.shape)
    print("Node DataFrame Shape:", node_df.shape)
    print("Sequence Data Shape:", sequence_data.shape)
    print("Sample Sequence Data:\n", sequence_data[0])
    print("Execution Time:", execution_time)
    
    # Save preprocessed data for next week's model training
    np.save("sequence_data.npy", sequence_data)
    edge_df.to_csv("edge_features.csv", index=False)
    node_df.to_csv("node_features.csv", index=False)
    with open("execution_time.txt", "w") as f:
        f.write(str(execution_time))
