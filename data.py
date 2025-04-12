import os
import json
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
import numpy as np
from collections import defaultdict
import pickle

# Define a function to parse a single JSON file and create a graph representation
def parse_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract programming details
    edges_data = data.get('programming_details', {}).get('Edges', [])
    nodes_data = data.get('programming_details', {}).get('Nodes', [])
    
    # Extract execution time from scheduling_data
    scheduling_data = data.get('scheduling_data', [])
    execution_time = None
    for item in scheduling_data:
        if item.get('name') == 'total_execution_time_ms':
            execution_time = item.get('value')
            break
    if execution_time is None:
        raise ValueError(f"No execution time found in {file_path}")
    
    # Create node mapping (name to index)
    node_name_to_idx = {node['Name']: idx for idx, node in enumerate(nodes_data)}
    
    # Initialize lists for graph construction
    edge_index = []
    edge_attr = []
    node_features = []
    node_sequences = []  # For LSTM input (sequential features)
    
    # Process nodes
    for node in nodes_data:
        node_name = node['Name']
        details = node.get('Details', {})
        
        # Structural features (for GNN)
        # Example: Op histogram and memory access patterns
        op_histogram = details.get('Op histogram', [])
        op_features = []
        for op in op_histogram:
            # Extract numerical value from strings like "Constant:   5"
            try:
                value = float(op.split(':')[-1].strip())
                op_features.append(value)
            except:
                op_features.append(0.0)
        
        memory_patterns = details.get('Memory access patterns', [])
        memory_features = []
        for pattern in memory_patterns:
            # Extract numerical values from strings like "Pointwise:      1 0 0 1"
            try:
                values = [float(x) for x in pattern.split(':')[-1].strip().split()]
                memory_features.extend(values)
            except:
                memory_features.extend([0.0] * 4)  # Assume 4 values per pattern
        
        # Pad or truncate to fixed length
        max_op_len = 24  # Adjust based on max op histogram length
        max_mem_len = 32  # Adjust based on max memory patterns (e.g., 8 patterns * 4 values)
        op_features = (op_features + [0.0] * max_op_len)[:max_op_len]
        memory_features = (memory_features + [0.0] * max_mem_len)[:max_mem_len]
        
        structural_features = op_features + memory_features
        
        # Sequential features (for LSTM)
        # Example: Scheduling features as a sequence
        scheduling_features = details.get('scheduling_feature', {})
        seq_features = []
        if scheduling_features:
            # Convert scheduling features to a sequence
            for key, value in scheduling_features.items():
                try:
                    seq_features.append(float(value))
                except:
                    seq_features.append(0.0)
        else:
            # Fallback: Use memory patterns as sequence
            seq_features = memory_features[:]
        
        # Pad or truncate sequential features
        max_seq_len = 50  # Adjust based on max scheduling features
        seq_features = (seq_features + [0.0] * max_seq_len)[:max_seq_len]
        
        node_features.append(structural_features)
        node_sequences.append(seq_features)
    
    # Process edges
    for edge in edges_data:
        from_node = edge.get('From')
        to_node = edge.get('To')
        if from_node in node_name_to_idx and to_node in node_name_to_idx:
            from_idx = node_name_to_idx[from_node]
            to_idx = node_name_to_idx[to_node]
            edge_index.append([from_idx, to_idx])
            
            # Edge features (e.g., Load Jacobians, Footprint transformations)
            details = edge.get('Details', {})
            jacobians = details.get('Load Jacobians', [])
            footprint = details.get('Footprint', [])
            
            edge_features = []
            # Process Jacobians
            for jac in jacobians:
                try:
                    values = [float(x) for x in jac.strip().split()]
                    edge_features.extend(values)
                except:
                    edge_features.extend([0.0] * 4)  # Adjust based on Jacobian size
            
            # Process Footprint (simplified: extract numerical values)
            for fp in footprint:
                try:
                    # Extract numbers from strings like "Min 0: (upsampled_linear__0._0.min/8)"
                    import re
                    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', fp)
                    values = [float(x) for x in numbers]
                    edge_features.extend(values)
                except:
                    edge_features.append(0.0)
            
            # Pad or truncate edge features
            max_edge_len = 50  # Adjust based on max edge feature length
            edge_features = (edge_features + [0.0] * max_edge_len)[:max_edge_len]
            edge_attr.append(edge_features)
    
    # Convert to tensors
    x = torch.tensor(node_features, dtype=torch.float)
    node_sequences = torch.tensor(node_sequences, dtype=torch.float)  # Shape: [num_nodes, seq_len]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = torch.tensor([execution_time], dtype=torch.float)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        node_sequences=node_sequences  # Custom attribute for LSTM
    )
    
    return data

# Define a custom Dataset class
class HalideDataset(Dataset):
    def __init__(self, data_list=None, root='data_g'):
        # Initialize data_list before parent class init
        self.data_list = data_list if data_list is not None else []
        super(HalideDataset, self).__init__(root)
        # Ensure processed directory exists
        os.makedirs(self.processed_dir, exist_ok=True)
    
    @property
    def processed_file_names(self):
        # If data_list is available, use its length
        if self.data_list:
            return [f'data_{i}.pt' for i in range(len(self.data_list))]
        # Otherwise, infer from existing files in processed_dir
        if os.path.exists(self.processed_dir):
            return [f for f in os.listdir(self.processed_dir) if f.endswith('.pt')]
        return []
    
    @property
    def num_graphs(self):
        # Return the number of graphs based on processed files or data_list
        if self.data_list:
            return len(self.data_list)
        if os.path.exists(self.processed_dir):
            return len([f for f in os.listdir(self.processed_dir) if f.endswith('.pt')])
        return 0
    
    def len(self):
        return self.num_graphs
    
    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    
    def process(self):
        if not self.data_list:
            return  # Skip processing if no data to process
        # Process and save data files
        for i, data in enumerate(self.data_list):
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))
    
    def _process(self):
        # Override to avoid weights_only issue
        if not self.data_list:
            return  # Nothing to process
        # Only process if processed files don't exist or are outdated
        expected_files = set(self.processed_file_names)
        existing_files = set(f for f in os.listdir(self.processed_dir) if f.endswith('.pt'))
        
        # If the expected files match existing files, skip processing
        if expected_files == existing_files:
            return
        
        # Clear existing files to avoid inconsistencies
        for f in existing_files:
            os.remove(os.path.join(self.processed_dir, f))
        
        # Process and save new files
        self.process()

# Main function to process all files
def create_dataset(data_dir='synthetic_data', output_dir='data_g'):
    data_list = []
    
    # Iterate through all subfolders and files
    for program_folder in os.listdir(data_dir):
        program_path = os.path.join(data_dir, program_folder)
        if not os.path.isdir(program_path):
            continue
        
        for file_name in os.listdir(program_path):
            if not file_name.endswith('.json'):
                continue
            
            file_path = os.path.join(program_path, file_name)
            try:
                data = parse_json_file(file_path)
                data_list.append(data)
                print(f"Processed {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    # Create dataset
    dataset = HalideDataset(data_list, root=output_dir)
    
    # Process and save the dataset
    dataset.process()
    
    # Save dataset metadata
    metadata_path = os.path.join(output_dir, 'metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump({
            'num_graphs': len(data_list),
            'node_feature_dim': data_list[0].x.shape[1] if data_list else 0,
            'edge_feature_dim': data_list[0].edge_attr.shape[1] if data_list and data_list[0].edge_attr.numel() > 0 else 0,
            'seq_feature_dim': data_list[0].node_sequences.shape[1] if data_list else 0
        }, f)
    
    print(f"Dataset saved to {output_dir} with {len(data_list)} graphs")
    return dataset

# Run the dataset creation
if __name__ == "__main__":
    dataset = create_dataset(data_dir='synthetic_data', output_dir='data_g')
    print("Dataset creation completed.")
