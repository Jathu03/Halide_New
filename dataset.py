import json
import os
import numpy as np
from typing import Dict, List
import re
import torch
from torch.utils.data import Dataset, DataLoader

def extract_features(file_path: str, debug=True) -> Dict:
    """Extract features from a JSON file, including edge, node, scheduling, and execution time."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract edge features with robust handling
    edge_features = []
    for edge in data.get('programming_details', {}).get('Edges', []):
        footprint_raw = edge.get('Details', {}).get('Footprint', [])
        footprint = []
        for f in footprint_raw:
            if f.strip():
                last_token = f.split()[-1].replace('(', '').replace(')', '')
                numeric_values = re.findall(r'[+-]?\d*\.?\d+', last_token)
                footprint.append(float(numeric_values[-1]) if numeric_values else 0.0)
            else:
                footprint.append(0.0)
        
        jacobian = []
        for x in ' '.join(edge.get('Details', {}).get('Load Jacobians', [])).split():
            if x.strip() and x not in ['_', '0']:
                try:
                    jacobian.append(float(x))
                except ValueError:
                    jacobian.append(0.0)
        
        edge_features.append(np.array(footprint + jacobian))
    
    # Extract node features
    node_features = []
    for node in data.get('programming_details', {}).get('Nodes', []):
        if 'Memory access patterns' in node.get('Details', {}):
            mem_patterns = [int(x) for pattern in node['Details']['Memory access patterns'] 
                           for x in pattern.split() if x.isdigit()]
            op_hist = [int(x.split()[-1]) for x in node['Details']['Op histogram']]
            node_features.append(np.array(mem_patterns + op_hist))
    
    # Extract scheduling features and execution time
    sched_features = []
    exec_time = None
    programming_details = data.get('programming_details', {})
    
    if 'Scheduling' in programming_details:
        for sched in programming_details['Scheduling']:
            if 'scheduling_feature' in sched.get('Details', {}):
                feat = sched['Details']['scheduling_feature']
                sched_vec = [
                    feat.get('bytes_at_production', 0.0),
                    feat.get('bytes_at_realization', 0.0),
                    feat.get('points_computed_total', 0.0),
                    feat.get('num_vectors', 0.0),
                    feat.get('vector_loads_per_vector', 0.0),
                    feat.get('scalar_loads_per_scalar', 0.0),
                    feat.get('working_set_at_root', 0.0)
                ]
                sched_features.append(np.array(sched_vec))
            elif sched.get('name') == 'total_execution_time_ms':
                exec_time = sched['value']
                if debug:
                    print(f"Found execution time {exec_time} in 'Scheduling' for {file_path}")
    else:
        if debug:
            print(f"'Scheduling' not found in 'programming_details' for {file_path}")
    
    # Fallback: Recursive search for 'total_execution_time_ms'
    if exec_time is None:
        def search_dict(d, key, depth=0, max_depth=10):
            if depth > max_depth or exec_time is not None:
                return None
            if isinstance(d, dict):
                for k, v in d.items():
                    if k.lower() == key.lower():  # Case-insensitive match
                        if isinstance(v, (int, float)):
                            return v
                    result = search_dict(v, key, depth + 1, max_depth)
                    if result is not None:
                        return result
            elif isinstance(d, list):
                for item in d:
                    result = search_dict(item, key, depth + 1, max_depth)
                    if result is not None:
                        return result
            return None
        
        exec_time = search_dict(data, 'total_execution_time_ms')
        if exec_time is not None and debug:
            print(f"Found execution time {exec_time} via recursive search in {file_path}")
    
    # Fallback: Search for similar keys (e.g., 'execution_time', 'exec_time_ms')
    if exec_time is None:
        def find_similar_key(d, partial_keys=['execution_time', 'exec_time'], depth=0, max_depth=10):
            if depth > max_depth:
                return None
            if isinstance(d, dict):
                for k, v in d.items():
                    if any(p.lower() in k.lower() for p in partial_keys):
                        if isinstance(v, (int, float)):
                            return v
                    result = find_similar_key(v, partial_keys, depth + 1, max_depth)
                    if result is not None:
                        return result
            elif isinstance(d, list):
                for item in d:
                    result = find_similar_key(item, partial_keys, depth + 1, max_depth)
                    if result is not None:
                        return result
            return None
        
        exec_time = find_similar_key(data)
        if exec_time is not None and debug:
            print(f"Found execution time {exec_time} via partial key match in {file_path}")
    
    # Final fallback: Assign a default value or skip
    if exec_time is None:
        if debug:
            print(f"No execution time found in {file_path}; assigning default value 0.0")
        exec_time = 0.0  # Default value instead of skipping
    
    return {
        'edge_seq': np.array(edge_features),
        'node_seq': np.array(node_features),
        'sched_context': np.mean(sched_features, axis=0) if sched_features else np.zeros(7),
        'exec_time': exec_time
    }

class HalideDataset(Dataset):
    """Custom Dataset class for loading and normalizing Halide scheduling data."""
    def __init__(self, data_dir: str, debug=True):
        self.data = []
        self.debug = debug
        for program in os.listdir(data_dir):
            program_path = os.path.join(data_dir, program)
            if os.path.isdir(program_path):
                for schedule_file in os.listdir(program_path):
                    file_path = os.path.join(program_path, schedule_file)
                    try:
                        features = extract_features(file_path, debug=debug)
                        self.data.append(features)
                    except Exception as e:
                        if debug:
                            print(f"Error processing {file_path}: {e}")
                        continue
        
        if not self.data:
            raise ValueError("No valid data found in dataset")
        
        # Normalize features
        self._normalize_features()
    
    def _normalize_features(self):
        """Normalize and pad edge, node sequences, and execution time."""
        edge_lens = [len(d['edge_seq']) for d in self.data]
        node_lens = [len(d['node_seq']) for d in self.data]
        self.max_edge_len = max(edge_lens) if edge_lens else 1
        self.max_node_len = max(node_lens) if node_lens else 1
        
        for item in self.data:
            # Pad edge sequences
            if len(item['edge_seq']) == 0:
                item['edge_seq'] = np.zeros((self.max_edge_len, 1))
            else:
                edge_pad = np.zeros((self.max_edge_len - len(item['edge_seq']), 
                                   item['edge_seq'].shape[1]))
                item['edge_seq'] = np.vstack([item['edge_seq'], edge_pad])
            
            # Pad node sequences
            if len(item['node_seq']) == 0:
                item['node_seq'] = np.zeros((self.max_node_len, 1))
            else:
                node_pad = np.zeros((self.max_node_len - len(item['node_seq']), 
                                   item['node_seq'].shape[1]))
                item['node_seq'] = np.vstack([item['node_seq'], node_pad])
            
            # Normalize execution time
            item['exec_time'] = np.log1p(item['exec_time'])  # Log transform for stability
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'edge_seq': torch.FloatTensor(item['edge_seq']),
            'node_seq': torch.FloatTensor(item['node_seq']),
            'sched_context': torch.FloatTensor(item['sched_context']),
            'exec_time': torch.FloatTensor([item['exec_time']])
        }

# Create dataset and dataloader
data_dir = "synthetic_data"
try:
    dataset = HalideDataset(data_dir, debug=True)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Train DataLoader size: {len(train_loader)}, Test DataLoader size: {len(test_loader)}")
except Exception as e:
    print(f"Error creating dataset: {e}")
