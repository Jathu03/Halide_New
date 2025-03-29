import json
import os
import numpy as np
from typing import Dict, List
import re
from torch.utils.data import Dataset, DataLoader
import torch
import tqdm

def extract_features(file_path: str, debug=False) -> Dict:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract edge features
    edge_features = []
    for edge in data['programming_details']['Edges']:
        footprint_raw = edge['Details']['Footprint']
        footprint = []
        for f in footprint_raw:
            nums = re.findall(r'[+-]?\d*\.?\d+', f)
            if nums:
                footprint.append(float(nums[-1]))
            else:
                footprint.append(0.0)
        
        jacobian_raw = ' '.join(edge['Details']['Load Jacobians']).split()
        jacobian = []
        for x in jacobian_raw:
            try:
                if x not in ['_', '0']:
                    jacobian.append(float(x))
            except ValueError:
                jacobian.append(0.0)
        
        edge_features.append(np.array(footprint + jacobian))
    
    # Extract node features
    node_features = []
    for node in data['programming_details']['Nodes']:
        if 'Memory access patterns' in node['Details']:
            mem_patterns = [int(x) for pattern in node['Details']['Memory access patterns'] 
                          for x in pattern.split() if x.isdigit()]
            op_hist = [int(x.split()[-1]) for x in node['Details']['Op histogram']]
            node_features.append(np.array(mem_patterns + op_hist))
    
    # Extract execution time and scheduling features
    exec_time = None
    sched_features = []
    
    # Check 'Scheduling' section explicitly for the provided structure
    if 'Scheduling' in data['programming_details']:
        scheduling = data['programming_details']['Scheduling']
        if isinstance(scheduling, list):
            for entry in scheduling:
                if isinstance(entry, dict) and 'name' in entry and 'value' in entry:
                    if entry['name'] == 'total_execution_time_ms':
                        exec_time = float(entry['value'])  # Ensure it's a float
                        if debug:
                            print(f"Found execution time {exec_time} in 'Scheduling' for {file_path}")
                    elif 'Details' in entry and 'scheduling_feature' in entry['Details']:
                        feat = entry['Details']['scheduling_feature']
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
        else:
            if debug:
                print(f"'Scheduling' is not a list in {file_path}: {scheduling}")
    
    # Fallback recursive search if not found in 'Scheduling'
    if exec_time is None:
        def search_dict(d, key):
            if isinstance(d, dict):
                for k, v in d.items():
                    if k == key and isinstance(v, (int, float)):
                        return float(v)
                    result = search_dict(v, key)
                    if result is not None:
                        return result
            elif isinstance(d, list):
                for item in d:
                    result = search_dict(item, key)
                    if result is not None:
                        return result
            return None
        
        exec_time = search_dict(data, 'total_execution_time_ms')
        if exec_time is not None and debug:
            print(f"Found execution time {exec_time} via recursive search in {file_path}")
    
    if exec_time is None:
        error_msg = f"No 'total_execution_time_ms' found in {file_path}"
        if debug:
            error_msg += f". Data structure: {json.dumps(data['programming_details'], indent=2)}"
        raise ValueError(error_msg)
    
    return {
        'edge_seq': np.array(edge_features),
        'node_seq': np.array(node_features),
        'sched_context': np.mean(sched_features, axis=0) if sched_features else np.zeros(7),
        'exec_time': exec_time
    }

class HalideDataset(Dataset):
    def __init__(self, data_dir: str, debug=False):
        self.data = []
        programs = os.listdir(data_dir)
        for program in tqdm.tqdm(programs, desc="Processing programs"):
            program_path = os.path.join(data_dir, program)
            if os.path.isdir(program_path):
                for schedule_file in os.listdir(program_path):
                    file_path = os.path.join(program_path, schedule_file)
                    try:
                        features = extract_features(file_path, debug=debug)
                        self.data.append(features)
                    except ValueError as e:
                        print(f"Skipping {file_path}: {e}")
                        continue
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        continue
        
        if not self.data:
            raise ValueError("No valid data found in dataset")
        
        self._normalize_features()
    
    def _normalize_features(self):
        edge_lens = [len(d['edge_seq']) for d in self.data]
        node_lens = [len(d['node_seq']) for d in self.data]
        self.max_edge_len = max(edge_lens)
        self.max_node_len = max(node_lens)
        
        edge_dim = self.data[0]['edge_seq'].shape[1] if self.data[0]['edge_seq'].size > 0 else 1
        node_dim = self.data[0]['node_seq'].shape[1] if self.data[0]['node_seq'].size > 0 else 1
        
        for item in self.data:
            if item['edge_seq'].size > 0:
                edge_pad = np.zeros((self.max_edge_len - len(item['edge_seq']), edge_dim))
                item['edge_seq'] = np.vstack([item['edge_seq'], edge_pad])
            else:
                item['edge_seq'] = np.zeros((self.max_edge_len, edge_dim))
            
            if item['node_seq'].size > 0:
                node_pad = np.zeros((self.max_node_len - len(item['node_seq']), node_dim))
                item['node_seq'] = np.vstack([item['node_seq'], node_pad])
            else:
                item['node_seq'] = np.zeros((self.max_node_len, node_dim))
            
            item['exec_time'] = np.log1p(item['exec_time'])
    
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
    # Set debug=True to see extraction details, False for normal operation
    dataset = HalideDataset(data_dir, debug=False)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Train DataLoader size (number of batches): {len(train_loader)}")
    print(f"Test DataLoader size (number of batches): {len(test_loader)}")
    print(f"Total dataset size (number of samples): {len(dataset)}")
    print(f"Train dataset size: {train_size}, Test dataset size: {test_size}")

except Exception as e:
    print(f"Error creating dataset: {e}")
