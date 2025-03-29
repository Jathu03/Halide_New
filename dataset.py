import json
import os
import numpy as np
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader
import torch

def extract_features(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract edge features
    edge_features = []
    for edge in data['programming_details']['Edges']:
        footprint = [float(f.split()[-1].replace('(', '').replace(')', '')) 
                    for f in edge['Details']['Footprint'] if f.strip()]
        jacobian = [float(x) for x in ' '.join(edge['Details']['Load Jacobians']).split() 
                   if x.strip() and x not in ['_', '0']]
        edge_features.append(np.array(footprint + jacobian))
    
    # Extract node features
    node_features = []
    for node in data['programming_details']['Nodes']:
        if 'Memory access patterns' in node['Details']:
            mem_patterns = [int(x) for pattern in node['Details']['Memory access patterns'] 
                          for x in pattern.split() if x.isdigit()]
            op_hist = [int(x.split()[-1]) for x in node['Details']['Op histogram']]
            node_features.append(np.array(mem_patterns + op_hist))
    
    # Extract scheduling features
    sched_features = []
    for sched in data['programming_details']['Scheduling']:
        if 'scheduling_feature' in sched['Details']:
            feat = sched['Details']['scheduling_feature']
            sched_vec = [
                feat['bytes_at_production'], feat['bytes_at_realization'],
                feat['points_computed_total'], feat['num_vectors'],
                feat['vector_loads_per_vector'], feat['scalar_loads_per_scalar'],
                feat['working_set_at_root']
            ]
            sched_features.append(np.array(sched_vec))
    
    # Target execution time
    exec_time = next(item['value'] for item in data['programming_details']['Scheduling'] 
                    if item.get('name') == 'total_execution_time_ms')
    
    return {
        'edge_seq': np.array(edge_features),
        'node_seq': np.array(node_features),
        'sched_context': np.mean(sched_features, axis=0),
        'exec_time': exec_time
    }
   
class HalideDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data = []
        for program in os.listdir(data_dir):
            program_path = os.path.join(data_dir, program)
            if os.path.isdir(program_path):
                for schedule_file in os.listdir(program_path):
                    file_path = os.path.join(program_path, schedule_file)
                    features = extract_features(file_path)
                    self.data.append(features)
        
        # Normalize features
        self._normalize_features()
    
    def _normalize_features(self):
        edge_lens = [len(d['edge_seq']) for d in self.data]
        node_lens = [len(d['node_seq']) for d in self.data]
        self.max_edge_len = max(edge_lens)
        self.max_node_len = max(node_lens)
        
        for item in self.data:
            # Pad sequences
            edge_pad = np.zeros((self.max_edge_len - len(item['edge_seq']), 
                               item['edge_seq'].shape[1]))
            item['edge_seq'] = np.vstack([item['edge_seq'], edge_pad])
            
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
dataset = HalideDataset(data_dir)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(train_loader)
