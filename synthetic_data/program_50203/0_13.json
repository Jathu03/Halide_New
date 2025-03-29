import json
import os
import numpy as np
from typing import Dict
from torch.utils.data import Dataset
import tqdm

def extract_features(file_path: str, debug=False) -> Dict:
    """Extract features including 'total_execution_time_ms' from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Initialize execution time
    exec_time = None
    
    # Check for 'scheduling_data' at the top level
    if 'scheduling_data' in data:
        scheduling_data = data['scheduling_data']
        if debug:
            print(f"'scheduling_data' found in {file_path}: {json.dumps(scheduling_data, indent=2)}")
        
        # Case 1: 'scheduling_data' is a list of dictionaries
        if isinstance(scheduling_data, list):
            for entry in scheduling_data:
                if isinstance(entry, dict) and entry.get('name') == 'total_execution_time_ms':
                    try:
                        exec_time = float(entry['value'])
                        if debug:
                            print(f"Found 'total_execution_time_ms' = {exec_time} in 'scheduling_data' list for {file_path}")
                        break
                    except (KeyError, ValueError) as e:
                        if debug:
                            print(f"Error accessing 'value' in {file_path}: {entry}, Error: {e}")
        
        # Case 2: 'scheduling_data' is a dictionary
        elif isinstance(scheduling_data, dict):
            if 'total_execution_time_ms' in scheduling_data:
                try:
                    exec_time = float(scheduling_data['total_execution_time_ms'])
                    if debug:
                        print(f"Found 'total_execution_time_ms' = {exec_time} in 'scheduling_data' dict for {file_path}")
                except ValueError as e:
                    if debug:
                        print(f"Error converting 'total_execution_time_ms' to float in {file_path}: {e}")
        
        else:
            if debug:
                print(f"'scheduling_data' is neither a list nor a dict in {file_path}: {scheduling_data}")
    
    else:
        if debug:
            print(f"'scheduling_data' key not found in {file_path}. Keys available: {list(data.keys())}")

    # Fallback: Recursive search for 'total_execution_time_ms'
    if exec_time is None:
        def search_dict(d, target_key):
            if isinstance(d, dict):
                for k, v in d.items():
                    if k == target_key and isinstance(v, (int, float)):
                        return float(v)
                    result = search_dict(v, target_key)
                    if result is not None:
                        return result
            elif isinstance(d, list):
                for item in d:
                    result = search_dict(item, target_key)
                    if result is not None:
                        return result
            return None
        
        exec_time = search_dict(data, 'total_execution_time_ms')
        if exec_time is not None and debug:
            print(f"Found 'total_execution_time_ms' = {exec_time} via recursive search in {file_path}")
    
    # If still not found, raise an error with detailed info
    if exec_time is None:
        error_msg = f"No 'total_execution_time_ms' found in {file_path}"
        if debug:
            error_msg += f"\nKeys in data: {list(data.keys())}"
            if 'scheduling_data' in data:
                error_msg += f"\n'scheduling_data' structure: {json.dumps(data['scheduling_data'], indent=2)}"
        raise ValueError(error_msg)
    
    # Return features (simplified to focus on exec_time)
    return {'exec_time': exec_time}

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
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# Run the dataset creation
data_dir = "synthetic_data"
try:
    dataset = HalideDataset(data_dir, debug=True)
    print(f"Total dataset size: {len(dataset)}")
except Exception as e:
    print(f"Error creating dataset: {e}")
