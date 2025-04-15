import json
import logging
import os
from typing import Dict, List, Optional

# Configure logging with DEBUG level for detailed inspection
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_json_file(file_path: str) -> Optional[Dict]:
    """
    Parse a JSON file and extract all relevant details, with improved execution time handling.
    Returns a dictionary with parsed data or None if invalid.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Log top-level keys for debugging
        logging.debug(f"JSON keys in {file_path}: {list(data.keys())}")
        
        # Extract execution time
        execution_time = None
        possible_keys = [
            ('total_execution_time_ms', data),
            ('Metrics', 'total_execution_time_ms', data),
            ('Performance', 'total_execution_time_ms', data),
            ('execution_time_ms', data),
            ('total_time_ms', data)
        ]
        
        for key_info in possible_keys:
            if len(key_info) == 2:
                key, source = key_info
                if key in source:
                    execution_time = source[key]
                    logging.debug(f"Found execution time at '{key}': {execution_time} in {file_path}")
                    break
            elif len(key_info) == 3:
                parent, key, source = key_info
                if parent in source and key in source[parent]:
                    execution_time = source[parent][key]
                    logging.debug(f"Found execution time at '{parent}.{key}': {execution_time} in {file_path}")
                    break
        
        # Validate execution time
        if execution_time is None:
            logging.warning(f"No execution time key found in {file_path}. Using default: 1.0")
            execution_time = 1.0
        elif not isinstance(execution_time, (int, float)):
            logging.warning(f"Invalid execution time type ({type(execution_time)}) in {file_path}: {execution_time}. Using default: 1.0")
            execution_time = 1.0
        elif execution_time < 0:
            logging.warning(f"Negative execution time in {file_path}: {execution_time}. Using default: 1.0")
            execution_time = 1.0
        else:
            logging.debug(f"Valid execution time: {execution_time} ms in {file_path}")

        # Extract nodes
        nodes = []
        if 'programming_details' in data and 'Nodes' in data['programming_details']:
            nodes = data['programming_details']['Nodes']
            if not nodes:
                logging.warning(f"No nodes found in 'programming_details.Nodes' for {file_path}.")
                return None
        else:
            logging.warning(f"No 'programming_details.Nodes' found in {file_path}.")
            return None

        # Extract edges
        edges = data.get('programming_details', {}).get('Edges', [])

        # Extract node details
        node_details = []
        for node in nodes:
            node_info = {
                'name': node.get('name', ''),
                'Details': node.get('Details', {}),
                'scheduling_feature': node.get('scheduling_feature', None),
            }
            node_details.append(node_info)

        # Store all extracted data
        sample = {
            'file_path': file_path,
            'execution_time_ms': execution_time,
            'nodes': node_details,
            'node_count': len(nodes),
            'edges': edges,
            'edge_count': len(edges),
            'raw_data': data
        }

        return sample

    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON in {file_path}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error processing {file_path}: {e}")
        return None

def find_json_files(root_dir: str) -> List[str]:
    """
    Recursively find all JSON files in the given directory and its subfolders.
    """
    json_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return sorted(json_files)

def create_dataset(root_dir: str) -> List[Dict]:
    """
    Create a dataset from all JSON files in subfolders of the given directory.
    Returns a list of valid samples.
    """
    json_files = find_json_files(root_dir)
    if not json_files:
        logging.error(f"No JSON files found in {root_dir} or its subfolders.")
        raise ValueError("Dataset creation failed: No JSON files found.")

    dataset = []
    for file_path in json_files:
        sample = parse_json_file(file_path)
        if sample:
            dataset.append(sample)
            logging.info(f"Processed {file_path}: {sample['node_count']} nodes, "
                         f"{sample['edge_count']} edges, execution time {sample['execution_time_ms']} ms")
        else:
            logging.warning(f"Skipped invalid sample: {file_path}")

    if not dataset:
        logging.error("No valid samples created. Check JSON files for valid 'Nodes' and 'execution_time_ms'.")
        raise ValueError("Dataset creation failed: No valid samples.")

    logging.info(f"Created dataset with {len(dataset)} valid samples.")
    return dataset

def save_dataset(dataset: List[Dict], output_file: str):
    """
    Save the dataset to a JSON file and a summary to a text file.
    """
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    logging.info(f"Dataset saved to {output_file}")

    summary_file = "dataset_summary.txt"
    with open(summary_file, "w") as f:
        for sample in dataset:
            f.write(f"File: {sample['file_path']}, Nodes: {sample['node_count']}, "
                    f"Edges: {sample['edge_count']}, Execution Time: {sample['execution_time_ms']} ms\n")
    logging.info(f"Dataset summary saved to {summary_file}")

if __name__ == "__main__":
    root_dir = "synthetic_data"
    output_file = "synthetic_dataset.json"
    
    try:
        dataset = create_dataset(root_dir)
        save_dataset(dataset, output_file)
    except ValueError as e:
        logging.error(f"Dataset creation failed: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
