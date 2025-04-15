import json
import logging
import os
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_json_file(file_path: str) -> Optional[Dict]:
    """
    Parse a JSON file and extract execution time and nodes.
    Returns a dictionary with parsed data or None if invalid.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract execution time
        execution_time = None
        if 'total_execution_time_ms' in data:
            execution_time = data['total_execution_time_ms']
            if not isinstance(execution_time, (int, float)) or execution_time <= 0:
                logging.warning(f"Invalid execution time {execution_time} in {file_path}. Using default: 1.0")
                execution_time = 1.0
        else:
            logging.warning(f"No 'total_execution_time_ms' found in {file_path}. Using default: 1.0")
            execution_time = 1.0

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

        if not nodes:
            logging.warning(f"Skipping {file_path} due to empty nodes list.")
            return None

        # Extract edges (optional)
        edges = data.get('programming_details', {}).get('Edges', [])

        return {
            'execution_time_ms': execution_time,
            'nodes': nodes,
            'edges': edges,
            'file_path': file_path
        }

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
    return json_files

def create_dataset(root_dir: str) -> List[Dict]:
    """
    Create a dataset from all JSON files in subfolders of the given directory.
    Returns a list of valid samples.
    """
    # Find all JSON files
    json_files = find_json_files(root_dir)
    if not json_files:
        logging.error(f"No JSON files found in {root_dir} or its subfolders.")
        raise ValueError("Dataset creation failed: No JSON files found.")

    dataset = []
    for file_path in json_files:
        sample = parse_json_file(file_path)
        if sample:
            dataset.append(sample)
            logging.info(f"Processed {file_path}: {len(sample['nodes'])} nodes, "
                         f"execution time {sample['execution_time_ms']} ms")
        else:
            logging.warning(f"Skipped invalid sample: {file_path}")

    if not dataset:
        logging.error("No valid samples created. Check JSON files for valid 'Nodes' and 'execution_time_ms'.")
        raise ValueError("Dataset creation failed: No valid samples.")

    logging.info(f"Created dataset with {len(dataset)} valid samples.")
    return dataset

if __name__ == "__main__":
    # Define the root directory
    root_dir = "synthetic_data"
    try:
        dataset = create_dataset(root_dir)
        # Optionally, save the dataset to a file or process further
        with open("dataset_summary.txt", "w") as f:
            for sample in dataset:
                f.write(f"File: {sample['file_path']}, Nodes: {len(sample['nodes'])}, "
                        f"Execution Time: {sample['execution_time_ms']} ms\n")
        logging.info("Dataset summary saved to dataset_summary.txt")
    except ValueError as e:
        logging.error(f"Dataset creation failed: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
