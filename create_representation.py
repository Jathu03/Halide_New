import json
import logging
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
        else:
            logging.warning(f"No 'programming_details.Nodes' found in {file_path}.")
            return None

        if not nodes:
            logging.warning(f"Skipping {file_path} due to empty nodes list.")
            return None

        # Extract edges (optional, for completeness)
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

def create_dataset(json_files: List[str]) -> List[Dict]:
    """
    Create a dataset from a list of JSON files.
    Returns a list of valid samples.
    """
    dataset = []
    for file_path in json_files:
        sample = parse_json_file(file_path)
        if sample:
            dataset.append(sample)
        else:
            logging.warning(f"Skipping invalid sample: {file_path}")

    if not dataset:
        logging.error("No valid samples created. Check JSON files for valid 'Nodes' and 'execution_time_ms'.")
        raise ValueError("Dataset creation failed: No valid samples.")

    logging.info(f"Created dataset with {len(dataset)} valid samples.")
    return dataset

# Example usage
if __name__ == "__main__":
    json_files = [
        "synthetic_data/program_50077/0_15.json",
        # Add other JSON files as needed
    ]
    try:
        dataset = create_dataset(json_files)
        for sample in dataset:
            logging.info(f"Processed {sample['file_path']}: {len(sample['nodes'])} nodes, "
                         f"execution time {sample['execution_time_ms']} ms")
    except ValueError as e:
        logging.error(f"Dataset creation failed: {e}")
