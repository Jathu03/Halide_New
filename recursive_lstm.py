import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
import glob
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Step 1: Feature Extraction
def extract_features(json_data):
    """Extract features from a JSON file."""
    features = {}
    
    # Helper function to find total_execution_time_ms in a list of dicts
    def find_execution_time(data_list):
        for item in data_list:
            if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
                try:
                    return float(item.get('value', 0))
                except (ValueError, TypeError):
                    logger.warning("Invalid total_execution_time_ms value, setting to 0")
                    return 0
        return None
    
    # Extract scheduling data (target)
    try:
        # Case 1: json_data is a dict with scheduling_data
        if isinstance(json_data, dict):
            scheduling_data = json_data.get('scheduling_data', [])
            if isinstance(scheduling_data, list):
                execution_time = find_execution_time(scheduling_data)
                if execution_time is not None:
                    features['execution_time'] = execution_time
                else:
                    logger.warning("No total_execution_time_ms found in scheduling_data, setting to 0")
                    features['execution_time'] = 0
            else:
                logger.warning("scheduling_data is not a list, setting execution_time to 0")
                features['execution_time'] = 0
        # Case 2: json_data is a list (original assumption)
        elif isinstance(json_data, list):
            execution_time = find_execution_time(json_data)
            if execution_time is not None:
                features['execution_time'] = execution_time
            else:
                logger.warning("No total_execution_time_ms found in root list, setting to 0")
                features['execution_time'] = 0
        else:
            logger.warning("json_data is neither a dict nor a list, setting execution_time to 0")
            features['execution_time'] = 0
    except Exception as e:
        logger.error(f"Error extracting execution time: {e}")
        features['execution_time'] = 0
    
    # Extract programming details
    programming_details = None
    if isinstance(json_data, dict):
        programming_details = json_data.get('programming_details', {})
    elif isinstance(json_data, list):
        for item in json_data:
            if isinstance(item, dict) and 'programming_details' in item:
                programming_details = item['programming_details']
                break
    
    if not programming_details:
        logger.warning("No programming_details found")
        return None
    
    edges = programming_details.get('Edges', []) or []
    nodes = programming_details.get('Nodes', []) or []
    
    # Node features
    node_features = []
    node_names = []
    for node in nodes:
        try:
            node_name = node.get('Name', '')
            if not node_name:
                logger.debug(f"Skipping node with no name")
                continue
            node_names.append(node_name)
            details = node.get('Details', {})
            
            # Memory access patterns
            mem_patterns = details.get('Memory access patterns', []) or []
            mem_vector = []
            for pattern in mem_patterns:
                if isinstance(pattern, str):
                    values = [int(x) for x in pattern.split() if x.isdigit()]
                    mem_vector.extend(values)
            
            # Op histogram
            op_hist = details.get('Op histogram', []) or []
            op_vector = []
            for op in op_hist:
                if isinstance(op, str):
                    try:
                        value = int(op.split(':')[-1].strip())
                        op_vector.append(value)
                    except (ValueError, IndexError):
                        logger.debug(f"Skipping invalid op histogram entry: {op}")
                        continue
            
            # Scheduling features
            sched_features = details.get('scheduling_feature', {}) or {}
            sched_vector = [float(v) for v in sched_features.values() if isinstance(v, (int, float))] if sched_features else []
            
            # Combine features
            node_feature = mem_vector + op_vector + sched_vector
            if not node_feature:
                logger.debug(f"No features extracted for node {node_name}")
                continue
            node_features.append(node_feature)
        except Exception as e:
            logger.error(f"Error processing node {node_name}: {e}")
            continue
    
    if not node_features:
        logger.warning("No valid node features extracted")
        return None
    
    # Pad node features to the same length
    max_len = max(len(f) for f in node_features)
    node_features = [f + [0] * (max_len - len(f)) for f in node_features]
    
    # Edge features and indices
    edge_index = []
    edge_features = []
    for edge in edges:
        try:
            from_node = edge.get('From', '')
            to_node = edge.get('To', '')
            if not (from_node and to_node):
                logger.debug(f"Skipping edge with missing From/To")
                continue
            if from_node in node_names and to_node in node_names:
                from_idx = node_names.index(from_node)
                to_idx = node_names.index(to_node)
                edge_index.append([from_idx, to_idx])
                
                # Extract Load Jacobians
                details = edge.get('Details', {})
                jacobians = details.get('Load Jacobians', []) or []
                jacobian_vector = []
                for row in jacobians:
                    if not isinstance(row, str):
                        logger.debug(f"Skipping non-string Jacobian: {row}")
                        continue
                    row = row.strip().split()
                    for val in row:
                        try:
                            if '/' in val:
                                num, denom = val.split('/')
                                jacobian_vector.append(float(num) / float(denom))
                            else:
                                jacobian_vector.append(float(val))
                        except (ValueError, ZeroDivisionError):
                            jacobian_vector.append(0.0)
                edge_features.append(jacobian_vector or [0])
        except Exception as e:
            logger.error(f"Error processing edge {from_node} -> {to_node}: {e}")
            continue
    
    # Pad edge features
    max_edge_len = max(len(f) for f in edge_features) if edge_features else 1
    edge_features = [f + [0] * (max_edge_len - len(f)) for f in edge_features]
    
    # Only return features if we have valid nodes and edges
    if not (node_features and edge_index):
        logger.warning("No valid nodes or edges, skipping")
        return None
    
    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'edge_features': edge_features,
        'execution_time': features['execution_time'],
        'node_names': node_names
    }

# Step 2: Create Dataset
def create_dataset(data_dir):
    """Create a dataset from the synthetic_data folder."""
    dataset = []
    scaler = StandardScaler()
    
    # Collect all JSON files
    json_files = glob.glob(os.path.join(data_dir, '**', '*.json'), recursive=True)
    logger.info(f"Found {len(json_files)} JSON files")
    
    # Extract raw features
    raw_features = []
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        try:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            features = extract_features(json_data)
            if features:
                raw_features.append(features)
            else:
                logger.debug(f"Skipping {json_file}: No valid features")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {json_file}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
            continue
    
    logger.info(f"Extracted features from {len(raw_features)} files")
    
    # Normalize node features
    all_node_features = []
    for features in raw_features:
        all_node_features.extend(features['node_features'])
    if all_node_features:
        try:
            scaler.fit(all_node_features)
        except Exception as e:
            logger.error(f"Error fitting scaler: {e}")
            return []
    else:
        logger.error("No node features to normalize")
        return []
    
    # Create PyG Data objects
    for features in tqdm(raw_features, desc="Creating Data objects"):
        try:
            node_features = scaler.transform(features['node_features']).astype(np.float32)
            edge_index = np.array(features['edge_index'], dtype=np.int64).T
            edge_features = np.array(features['edge_features'], dtype=np.float32)
            y = np.array([features['execution_time']], dtype=np.float32)
            
            # Create PyG Data object
            data = Data(
                x=torch.tensor(node_features, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_features, dtype=torch.float),
                y=torch.tensor(y, dtype=torch.float)
            )
            dataset.append(data)
        except Exception as e:
            logger.error(f"Error creating Data object for file: {e}")
            continue
    
    logger.info(f"Created {len(dataset)} Data objects")
    return dataset

# Step 3: DAG-LSTM Model
class DAGLSTM(nn.Module):
    """DAG-LSTM model using GCN layers for simplicity."""
    def __init__(self, input_dim, hidden_dim, edge_dim, num_layers=2):
        super(DAGLSTM, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.conv_layers.append(GCNConv(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        
        # Global pooling (mean)
        x = x.mean(dim=0, keepdim=True)
        x = self.fc(x)
        return x

# Step 4: Training Loop
def train_model(dataset, batch_size=32, epochs=100, hidden_dim=64, num_layers=2):
    """Train the DAG-LSTM model."""
    if not dataset:
        logger.error("Empty dataset provided")
        return None
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:]
    logger.info(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    try:
        input_dim = dataset[0].x.size(1)
        edge_dim = dataset[0].edge_attr.size(1) if dataset[0].edge_attr is not None else 0
    except IndexError:
        logger.error("No valid data to initialize model")
        return None
    
    model = DAGLSTM(input_dim, hidden_dim, edge_dim, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f"Using device: {device}")
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_batches = 0
        for batch in train_loader:
            try:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch.num_graphs
                train_batches += batch.num_graphs
            except Exception as e:
                logger.error(f"Error in training batch: {e}")
                continue
        train_loss = train_loss / train_batches if train_batches > 0 else float('inf')
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                try:
                    batch = batch.to(device)
                    out = model(batch)
                    loss = criterion(out, batch.y)
                    val_loss += loss.item() * batch.num_graphs
                    val_batches += batch.num_graphs
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue
        val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        
        logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return model

# Main Execution
if __name__ == "__main__":
    data_dir = "synthetic_data"
    if not os.path.exists(data_dir):
        logger.error(f"Directory {data_dir} does not exist")
        raise FileNotFoundError(f"Directory {data_dir} does not exist")
    
    logger.info("Creating dataset...")
    dataset = create_dataset(data_dir)
    logger.info(f"Dataset size: {len(dataset)}")
    
    if not dataset:
        logger.error("No valid data found in the dataset")
        raise ValueError("No valid data found in the dataset")
    
    logger.info("Training model...")
    model = train_model(dataset)
    if model:
        logger.info("Training completed")
    else:
        logger.error("Training failed")
