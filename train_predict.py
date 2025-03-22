import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# Constants
HIDDEN_DIM = 128
NODE_FEATURE_DIM = 64  # Will be determined based on feature extraction
MAX_CHILDREN = 10  # Maximum number of children for a node in the tree
MAX_TREE_DEPTH = 10
MAX_COMPUTATIONS_PER_NODE = 5
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

###########################################
# Data Loading and Preprocessing
###########################################

def load_tiramisu_programs(folder_path):
    """Load all Tiramisu programs from a folder."""
    programs = []
    
    # Walk through the directory
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        program_data = json.load(f)
                        # Add file path for reference
                        program_data['file_path'] = file_path
                        programs.append(program_data)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(programs)} Tiramisu programs from {folder_path}")
    return programs

def extract_execution_time(exploration_trace):
    """Extract execution time from exploration trace."""
    if exploration_trace is None:
        return None
    
    # Get the execution time from the evaluation field
    return exploration_trace.get('evaluation', None)

def extract_best_schedule(exploration_trace, depth_limit=None):
    """
    Recursively find the best schedule in the exploration trace.
    Returns a tuple of (schedule, execution_time)
    """
    if exploration_trace is None or not exploration_trace:
        return None, float('inf')
    
    current_schedule = exploration_trace.get('schedule', '')
    current_time = exploration_trace.get('evaluation', float('inf'))
    current_depth = exploration_trace.get('depth', 0)
    
    # If we've reached the depth limit, return current schedule
    if depth_limit is not None and current_depth >= depth_limit:
        return current_schedule, current_time
    
    # Check all children for better schedules
    best_schedule, best_time = current_schedule, current_time
    
    children = exploration_trace.get('children', [])
    for child in children:
        child_schedule, child_time = extract_best_schedule(child, depth_limit)
        if child_time < best_time:
            best_schedule, best_time = child_schedule, child_time
    
    return best_schedule, best_time

def extract_access_patterns(accesses):
    """Extract features from access patterns."""
    if not accesses:
        return []
    
    features = []
    for access in accesses:
        # Extract buffer ID and whether it's a reduction
        buffer_id = access.get('buffer_id', -1)
        is_reduction = access.get('access_is_reduction', False)
        
        # Extract access matrix properties
        matrix = access.get('access_matrix', [])
        matrix_flat = []
        for row in matrix:
            matrix_flat.extend(row)
        
        # Pad or truncate the matrix to a fixed size
        matrix_flat = matrix_flat[:10]  # Limit to first 10 elements
        while len(matrix_flat) < 10:
            matrix_flat.append(0)  # Pad with zeros
        
        # Combine features
        access_features = [buffer_id, int(is_reduction)] + matrix_flat
        features.append(access_features)
    
    # Ensure we have a fixed number of access patterns
    padded_features = features[:5]  # Limit to 5 access patterns
    while len(padded_features) < 5:
        padded_features.append([0] * 12)  # Pad with zero features
    
    return padded_features

def flatten_expression(expr):
    """Flatten an expression tree into a feature vector."""
    if not expr:
        return []
    
    # Get expression type and encode it
    expr_type = expr.get('expr_type', 'unknown')
    expr_types = ['add', 'mul', 'div', 'sub', 'max', 'min', 'sqrt', 'access', 'value', 'unknown']
    expr_type_encoding = [0] * len(expr_types)
    if expr_type in expr_types:
        expr_type_encoding[expr_types.index(expr_type)] = 1
    
    # Recursively process children up to a fixed depth
    children = expr.get('children', [])
    child_features = []
    
    for child in children[:2]:  # Limit to first 2 children
        child_features.extend(flatten_expression(child))
    
    # Ensure fixed-size feature vector
    all_features = expr_type_encoding + child_features
    all_features = all_features[:30]  # Limit to 30 features
    while len(all_features) < 30:
        all_features.append(0)  # Pad with zeros
    
    return all_features

def extract_computation_features(computation):
    """Extract features from a computation."""
    features = []
    
    # Basic computation properties
    features.append(computation.get('absolute_order', 0))
    features.append(1 if computation.get('comp_is_reduction', False) else 0)
    
    # Data type encoding
    data_types = ['float32', 'float64', 'int32', 'int64', 'unknown']
    data_type = computation.get('data_type', 'unknown')
    data_type_encoding = [0] * len(data_types)
    if data_type in data_types:
        data_type_encoding[data_types.index(data_type)] = 1
    features.extend(data_type_encoding)
    
    # Number of iterators
    iterators = computation.get('iterators', [])
    features.append(len(iterators))
    
    # Access patterns
    access_pattern_features = []
    for access_pattern in extract_access_patterns(computation.get('accesses', [])):
        access_pattern_features.extend(access_pattern)
    features.extend(access_pattern_features[:60])  # Limit access pattern features
    
    # Expression representation
    expr_features = flatten_expression(computation.get('expression_representation', {}))
    features.extend(expr_features)
    
    return features

def extract_schedule_features(schedule):
    """Extract features from a schedule."""
    if not schedule:
        return [0] * 20  # Return zeros if no schedule
    
    features = []
    
    # Tiling information
    tiling = schedule.get('tiling', {})
    has_tiling = 1 if tiling else 0
    features.append(has_tiling)
    
    # Tiling factors (up to 3 dimensions)
    tiling_factors = []
    for i in range(3):
        if i < len(tiling.keys()):
            factor = list(tiling.values())[i]
            tiling_factors.append(factor)
        else:
            tiling_factors.append(0)
    features.extend(tiling_factors)
    
    # Unrolling
    unrolling_factor = schedule.get('unrolling_factor', 0)
    features.append(1 if unrolling_factor else 0)
    features.append(unrolling_factor if unrolling_factor else 0)
    
    # Parallelization
    parallelized_dim = schedule.get('parallelized_dim', None)
    features.append(1 if parallelized_dim is not None else 0)
    if parallelized_dim is not None:
        # Encode which dimension is parallelized (assume up to 5 dims)
        dim_encoding = [0] * 5
        if isinstance(parallelized_dim, int) and 0 <= parallelized_dim < 5:
            dim_encoding[parallelized_dim] = 1
        features.extend(dim_encoding)
    else:
        features.extend([0] * 5)
    
    # Transformations
    transformations = schedule.get('transformations_list', [])
    features.append(len(transformations))
    
    # Ensure fixed length
    features = features[:20]  # Limit to 20 features
    while len(features) < 20:
        features.append(0)  # Pad with zeros
    
    return features

def extract_loop_features(loop_node, iterators_info):
    """Extract features from a loop node."""
    loop_name = loop_node.get('loop_name', '')
    
    # Get iterator information
    iterator_info = iterators_info.get(loop_name, {})
    
    features = []
    
    # Loop bounds
    lower_bound = iterator_info.get('lower_bound', 0)
    upper_bound = iterator_info.get('upper_bound', 0)
    
    # Convert bounds to numeric values if possible
    if isinstance(lower_bound, str):
        # If it's a variable reference, set to -1
        lower_bound = -1
    if isinstance(upper_bound, str):
        # If it's a variable reference, set to -1
        upper_bound = -1
    
    features.append(lower_bound)
    features.append(upper_bound)
    
    # Loop depth and children
    parent = iterator_info.get('parent', None)
    features.append(1 if parent else 0)
    
    children = iterator_info.get('children', [])
    features.append(len(children))
    
    # Number of computations in this loop
    computations = loop_node.get('computations_list', [])
    features.append(len(computations))
    
    return features

def build_tree_structure(node, iterators_info, computations_info, schedules_info):
    """
    Recursively build a tree structure with features.
    Returns a dict with node features and children.
    """
    if not node:
        return None
    
    # Extract loop features
    loop_features = extract_loop_features(node, iterators_info)
    
    # Extract computation features for computations in this loop
    computation_features = []
    for comp_name in node.get('computations_list', []):
        if comp_name in computations_info:
            comp_features = extract_computation_features(computations_info[comp_name])
            schedule_features = extract_schedule_features(schedules_info.get(comp_name, {}))
            computation_features.append(comp_features + schedule_features)
    
    # Pad computation features to fixed size
    while len(computation_features) < MAX_COMPUTATIONS_PER_NODE:
        # Add empty computation (zeros)
        computation_features.append([0] * (len(extract_computation_features({})) + 
                                          len(extract_schedule_features({}))))
    
    # Cap at maximum number of computations
    computation_features = computation_features[:MAX_COMPUTATIONS_PER_NODE]
    
    # Process children
    children = []
    for child in node.get('child_list', [])[:MAX_CHILDREN]:
        child_node = build_tree_structure(child, iterators_info, computations_info, schedules_info)
        if child_node:
            children.append(child_node)
    
    # Pad children to fixed size
    while len(children) < MAX_CHILDREN:
        children.append(None)
    
    # Cap at maximum number of children
    children = children[:MAX_CHILDREN]
    
    return {
        'loop_features': loop_features,
        'computation_features': computation_features,
        'children': children
    }

def preprocess_tiramisu_program(program):
    """
    Preprocess a single Tiramisu program.
    Extract features and structure for the cost model.
    """
    # Extract program annotation
    annotation = program.get('program_annotation', {})
    
    # Extract iterator information
    iterators_info = {}
    for it_name, it_info in annotation.get('iterators', {}).items():
        iterators_info[it_name] = {
            'lower_bound': it_info.get('lower_bound', 0),
            'upper_bound': it_info.get('upper_bound', 0),
            'parent': it_info.get('parent_iterator', None),
            'children': it_info.get('child_iterators', [])
        }
    
    # Extract computation information
    computations_info = annotation.get('computations', {})
    
    # Extract schedules
    schedules_info = {}
    for schedule_entry in program.get('schedules_list', []):
        for comp_name, comp_schedule in schedule_entry.items():
            if comp_name not in ['fusions', 'sched_str', 'tree_structure', 'legality_check', 'exploration_method']:
                schedules_info[comp_name] = comp_schedule
    
    # Extract tree structure
    tree_roots = []
    for schedule_entry in program.get('schedules_list', []):
        if 'tree_structure' in schedule_entry and 'roots' in schedule_entry['tree_structure']:
            tree_roots = schedule_entry['tree_structure']['roots']
            break
    
    # Build tree structure with features
    tree_structure = []
    for root in tree_roots:
        tree_node = build_tree_structure(root, iterators_info, computations_info, schedules_info)
        if tree_node:
            tree_structure.append(tree_node)
    
    # Extract execution time from exploration trace
    exploration_trace = program.get('exploration_trace', {})
    best_schedule, execution_time = extract_best_schedule(exploration_trace)
    
    return {
        'tree_structure': tree_structure,
        'execution_time': execution_time,
        'best_schedule': best_schedule,
        'file_path': program.get('file_path', '')
    }

def flatten_tree_to_sequence(tree_node, depth=0):
    """
    Flatten a tree structure into a sequence of nodes.
    Each node contains its features and depth information.
    """
    if tree_node is None:
        return []
    
    # Current node features
    node_seq = [{
        'loop_features': tree_node['loop_features'],
        'computation_features': tree_node['computation_features'],
        'depth': depth
    }]
    
    # Add children nodes
    for child in tree_node['children']:
        if child is not None:
            node_seq.extend(flatten_tree_to_sequence(child, depth + 1))
    
    return node_seq

###########################################
# Neural Network Model
###########################################

class TreeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input gates
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)
        
        # Forget gate for each child
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)
        
        # Output gate
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)
        
        # Cell update
        self.W_u = nn.Linear(input_size, hidden_size)
        self.U_u = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x, child_h, child_c):
        """
        Forward pass for Tree-LSTM cell.
        
        Args:
            x: Input tensor (node features)
            child_h: List of hidden states from children
            child_c: List of cell states from children
        
        Returns:
            h: Updated hidden state
            c: Updated cell state
        """
        # If no children, create empty states
        if not child_h:
            child_h = [torch.zeros(1, self.hidden_size).to(x.device)]
            child_c = [torch.zeros(1, self.hidden_size).to(x.device)]
        
        # Stack child hidden states
        h_sum = torch.sum(torch.stack(child_h), dim=0)
        
        # Input gate
        i = torch.sigmoid(self.W_i(x) + self.U_i(h_sum))
        
        # Forget gates (one per child)
        f_k = []
        for h_k in child_h:
            f_k.append(torch.sigmoid(self.W_f(x) + self.U_f(h_k)))
        
        # Output gate
        o = torch.sigmoid(self.W_o(x) + self.U_o(h_sum))
        
        # Cell update
        u = torch.tanh(self.W_u(x) + self.U_u(h_sum))
        
        # Update cell state
        c = i * u
        for f, c_k in zip(f_k, child_c):
            c = c + f * c_k
        
        # Update hidden state
        h = o * torch.tanh(c)
        
        return h, c

class RecursiveTreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RecursiveTreeLSTM, self).__init__()
        self.hidden_size = hidden_size
        
        # Embedding layers for different feature types
        self.loop_embedding = nn.Linear(5, hidden_size // 4)  # Assuming 5 loop features
        
        # Each computation has combined computation+schedule features
        comp_sched_feature_size = len(extract_computation_features({})) + len(extract_schedule_features({}))
        self.comp_embedding = nn.Linear(comp_sched_feature_size, hidden_size // 4)
        
        # Cell for recursive processing
        self.cell = TreeLSTMCell(hidden_size, hidden_size)
    
    def forward(self, tree_node):
        """
        Recursively process a tree node and its children.
        
        Args:
            tree_node: Dict containing node features and children
        
        Returns:
            h: Hidden state for this subtree
            c: Cell state for this subtree
        """
        if tree_node is None:
            return (torch.zeros(1, self.hidden_size), 
                    torch.zeros(1, self.hidden_size))
        
        # Process children first
        child_h = []
        child_c = []
        
        for child in tree_node['children']:
            if child is not None:
                h_k, c_k = self.forward(child)
                child_h.append(h_k)
                child_c.append(c_k)
        
        # Convert loop features to tensor and embed
        loop_features = torch.tensor(tree_node['loop_features'], dtype=torch.float32).unsqueeze(0)
        loop_embed = self.loop_embedding(loop_features)
        
        # Process and embed computation features
        comp_embeds = []
        for comp_features in tree_node['computation_features']:
            comp_tensor = torch.tensor(comp_features, dtype=torch.float32).unsqueeze(0)
            comp_embed = self.comp_embedding(comp_tensor)
            comp_embeds.append(comp_embed)
        
        # Combine all computation embeddings
        if comp_embeds:
            comp_combined = torch.mean(torch.stack(comp_embeds), dim=0)
        else:
            comp_combined = torch.zeros(1, self.hidden_size // 4)
        
        # Concatenate all features
        padding = torch.zeros(1, self.hidden_size // 2)  # Padding to reach hidden_size
        x = torch.cat([loop_embed, comp_combined, padding], dim=1)
        
        # Apply Tree-LSTM cell
        h, c = self.cell(x, child_h, child_c)
        
        return h, c

class SequenceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SequenceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        """
        Process a sequence of node features.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, input_size)
        
        Returns:
            output: LSTM output for each time step
            (h_n, c_n): Final hidden and cell states
        """
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)

class TiramisuCostModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TiramisuCostModel, self).__init__()
        self.hidden_size = hidden_size
        
        # Tree-LSTM for hierarchical structure
        self.tree_lstm = RecursiveTreeLSTM(input_size, hidden_size)
        
        # Sequence LSTM for processing node attributes
        self.seq_lstm = SequenceLSTM(input_size + 1, hidden_size)  # +1 for depth
        
        # Final prediction layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, tree_structure):
        """
        Forward pass for the cost model.
        
        Args:
            tree_structure: List of trees to process
        
        Returns:
            execution_time: Predicted execution time
        """
        # Process tree structure with Tree-LSTM
        tree_outputs = []
        for tree in tree_structure:
            if tree is not None:
                h, _ = self.tree_lstm(tree)
                tree_outputs.append(h)
        
        # Combine multiple trees if present
        if tree_outputs:
            tree_encoding = torch.mean(torch.stack(tree_outputs, dim=0), dim=0)
        else:
            tree_encoding = torch.zeros(1, self.hidden_size)
        
        # Flatten trees into sequences for LSTM processing
        seq_features = []
        for tree in tree_structure:
            if tree is not None:
                seq_nodes = flatten_tree_to_sequence(tree)
                
                # Prepare features for each node in the sequence
                for node in seq_nodes:
                    # Combine loop and computation features
                    loop_features = torch.tensor(node['loop_features'], dtype=torch.float32)
                    
                    # Process each computation
                    comp_tensors = []
                    for comp in node['computation_features']:
                        comp_tensor = torch.tensor(comp, dtype=torch.float32)
                        comp_tensors.append(comp_tensor)
                    
                    # Combine computation features
                    if comp_tensors:
                        comp_combined = torch.mean(torch.stack(comp_tensors), dim=0)
                    else:
                        comp_combined = torch.zeros(len(extract_computation_features({})) + 
                                                   len(extract_schedule_features({})))
                    
                    # Add depth information
                    depth = torch.tensor([node['depth']], dtype=torch.float32)
                    
                    # Combine all features
                    node_features = torch.cat([loop_features, comp_combined, depth])
                    seq_features.append(node_features)
        
        # Process sequence with LSTM if we have features
        if seq_features:
            seq_tensor = torch.stack(seq_features).unsqueeze(0)  # Add batch dimension
            _, (h_n, _) = self.seq_lstm(seq_tensor)
            seq_encoding = h_n.squeeze(0)
        else:
            seq_encoding = torch.zeros(1, self.hidden_size)
        
        # Combine tree and sequence encodings
        combined = torch.cat([tree_encoding, seq_encoding], dim=1)
        
        # Final prediction
        hidden = F.relu(self.fc1(combined))
        execution_time = self.fc2(hidden)
        
        return execution_time

###########################################
# Dataset and DataLoader
###########################################

class TiramisuDataset(Dataset):
    def __init__(self, preprocessed_programs):
        self.programs = preprocessed_programs
    
    def __len__(self):
        return len(self.programs)
    
    def __getitem__(self, idx):
        program = self.programs[idx]
        
        # Return tree structure and execution time
        return {
            'tree_structure': program['tree_structure'],
            'execution_time': program['execution_time'],
            'file_path': program['file_path']
        }

def collate_fn(batch):
    """
    Custom collate function for batching tree structures.
    Each item is processed individually since tree structures can't be batched easily.
    """
    return batch

###########################################
# Training and Evaluation
###########################################

def train_tiramisu_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)"):
            optimizer.zero_grad()
            
            batch_loss = 0
            for item in batch:
                # Move data to device
                tree_structure = item['tree_structure']
                true_time = torch.tensor(item['execution_time'], dtype=torch.float32).to(device)
                
                # Forward pass
                pred_time = model(tree_structure).squeeze()
                
                # Calculate loss
                loss = criterion(pred_time, true_time)
                batch_loss += loss
            
            # Average loss over batch
            batch_loss /= len(batch)
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)"):
                batch_loss = 0
                for item in batch:
                    # Move data to device
                    tree_structure = item['tree_structure']
                    true_time = torch.tensor(item['execution_time'], dtype=torch.float32).to(device)
                    
                    # Forward pass
                    pred_time = model(tree_structure).squeeze()
                    
                    # Calculate loss
                    loss = criterion(pred_time, true_time)
                    batch_loss += loss
                
                # Average loss over batch
                batch_loss /= len(batch)
                val_loss += batch_loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_tiramisu_model.pth")
            print(f"Saved best model with validation loss: {best_val_loss:.6f}")
    
    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()
    
    return model, train_losses, val_losses

def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    criterion = nn.MSELoss()
    test_loss = 0
    
    # For calculating metrics
    true_times = []
    pred_times = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch_loss = 0
            
            for item in batch:
                # Move data to device
                tree_structure = item['tree_structure']
                true_time = torch.tensor(item['execution_time'], dtype=torch.float32).to(device)
                
                # Forward pass
                pred_time = model(tree_structure).squeeze()
                
                # Calculate loss
                loss = criterion(pred_time, true_time)
                batch_loss += loss
                
                # Store for metrics
                true_times.append(true_time.item())
                pred_times.append(pred_time.item())
            
            # Average loss over batch
            batch_loss /= len(batch)
            test_loss += batch_loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    
    # Calculate additional metrics
    true_times = np.array(true_times)
    pred_times = np.array(pred_times)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(true_times - pred_times))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((true_times - pred_times) / true_times)) * 100
    
    # R^2 score
    ss_tot = np.sum((true_times - np.mean(true_times)) ** 2)
    ss_res = np.sum((true_times - pred_times) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"Test Loss (MSE): {avg_test_loss:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"R^2 Score: {r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(true_times, pred_times, alpha=0.5)
    plt.plot([min(true_times), max(true_times)], [min(true_times),
