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
NODE_FEATURE_DIM = 64
MAX_CHILDREN = 10
MAX_TREE_DEPTH = 10
MAX_COMPUTATIONS_PER_NODE = 5
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 0.001

### Utility Function to Handle Case Sensitivity
def lowercase_keys(data):
    """Recursively convert all keys in a dictionary or list of dictionaries to lowercase."""
    if isinstance(data, dict):
        return {k.lower(): lowercase_keys(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [lowercase_keys(item) for item in data]
    else:
        return data

### Data Loading and Preprocessing
def load_tiramisu_programs(folder_path):
    """Load Tiramisu programs from the folder, expecting a nested structure."""
    programs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        program_data = json.load(f)
                        program_data = lowercase_keys(program_data)
                        if not program_data or len(program_data) == 0:
                            print(f"Skipping {file_path}: Empty JSON or no top-level key. Available keys: {list(program_data.keys())}")
                            continue
                        function_key = list(program_data.keys())[0]
                        nested_data = program_data[function_key]
                        if not isinstance(nested_data, dict):
                            print(f"Skipping {file_path}: Nested data under '{function_key}' is not a dictionary. Available keys: {list(program_data.keys())}")
                            continue
                        if 'program_annotation' not in nested_data or 'schedules_list' not in nested_data:
                            print(f"Skipping {file_path}: Missing 'program_annotation' or 'schedules_list' under '{function_key}'. Available keys: {list(nested_data.keys())}")
                            continue
                        nested_data['file_path'] = file_path
                        nested_data['function_key'] = function_key
                        programs.append(nested_data)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {file_path}. Skipping.")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}. Skipping.")
    print(f"Loaded {len(programs)} valid Tiramisu programs from {folder_path}")
    return programs

def compute_execution_time(execution_times):
    """Compute the average execution time from a list of times."""
    if not execution_times:
        return None
    valid_times = [t for t in execution_times if isinstance(t, (int, float)) and t != float('inf')]
    return np.mean(valid_times) if valid_times else None

def extract_access_patterns(accesses):
    """Extract features from access patterns."""
    if not accesses:
        return []
    features = []
    for access in accesses:
        buffer_id = access.get('buffer_id', -1)
        is_reduction = access.get('access_is_reduction', False)
        matrix = access.get('access_matrix', [])
        matrix_flat = [item for row in matrix for item in row][:10]
        matrix_flat += [0] * (10 - len(matrix_flat))
        access_features = [buffer_id, int(is_reduction)] + matrix_flat
        features.append(access_features)
    padded_features = features[:5]
    padded_features += [[0] * 12] * (5 - len(padded_features))
    return padded_features

def flatten_expression(expr):
    """Flatten an expression tree into a feature vector."""
    if not expr:
        return []
    expr_type = expr.get('expr_type', 'unknown')
    expr_types = ['add', 'mul', 'div', 'sub', 'max', 'min', 'sqrt', 'access', 'value', 'unknown']
    expr_type_encoding = [int(expr_type == et) for et in expr_types]
    children = expr.get('children', [])
    child_features = [flatten_expression(child) for child in children[:2]]
    all_features = expr_type_encoding + [item for sublist in child_features for item in sublist]
    all_features = all_features[:30]
    all_features += [0] * (30 - len(all_features))
    return all_features

def extract_computation_features(computation):
    """Extract features from a computation."""
    if not computation:
        return [0] * 100
    features = [
        computation.get('absolute_order', 0),
        int(computation.get('comp_is_reduction', False))
    ]
    data_types = ['float32', 'float64', 'int32', 'int64', 'unknown']
    data_type = computation.get('data_type', 'unknown')
    data_type_encoding = [int(data_type == dt) for dt in data_types]
    features += data_type_encoding
    iterators = computation.get('iterators', [])
    features.append(len(iterators))
    access_pattern_features = [item for sublist in extract_access_patterns(computation.get('accesses', [])) for item in sublist][:60]
    features += access_pattern_features
    expr_features = flatten_expression(computation.get('expression_representation', {}))
    features += expr_features
    features = features[:100]
    features += [0] * (100 - len(features))
    return features

def extract_schedule_features(schedule):
    """Extract features from a schedule."""
    if not schedule:
        return [0] * 20
    features = []
    tiling = schedule.get('tiling', {})
    features.append(int(bool(tiling)))
    tiling_factors = [tiling.get(f'l{i}_factor', 0) for i in range(3)]
    features += tiling_factors
    unrolling_factor = schedule.get('unrolling_factor', 0)
    features += [int(bool(unrolling_factor)), unrolling_factor]
    parallelized_dim = schedule.get('parallelized_dim', None)
    features.append(int(parallelized_dim is not None))
    if parallelized_dim is not None:
        dim_encoding = [0] * 5
        if isinstance(parallelized_dim, int) and 0 <= parallelized_dim < 5:
            dim_encoding[parallelized_dim] = 1
        elif isinstance(parallelized_dim, str) and parallelized_dim in ['i0', 'i1', 'i2', 'i3', 'i4']:
            dim_encoding[int(parallelized_dim[1])] = 1
        features += dim_encoding
    else:
        features += [0] * 5
    transformations = schedule.get('transformations_list', [])
    features.append(len(transformations))
    features = features[:20]
    features += [0] * (20 - len(features))
    return features

def extract_loop_features(loop_node, iterators_info):
    """Extract features from a loop node."""
    if not loop_node or not iterators_info:
        return [0] * 5
    loop_name = loop_node.get('loop_name', '')
    iterator_info = iterators_info.get(loop_name, {})
    lower_bound = iterator_info.get('lower_bound', 0)
    upper_bound = iterator_info.get('upper_bound', 0)
    if isinstance(lower_bound, str):
        lower_bound = -1
    if isinstance(upper_bound, str):
        upper_bound = -1
    features = [lower_bound, upper_bound]
    parent = iterator_info.get('parent_iterator', None)
    features.append(int(bool(parent)))
    children = iterator_info.get('child_iterators', [])
    features.append(len(children))
    computations = loop_node.get('computations_list', [])
    features.append(len(computations))
    return features

def build_tree_structure(node, iterators_info, computations_info, schedules_info):
    """Recursively build a tree structure with features."""
    if not node:
        return None
    loop_features = extract_loop_features(node, iterators_info)
    computation_features = []
    for comp_name in node.get('computations_list', []):
        if comp_name in computations_info:
            comp_features = extract_computation_features(computations_info[comp_name])
            schedule_features = extract_schedule_features(schedules_info.get(comp_name, {}))
            computation_features.append(comp_features + schedule_features)
    while len(computation_features) < MAX_COMPUTATIONS_PER_NODE:
        computation_features.append([0] * (100 + 20))
    computation_features = computation_features[:MAX_COMPUTATIONS_PER_NODE]
    children = []
    for child in node.get('child_list', [])[:MAX_CHILDREN]:
        child_node = build_tree_structure(child, iterators_info, computations_info, schedules_info)
        if child_node:
            children.append(child_node)
    while len(children) < MAX_CHILDREN:
        children.append(None)
    children = children[:MAX_CHILDREN]
    return {
        'loop_features': loop_features,
        'computation_features': computation_features,
        'children': children
    }

def preprocess_tiramisu_program(program):
    """Preprocess a Tiramisu program, creating one item per schedule."""
    file_path = program.get('file_path', 'unknown')
    annotation = program.get('program_annotation', {})
    if not annotation:
        print(f"Warning: No program annotation found in {file_path}")
        return []
    
    iterators_info = annotation.get('iterators', {})
    computations_info = annotation.get('computations', {})
    
    schedules_list = program.get('schedules_list', [])
    if not schedules_list:
        print(f"Warning: No schedules found in {file_path}")
        return []
    
    preprocessed_items = []
    for schedule_entry in schedules_list:
        if not isinstance(schedule_entry, dict):
            print(f"Skipping schedule in {file_path}: Not a dictionary")
            continue
        
        tree_structure_data = schedule_entry.get('tree_structure', {})
        if 'roots' not in tree_structure_data:
            print(f"Skipping schedule in {file_path}: No 'roots' in tree_structure")
            continue
        tree_roots = tree_structure_data['roots']
        if not tree_roots:
            print(f"Skipping schedule in {file_path}: Empty 'roots' in tree_structure")
            continue
        
        execution_times = schedule_entry.get('execution_times', [])
        execution_time = compute_execution_time(execution_times)
        if execution_time is None or not isinstance(execution_time, (int, float)):
            print(f"Skipping schedule in {file_path}: Invalid execution time {execution_time}")
            continue
        
        schedules_info = {
            comp_name: comp_schedule 
            for comp_name, comp_schedule in schedule_entry.items() 
            if comp_name not in ['tree_structure', 'execution_times', 'fusions', 'sched_str', 'legality_check', 'exploration_method']
        }
        
        tree_structure = []
        for root in tree_roots:
            tree_node = build_tree_structure(root, iterators_info, computations_info, schedules_info)
            if tree_node:
                tree_structure.append(tree_node)
        
        if tree_structure:
            preprocessed_items.append({
                'tree_structure': tree_structure,
                'execution_time': float(execution_time),  # Ensure it's a float
                'file_path': file_path,
                'schedule_entry': schedule_entry
            })
    
    if not preprocessed_items:
        print(f"Warning: No valid schedules processed in {file_path}")
    return preprocessed_items

def flatten_tree_to_sequence(tree_node, depth=0):
    """Flatten a tree structure into a sequence of nodes."""
    if tree_node is None:
        return []
    node_seq = [{
        'loop_features': tree_node['loop_features'],
        'computation_features': tree_node['computation_features'],
        'depth': depth
    }]
    for child in tree_node['children']:
        if child is not None:
            node_seq.extend(flatten_tree_to_sequence(child, depth + 1))
    return node_seq

### Neural Network Model
class TreeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_i = nn.Linear(input_size, hidden_size)
        self.U_i = nn.Linear(hidden_size, hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)
        self.W_u = nn.Linear(input_size, hidden_size)
        self.U_u = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x, child_h, child_c):
        device = x.device
        if not child_h:
            child_h = [torch.zeros(1, self.hidden_size, device=device)]
            child_c = [torch.zeros(1, self.hidden_size, device=device)]
        h_sum = torch.sum(torch.stack(child_h), dim=0)
        i = torch.sigmoid(self.W_i(x) + self.U_i(h_sum))
        f_k = [torch.sigmoid(self.W_f(x) + self.U_f(h_k)) for h_k in child_h]
        o = torch.sigmoid(self.W_o(x) + self.U_o(h_sum))
        u = torch.tanh(self.W_u(x) + self.U_u(h_sum))
        c = i * u
        for f, c_k in zip(f_k, child_c):
            c = c + f * c_k
        h = o * torch.tanh(c)
        return h, c

class RecursiveTreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RecursiveTreeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.loop_embedding = nn.Linear(5, hidden_size // 4)
        comp_sched_feature_size = 100 + 20
        self.comp_embedding = nn.Linear(comp_sched_feature_size, hidden_size // 4)
        self.cell = TreeLSTMCell(hidden_size, hidden_size)
    
    def forward(self, tree_node, device):
        if tree_node is None:
            return (torch.zeros(1, self.hidden_size, device=device), torch.zeros(1, self.hidden_size, device=device))
        
        child_h, child_c = [], []
        for child in tree_node['children']:
            if child is not None:
                h_k, c_k = self.forward(child, device)
                child_h.append(h_k)
                child_c.append(c_k)
        
        loop_features = torch.tensor(tree_node['loop_features'], dtype=torch.float32, device=device).unsqueeze(0)
        loop_embed = self.loop_embedding(loop_features)
        
        comp_embeds = []
        for comp_features in tree_node['computation_features']:
            comp_tensor = torch.tensor(comp_features, dtype=torch.float32, device=device).unsqueeze(0)
            comp_embed = self.comp_embedding(comp_tensor)
            comp_embeds.append(comp_embed)
        
        comp_combined = torch.mean(torch.stack(comp_embeds), dim=0) if comp_embeds else torch.zeros(1, self.hidden_size // 4, device=device)
        padding = torch.zeros(1, self.hidden_size // 2, device=device)
        x = torch.cat([loop_embed, comp_combined, padding], dim=1)
        h, c = self.cell(x, child_h, child_c)
        return h, c

class SequenceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SequenceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return output, (h_n, c_n)

class TiramisuCostModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TiramisuCostModel, self).__init__()
        self.hidden_size = hidden_size
        self.tree_lstm = RecursiveTreeLSTM(input_size, hidden_size)
        # Corrected input_size to 126 (5 from loop_features + 120 from comp_combined + 1 from depth)
        self.seq_lstm = SequenceLSTM(126, hidden_size)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, tree_structure, device):
        if not tree_structure:
            return torch.tensor([[0.0]], device=device)
        
        tree_outputs = []
        for tree in tree_structure:
            if tree is not None:
                h, _ = self.tree_lstm(tree, device)
                tree_outputs.append(h)
        
        tree_encoding = torch.mean(torch.stack(tree_outputs), dim=0) if tree_outputs else torch.zeros(1, self.hidden_size, device=device)
        
        seq_features = []
        for tree in tree_structure:
            if tree is not None:
                seq_nodes = flatten_tree_to_sequence(tree)
                for node in seq_nodes:
                    loop_features = torch.tensor(node['loop_features'], dtype=torch.float32, device=device)
                    comp_tensors = [torch.tensor(comp, dtype=torch.float32, device=device) for comp in node['computation_features']]
                    comp_combined = torch.mean(torch.stack(comp_tensors), dim=0) if comp_tensors else torch.zeros(120, device=device)
                    depth = torch.tensor([node['depth']], dtype=torch.float32, device=device)
                    node_features = torch.cat([loop_features, comp_combined, depth])
                    seq_features.append(node_features)
        
        if seq_features:
            seq_tensor = torch.stack(seq_features).unsqueeze(0)
            _, (h_n, _) = self.seq_lstm(seq_tensor)
            seq_encoding = h_n.squeeze(0)
        else:
            seq_encoding = torch.zeros(1, self.hidden_size, device=device)
        
        combined = torch.cat([tree_encoding, seq_encoding], dim=1)
        hidden = F.relu(self.fc1(combined))
        execution_time = self.fc2(hidden)
        return execution_time

### Dataset and DataLoader
class TiramisuDataset(Dataset):
    def __init__(self, preprocessed_items):
        self.items = [item for item in preprocessed_items if item['execution_time'] is not None and isinstance(item['execution_time'], (int, float))]
        print(f"Created dataset with {len(self.items)} valid schedule items")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            'tree_structure': item['tree_structure'],
            'execution_time': item['execution_time'],
            'file_path': item['file_path']
        }

def collate_fn(batch):
    """Custom collate function to filter out invalid items."""
    filtered_batch = [item for item in batch if item['execution_time'] is not None and isinstance(item['execution_time'], (int, float))]
    if len(filtered_batch) < len(batch):
        print(f"Warning: Filtered out {len(batch) - len(filtered_batch)} items with invalid execution times in batch")
    if not filtered_batch:
        print("Warning: Entire batch had invalid execution times")
        return []
    return filtered_batch

### Training and Evaluation
def train_tiramisu_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE):
    """Train the Tiramisu cost model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)"):
            if not batch:
                continue
            
            optimizer.zero_grad()
            batch_loss = 0
            batch_item_count = 0
            
            for item in batch:
                try:
                    tree_structure = item['tree_structure']
                    if not tree_structure:
                        continue
                    
                    true_time = torch.tensor([item['execution_time']], dtype=torch.float32, device=device)
                    pred_time = model(tree_structure, device).squeeze()
                    loss = criterion(pred_time, true_time)
                    batch_loss += loss
                    batch_item_count += 1
                except Exception as e:
                    print(f"Error processing training item: {str(e)}")
                    print(f"Item path: {item['file_path']}")
                    continue
            
            if batch_item_count > 0:
                batch_loss /= batch_item_count
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()
                batch_count += 1
        
        if batch_count > 0:
            avg_train_loss = epoch_loss / batch_count
            train_losses.append(avg_train_loss)
        else:
            print("Warning: No valid batches in training epoch")
            train_losses.append(float('inf'))
        
        model.eval()
        val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)"):
                if not batch:
                    continue
                
                batch_loss = 0
                batch_item_count = 0
                
                for item in batch:
                    try:
                        tree_structure = item['tree_structure']
                        if not tree_structure:
                            continue
                        
                        true_time = torch.tensor([item['execution_time']], dtype=torch.float32, device=device)
                        pred_time = model(tree_structure, device).squeeze()
                        loss = criterion(pred_time, true_time)
                        batch_loss += loss
                        batch_item_count += 1
                    except Exception as e:
                        print(f"Error processing validation item: {str(e)}")
                        continue
                
                if batch_item_count > 0:
                    batch_loss /= batch_item_count
                    val_loss += batch_loss.item()
                    val_batch_count += 1
        
        if val_batch_count > 0:
            avg_val_loss = val_loss / val_batch_count
            val_losses.append(avg_val_loss)
        else:
            print("Warning: No valid batches in validation")
            val_losses.append(float('inf'))
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}")
        
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            torch.save(model.state_dict(), "best_tiramisu_model.pth")
            print(f"Saved best model with validation loss: {best_val_loss:.6f}")
    
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
    """Evaluate the trained model on the test set."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0
    true_times = []
    pred_times = []
    batch_count = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            if not batch:
                continue
            
            batch_loss = 0
            batch_item_count = 0
            
            for item in batch:
                try:
                    tree_structure = item['tree_structure']
                    if not tree_structure:
                        continue
                    
                    true_time = torch.tensor([item['execution_time']], dtype=torch.float32, device=device)
                    pred_time = model(tree_structure, device).squeeze()
                    loss = criterion(pred_time, true_time)
                    batch_loss += loss
                    batch_item_count += 1
                    
                    true_times.append(true_time.item())
                    pred_times.append(pred_time.item())
                except Exception as e:
                    print(f"Error processing test item: {str(e)}")
                    continue
            
            if batch_item_count > 0:
                batch_loss /= batch_item_count
                test_loss += batch_loss.item()
                batch_count += 1
    
    if batch_count > 0:
        avg_test_loss = test_loss / batch_count
    else:
        print("Warning: No valid batches in testing")
        avg_test_loss = float('inf')
    
    if true_times and pred_times:
        true_times = np.array(true_times)
        pred_times = np.array(pred_times)
        mae = np.mean(np.abs(true_times - pred_times))
        mape = np.mean(np.abs((true_times - pred_times) / true_times)) * 100
        print(f"Test Loss (MSE): {avg_test_loss:.6f}")
        print(f"Mean Absolute Error (MAE): {mae:.6f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        plt.figure(figsize=(10, 5))
        plt.scatter(true_times, pred_times, alpha=0.5)
        plt.plot([min(true_times), max(true_times)], [min(true_times), max(true_times)], 'r--')
        plt.xlabel('True Execution Time')
        plt.ylabel('Predicted Execution Time')
        plt.title('True vs Predicted Execution Times')
        plt.savefig('prediction_scatter.png')
        plt.close()
    else:
        print("Warning: No valid predictions made during evaluation")
        mae, mape = float('inf'), float('inf')
    
    return avg_test_loss, mae, mape

### Main Execution
if __name__ == "__main__":
    folder_path = "./Tiramisu"
    if not os.path.exists(folder_path):
        print(f"Directory {folder_path} does not exist. Please create it and add your JSON files.")
        exit(1)
    
    raw_programs = load_tiramisu_programs(folder_path)
    if not raw_programs:
        print("No valid programs loaded. Ensure each JSON file has a top-level key (e.g., 'function003306') containing 'program_annotation' and 'schedules_list'. Exiting.")
        exit(1)
    
    preprocessed_items = []
    for program in raw_programs:
        items = preprocess_tiramisu_program(program)
        preprocessed_items.extend(items)
    
    if not preprocessed_items:
        print("No valid schedule items found. Verify 'tree_structure' and 'execution_times' in schedules_list. Exiting.")
        exit(1)
    
    train_items, temp_items = train_test_split(preprocessed_items, test_size=0.3, random_state=42)
    val_items, test_items = train_test_split(temp_items, test_size=0.5, random_state=42)
    
    train_dataset = TiramisuDataset(train_items)
    val_dataset = TiramisuDataset(val_items)
    test_dataset = TiramisuDataset(test_items)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print("One or more datasets are empty. Verify input data. Exiting.")
        exit(1)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    input_size = HIDDEN_DIM
    model = TiramisuCostModel(input_size, HIDDEN_DIM)
    
    trained_model, train_losses, val_losses = train_tiramisu_model(model, train_loader, val_loader)
    
    test_loss, mae, mape = evaluate_model(trained_model, test_loader)
    
    print(f"\nFinal Test Results:")
    print(f"Test Loss (MSE): {test_loss:.6f}")
    print(f"Mean Absolute Error (MAE): {mae:.6f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
