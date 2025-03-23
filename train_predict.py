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

###########################################
# Data Loading and Preprocessing
###########################################

def load_tiramisu_programs(folder_path):
    """Load all Tiramisu programs from a folder."""
    programs = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        program_data = json.load(f)
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
    return exploration_trace.get('evaluation', None)

def extract_best_schedule(exploration_trace, depth_limit=None):
    """
    Recursively find the best schedule in the exploration trace.
    Returns a tuple of (schedule, execution_time)
    """
    if exploration_trace is None or not exploration_trace:
        return None, None  # Changed from float('inf') to None for clarity
    
    current_schedule = exploration_trace.get('schedule', '')
    current_time = exploration_trace.get('evaluation', None)  # Allow None initially
    current_depth = exploration_trace.get('depth', 0)
    
    if depth_limit is not None and current_depth >= depth_limit:
        return current_schedule, current_time
    
    best_schedule, best_time = current_schedule, current_time
    children = exploration_trace.get('children', [])
    
    for child in children:
        child_schedule, child_time = extract_best_schedule(child, depth_limit)
        if child_time is not None and (best_time is None or child_time < best_time):
            best_schedule, best_time = child_schedule, child_time
    
    return best_schedule, best_time

def extract_access_patterns(accesses):
    """Extract features from access patterns."""
    if not accesses:
        return []
    features = []
    for access in accesses:
        buffer_id = access.get('buffer_id', -1)
        is_reduction = access.get('access_is_reduction', False)
        matrix = access.get('access_matrix', [])
        matrix_flat = []
        for row in matrix:
            matrix_flat.extend(row)
        matrix_flat = matrix_flat[:10]
        while len(matrix_flat) < 10:
            matrix_flat.append(0)
        access_features = [buffer_id, int(is_reduction)] + matrix_flat
        features.append(access_features)
    padded_features = features[:5]
    while len(padded_features) < 5:
        padded_features.append([0] * 12)
    return padded_features

def flatten_expression(expr):
    """Flatten an expression tree into a feature vector."""
    if not expr:
        return []
    expr_type = expr.get('expr_type', 'unknown')
    expr_types = ['add', 'mul', 'div', 'sub', 'max', 'min', 'sqrt', 'access', 'value', 'unknown']
    expr_type_encoding = [0] * len(expr_types)
    if expr_type in expr_types:
        expr_type_encoding[expr_types.index(expr_type)] = 1
    children = expr.get('children', [])
    child_features = []
    for child in children[:2]:
        child_features.extend(flatten_expression(child))
    all_features = expr_type_encoding + child_features
    all_features = all_features[:30]
    while len(all_features) < 30:
        all_features.append(0)
    return all_features

def extract_computation_features(computation):
    """Extract features from a computation."""
    features = []
    features.append(computation.get('absolute_order', 0))
    features.append(1 if computation.get('comp_is_reduction', False) else 0)
    data_types = ['float32', 'float64', 'int32', 'int64', 'unknown']
    data_type = computation.get('data_type', 'unknown')
    data_type_encoding = [0] * len(data_types)
    if data_type in data_types:
        data_type_encoding[data_types.index(data_type)] = 1
    features.extend(data_type_encoding)
    iterators = computation.get('iterators', [])
    features.append(len(iterators))
    access_pattern_features = []
    for access_pattern in extract_access_patterns(computation.get('accesses', [])):
        access_pattern_features.extend(access_pattern)
    features.extend(access_pattern_features[:60])
    expr_features = flatten_expression(computation.get('expression_representation', {}))
    features.extend(expr_features)
    return features

def extract_schedule_features(schedule):
    """Extract features from a schedule."""
    if not schedule:
        return [0] * 20
    features = []
    tiling = schedule.get('tiling', {})
    has_tiling = 1 if tiling else 0
    features.append(has_tiling)
    tiling_factors = []
    for i in range(3):
        if i < len(tiling.keys()):
            factor = list(tiling.values())[i]
            tiling_factors.append(factor)
        else:
            tiling_factors.append(0)
    features.extend(tiling_factors)
    unrolling_factor = schedule.get('unrolling_factor', 0)
    features.append(1 if unrolling_factor else 0)
    features.append(unrolling_factor if unrolling_factor else 0)
    parallelized_dim = schedule.get('parallelized_dim', None)
    features.append(1 if parallelized_dim is not None else 0)
    if parallelized_dim is not None:
        dim_encoding = [0] * 5
        if isinstance(parallelized_dim, int) and 0 <= parallelized_dim < 5:
            dim_encoding[parallelized_dim] = 1
        features.extend(dim_encoding)
    else:
        features.extend([0] * 5)
    transformations = schedule.get('transformations_list', [])
    features.append(len(transformations))
    features = features[:20]
    while len(features) < 20:
        features.append(0)
    return features

def extract_loop_features(loop_node, iterators_info):
    """Extract features from a loop node."""
    loop_name = loop_node.get('loop_name', '')
    iterator_info = iterators_info.get(loop_name, {})
    features = []
    lower_bound = iterator_info.get('lower_bound', 0)
    upper_bound = iterator_info.get('upper_bound', 0)
    if isinstance(lower_bound, str):
        lower_bound = -1
    if isinstance(upper_bound, str):
        upper_bound = -1
    features.append(lower_bound)
    features.append(upper_bound)
    parent = iterator_info.get('parent', None)
    features.append(1 if parent else 0)
    children = iterator_info.get('children', [])
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
        computation_features.append([0] * (len(extract_computation_features({})) + 
                                          len(extract_schedule_features({}))))
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
    """Preprocess a single Tiramisu program."""
    annotation = program.get('program_annotation', {})
    iterators_info = {}
    for it_name, it_info in annotation.get('iterators', {}).items():
        iterators_info[it_name] = {
            'lower_bound': it_info.get('lower_bound', 0),
            'upper_bound': it_info.get('upper_bound', 0),
            'parent': it_info.get('parent_iterator', None),
            'children': it_info.get('child_iterators', [])
        }
    computations_info = annotation.get('computations', {})
    schedules_info = {}
    for schedule_entry in program.get('schedules_list', []):
        for comp_name, comp_schedule in schedule_entry.items():
            if comp_name not in ['fusions', 'sched_str', 'tree_structure', 'legality_check', 'exploration_method']:
                schedules_info[comp_name] = comp_schedule
    tree_roots = []
    for schedule_entry in program.get('schedules_list', []):
        if 'tree_structure' in schedule_entry and 'roots' in schedule_entry['tree_structure']:
            tree_roots = schedule_entry['tree_structure']['roots']
            break
    tree_structure = []
    for root in tree_roots:
        tree_node = build_tree_structure(root, iterators_info, computations_info, schedules_info)
        if tree_node:
            tree_structure.append(tree_node)
    
    exploration_trace = program.get('exploration_trace', {})
    best_schedule, execution_time = extract_best_schedule(exploration_trace)
    
    # Debugging output
    file_path = program.get('file_path', 'unknown')
    if execution_time is None:
        print(f"Warning: No valid execution time found in {file_path}. Exploration trace: {exploration_trace}")
    elif not isinstance(execution_time, (int, float)) or execution_time == float('inf'):
        print(f"Warning: Invalid execution time {execution_time} in {file_path}")
    
    return {
        'tree_structure': tree_structure,
        'execution_time': execution_time,
        'best_schedule': best_schedule,
        'file_path': file_path
    }

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

###########################################
# Neural Network Model
###########################################

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
        if not child_h:
            child_h = [torch.zeros(1, self.hidden_size).to(x.device)]
            child_c = [torch.zeros(1, self.hidden_size).to(x.device)]
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
        comp_sched_feature_size = len(extract_computation_features({})) + len(extract_schedule_features({}))
        self.comp_embedding = nn.Linear(comp_sched_feature_size, hidden_size // 4)
        self.cell = TreeLSTMCell(hidden_size, hidden_size)
    
    def forward(self, tree_node):
        if tree_node is None:
            return (torch.zeros(1, self.hidden_size), torch.zeros(1, self.hidden_size))
        child_h, child_c = [], []
        for child in tree_node['children']:
            if child is not None:
                h_k, c_k = self.forward(child)
                child_h.append(h_k)
                child_c.append(c_k)
        loop_features = torch.tensor(tree_node['loop_features'], dtype=torch.float32).unsqueeze(0)
        loop_embed = self.loop_embedding(loop_features)
        comp_embeds = []
        for comp_features in tree_node['computation_features']:
            comp_tensor = torch.tensor(comp_features, dtype=torch.float32).unsqueeze(0)
            comp_embed = self.comp_embedding(comp_tensor)
            comp_embeds.append(comp_embed)
        comp_combined = torch.mean(torch.stack(comp_embeds), dim=0) if comp_embeds else torch.zeros(1, self.hidden_size // 4)
        padding = torch.zeros(1, self.hidden_size // 2)
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
        self.seq_lstm = SequenceLSTM(input_size + 1, hidden_size)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, tree_structure):
        tree_outputs = []
        for tree in tree_structure:
            if tree is not None:
                h, _ = self.tree_lstm(tree)
                tree_outputs.append(h)
        tree_encoding = torch.mean(torch.stack(tree_outputs), dim=0) if tree_outputs else torch.zeros(1, self.hidden_size)
        seq_features = []
        for tree in tree_structure:
            if tree is not None:
                seq_nodes = flatten_tree_to_sequence(tree)
                for node in seq_nodes:
                    loop_features = torch.tensor(node['loop_features'], dtype=torch.float32)
                    comp_tensors = [torch.tensor(comp, dtype=torch.float32) for comp in node['computation_features']]
                    comp_combined = torch.mean(torch.stack(comp_tensors), dim=0) if comp_tensors else torch.zeros(len(extract_computation_features({})) + len(extract_schedule_features({})))
                    depth = torch.tensor([node['depth']], dtype=torch.float32)
                    node_features = torch.cat([loop_features, comp_combined, depth])
                    seq_features.append(node_features)
        if seq_features:
            seq_tensor = torch.stack(seq_features).unsqueeze(0)
            _, (h_n, _) = self.seq_lstm(seq_tensor)
            seq_encoding = h_n.squeeze(0)
        else:
            seq_encoding = torch.zeros(1, self.hidden_size)
        combined = torch.cat([tree_encoding, seq_encoding], dim=1)
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
        return {
            'tree_structure': program['tree_structure'],
            'execution_time': program['execution_time'],
            'file_path': program['file_path']
        }

def collate_fn(batch):
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
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)"):
            optimizer.zero_grad()
            batch_loss = 0
            for item in batch:
                tree_structure = item['tree_structure']
                true_time = torch.tensor(item['execution_time'], dtype=torch.float32).to(device)
                pred_time = model(tree_structure).squeeze()
                loss = criterion(pred_time, true_time)
                batch_loss += loss
            batch_loss /= len(batch)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)"):
                batch_loss = 0
                for item in batch:
                    tree_structure = item['tree_structure']
                    true_time = torch.tensor(item['execution_time'], dtype=torch.float32).to(device)
                    pred_time = model(tree_structure).squeeze()
                    loss = criterion(pred_time, true_time)
                    batch_loss += loss
                batch_loss /= len(batch)
                val_loss += batch_loss.item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0
    true_times = []
    pred_times = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            batch_loss = 0
            for item in batch:
                tree_structure = item['tree_structure']
                true_time = torch.tensor(item['execution_time'], dtype=torch.float32).to(device)
                pred_time = model(tree_structure).squeeze()
                loss = criterion(pred_time, true_time)
                batch_loss += loss
                true_times.append(true_time.item())
                pred_times.append(pred_time.item())
            batch_loss /= len(batch)
            test_loss += batch_loss.item()
    avg_test_loss = test_loss / len(test_loader)
    true_times = np.array(true_times)
    pred_times = np.array(pred_times)
    mae = np.mean(np.abs(true_times - pred_times))
    mape = np.mean(np.abs((true_times - pred_times) / true_times)) * 100
    ss_tot = np.sum((true_times - np.mean(true_times)) ** 2)
    ss_res = np.sum((true_times - pred_times) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"Test Loss (MSE): {avg_test_loss:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"R^2 Score: {r2:.4f}")
    plt.figure(figsize=(10, 6))
    plt.scatter(true_times, pred_times, alpha=0.5)
    plt.plot([min(true_times), max(true_times)], [min(true_times), max(true_times)], 'r--', label='Perfect Prediction')
    plt.xlabel('True Execution Time')
    plt.ylabel('Predicted Execution Time')
    plt.title('Predicted vs Actual Execution Times')
    plt.legend()
    plt.savefig('prediction_vs_actual.png')
    plt.close()
    return avg_test_loss, mae, mape, r2

# Main execution block
if __name__ == "__main__":
    folder_path = "./Tiramisu"  # Matches your setup: ~/jathu/Halide_New/Tiramisu
    if not os.path.exists(folder_path):
        print(f"Error: The folder path '{folder_path}' does not exist.")
        exit(1)

    programs = load_tiramisu_programs(folder_path)
    if not programs:
        print("Error: No Tiramisu programs loaded. Check the 'Tiramisu' folder for valid JSON files.")
        exit(1)
    
    preprocessed_programs = []
    for program in programs:
        preprocessed = preprocess_tiramisu_program(program)
        # Only include programs with valid numeric execution times
        if preprocessed['execution_time'] is not None and isinstance(preprocessed['execution_time'], (int, float)) and preprocessed['execution_time'] != float('inf'):
            preprocessed_programs.append(preprocessed)
    
    print(f"Preprocessed {len(preprocessed_programs)} programs with valid execution times")
    if not preprocessed_programs:
        print("Error: No programs with valid execution times found. Check the JSON files for 'exploration_trace' with numeric 'evaluation' values.")
        print("Sample a few JSON files to verify their structure:")
        for i, program in enumerate(programs[:3]):  # Show first 3 for debugging
            print(f"Program {i+1} from {program['file_path']}:")
            print(f"  Exploration trace: {program.get('exploration_trace', 'Missing')}")
        exit(1)

    train_val_programs, test_programs = train_test_split(preprocessed_programs, test_size=0.2, random_state=42)
    train_programs, val_programs = train_test_split(train_val_programs, test_size=0.25, random_state=42)

    print(f"Training set: {len(train_programs)} programs")
    print(f"Validation set: {len(val_programs)} programs")
    print(f"Test set: {len(test_programs)} programs")

    train_dataset = TiramisuDataset(train_programs)
    val_dataset = TiramisuDataset(val_programs)
    test_dataset = TiramisuDataset(test_programs)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    input_size = (len(extract_computation_features({})) + len(extract_schedule_features({})))
    model = TiramisuCostModel(input_size, HIDDEN_DIM)

    print("Starting training...")
    model, train_losses, val_losses = train_tiramisu_model(model, train_loader, val_loader)

    print("\nEvaluating on test set...")
    test_loss, mae, mape, r2 = evaluate_model(model, test_loader)

    torch.save(model.state_dict(), "final_tiramisu_model.pth")
    print("Training and evaluation completed. Final model saved as 'final_tiramisu_model.pth'")
