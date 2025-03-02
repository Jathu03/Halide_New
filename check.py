import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Constants from data_utils (assumed values, adjust as needed)
MAX_NUM_TRANSFORMATIONS = 2  # Number of transformation types (e.g., tile, vectorize)
MAX_TAGS = 5  # Number of tags per transformation (e.g., parameters)

def get_execution_time(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw_content = f.read()
            content = raw_content.decode('utf-8', errors='replace').replace('\0', '')
            data = json.loads(content)
        schedules = data["scheduling_data"]
        for item in schedules:
            if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
                execution_time = item.get('value')
                if execution_time is not None:
                    return float(execution_time)
        print(f"Warning: 'total_execution_time_ms' not found in 'Schedules' of {file_path}")
        return schedules[-1]["value"]
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def extract_optimized_features(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        execution_time = get_execution_time(file_path)
        if execution_time is None or execution_time < 0:
            print(f"Invalid execution time in {file_path}: {execution_time}")
            return None
        
        nodes = data["programming_details"]["Nodes"]
        edges = data["programming_details"]["Edges"]
        scheduling = data["scheduling_data"]
        
        node_dict = {node["Name"]: node["Details"] for node in nodes}
        sched_dict = {item["Name"]: item["Details"]["scheduling_feature"] for item in scheduling if "Name" in item and "Details" in item}
        
        # Tree structure: single root with all computations as a flat level
        tree = {
            "roots": [{
                "has_comps": True,
                "computations_indices": torch.tensor(list(range(len(nodes))), dtype=torch.long),
                "child_list": [],  # No child loops for simplicity
                "loop_index": torch.tensor([0], dtype=torch.long)  # Single loop level
            }]
        }
        
        # Computation features (first part: operation counts)
        op_counts = []
        for node_name, details in node_dict.items():
            op_hist = details.get("Op histogram", [])
            ops = {'add': 0, 'mul': 0, 'div': 0, 'min': 0, 'max': 0}
            for entry in op_hist:
                parts = entry.split(':')
                if len(parts) == 2:
                    op_name = parts[0].strip().lower()
                    op_count = int(parts[1].strip().split()[0])
                    if op_name in ops:
                        ops[op_name] = op_count
            op_counts.append([ops['add'], ops['mul'], ops['div'], ops['min'], ops['max']])
        comps_tensor_first_part = torch.tensor(op_counts, dtype=torch.float32)  # [num_comps, 5]
        
        # Transformation vectors (simplified as scheduling features)
        comps_tensor_vectors = []
        for node_name in node_dict:
            sched = sched_dict.get(node_name, {})
            vector = [
                1 if sched.get("inner_parallelism", 1.0) > 1.0 else 0,
                1 if sched.get("unrolled_loop_extent", 1.0) > 1.0 else 0,
                1 if sched.get("vector_size", 1.0) > 1.0 else 0,
                0, 0  # Pad to MAX_TAGS=5
            ]
            comps_tensor_vectors.append(vector[:MAX_TAGS])
        comps_tensor_vectors = torch.tensor(comps_tensor_vectors, dtype=torch.float32)  # [num_comps, MAX_TAGS]
        
        # Third part: structural features per computation
        comps_tensor_third_part = []
        for node in nodes:
            inputs = len([e for e in edges if e["To"] == node["Name"]])
            reductions = 1 if any(".update(" in e["To"] for e in edges if e["To"] == node["Name"]) else 0
            comps_tensor_third_part.append([inputs, reductions])
        comps_tensor_third_part = torch.tensor(comps_tensor_third_part, dtype=torch.float32)  # [num_comps, 2]
        
        # Loops tensor (simplified to scheduling aggregates)
        loops_tensor = torch.tensor([[
            sum(1 for sched in sched_dict.values() if sched.get("inner_parallelism", 1.0) > 1.0),
            sum(sched.get("unrolled_loop_extent", 1.0) for sched in sched_dict.values()),
            sum(sched.get("vector_size", 1.0) for sched in sched_dict.values()),
            sum(1 for sched in sched_dict.values() if sched.get("unrolled_loop_extent", 1.0) > 1.0),
            len(nodes), len(edges),
            sum(sched.get('bytes_at_production', 0) for sched in sched_dict.values()),
            sum(sched.get('num_vectors', 0) for sched in sched_dict.values())
        ]], dtype=torch.float32)  # [1, loops_tensor_size=8]
        
        # Expression embedding (simplified as operation counts again)
        functions_comps_expr_tree = comps_tensor_first_part.unsqueeze(1).expand(-1, 1, -1)  # [num_comps, seq_len=1, 5]
        # Pad to match expected input size (11)
        padding = torch.zeros(functions_comps_expr_tree.shape[0], 1, 11 - functions_comps_expr_tree.shape[2])
        functions_comps_expr_tree = torch.cat((functions_comps_expr_tree, padding), dim=2)  # [num_comps, 1, 11]
        
        tree_tensors = (
            tree,
            comps_tensor_first_part.unsqueeze(0),  # [1, num_comps, 5]
            comps_tensor_vectors.unsqueeze(0),     # [1, num_comps, MAX_TAGS]
            comps_tensor_third_part.unsqueeze(0),  # [1, num_comps, 2]
            loops_tensor,                          # [1, 8]
            functions_comps_expr_tree.unsqueeze(0) # [1, num_comps, 1, 11]
        )
        
        return {"tree_tensors": tree_tensors, "execution_time": execution_time}
    
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

def process_main_directory(main_dir):
    train_data = []
    test_data = []
    test_file_names = []
    
    subdirs = sorted([d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))])
    
    if not subdirs:
        raise ValueError(f"No subdirectories found in {main_dir}")
    
    for subdir in subdirs:
        subdir_path = os.path.join(main_dir, subdir)
        all_data = []
        all_file_names = []
        
        for filename in sorted(os.listdir(subdir_path)):
            if filename.endswith('.json'):
                file_path = os.path.join(subdir_path, filename)
                data = extract_optimized_features(file_path)
                if data is not None:
                    all_data.append(data)
                    all_file_names.append(filename)
        
        if len(all_data) != 32:
            print(f"Warning: Expected 32 files in {subdir_path}, found {len(all_data)}")
            continue
        
        train_data.extend(all_data[:30])
        test_data.extend(all_data[30:])
        test_file_names.extend([os.path.join(subdir, fname) for fname in all_file_names[30:]])
        print(f"Processed {len(all_data)} files from {subdir}: 30 for training, 2 for testing")
    
    return train_data, test_data, test_file_names

def prepare_data_for_model(train_data, test_data):
    # Separate tree tensors and execution times
    X_train = [item["tree_tensors"] for item in train_data]
    y_train = torch.tensor([item["execution_time"] for item in train_data], dtype=torch.float32).view(-1, 1)
    X_test = [item["tree_tensors"] for item in test_data]
    y_test = torch.tensor([item["execution_time"] for item in test_data], dtype=torch.float32).view(-1, 1)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.numpy())
    y_test_scaled = scaler_y.transform(y_test.numpy())
    
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    return X_train, y_train_tensor, X_test, y_test_tensor, scaler_y

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16):
    train_dataset = TensorDataset(torch.tensor([0]*len(X_train), dtype=torch.long), y_train)  # Dummy indices
    test_dataset = TensorDataset(torch.tensor([0]*len(X_test), dtype=torch.long), y_test)     # Dummy indices
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, X_train, X_test

class Model_Recursive_LSTM_v2(nn.Module):
    def __init__(
        self,
        input_size=5,  # Adjusted for comps_tensor_first_part
        comp_embed_layer_sizes=[600, 350, 200, 180],
        drops=[0.225, 0.225, 0.225, 0.225],
        output_size=1,
        lstm_embedding_size=100,
        expr_embed_size=100,
        loops_tensor_size=8,
        device="cpu",
        num_layers=1,
        bidirectional=True,
    ):
        super().__init__()
        self.device = device
        embedding_size = comp_embed_layer_sizes[-1]
        
        regression_layer_sizes = [embedding_size] + comp_embed_layer_sizes[-2:]
        concat_layer_sizes = [embedding_size * 2 + loops_tensor_size] + comp_embed_layer_sizes[-2:]
        
        comp_embed_layer_sizes = [
            input_size + lstm_embedding_size * (2 if bidirectional else 1) * num_layers + expr_embed_size
        ] + comp_embed_layer_sizes
        
        self.comp_embedding_layers = nn.ModuleList()
        self.comp_embedding_dropouts = nn.ModuleList()
        self.regression_layers = nn.ModuleList()
        self.regression_dropouts = nn.ModuleList()
        self.concat_layers = nn.ModuleList()
        self.concat_dropouts = nn.ModuleList()
        
        self.encode_vectors = nn.Linear(MAX_TAGS, MAX_TAGS, bias=True)
        for i in range(len(comp_embed_layer_sizes) - 1):
            self.comp_embedding_layers.append(
                nn.Linear(comp_embed_layer_sizes[i], comp_embed_layer_sizes[i + 1], bias=True)
            )
            nn.init.xavier_uniform_(self.comp_embedding_layers[i].weight)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))
        for i in range(len(regression_layer_sizes) - 1):
            self.regression_layers.append(
                nn.Linear(regression_layer_sizes[i], regression_layer_sizes[i + 1], bias=True)
            )
            nn.init.xavier_uniform_(self.regression_layers[i].weight)
            self.regression_dropouts.append(nn.Dropout(drops[i]))
        for i in range(len(concat_layer_sizes) - 1):
            self.concat_layers.append(
                nn.Linear(concat_layer_sizes[i], concat_layer_sizes[i + 1], bias=True)
            )
            nn.init.xavier_uniform_(self.concat_layers[i].weight)
            nn.init.zeros_(self.concat_layers[i].weight)
            self.concat_dropouts.append(nn.Dropout(drops[i]))
        
        self.predict = nn.Linear(regression_layer_sizes[-1], output_size, bias=True)
        nn.init.xavier_uniform_(self.predict.weight)
        self.ELU = nn.ELU()
        self.LeakyReLU = nn.LeakyReLU(0.01)
        
        self.no_comps_tensor = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, embedding_size)))
        self.no_nodes_tensor = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, embedding_size)))
        
        self.comps_lstm = nn.LSTM(comp_embed_layer_sizes[-1], embedding_size, batch_first=True)
        self.nodes_lstm = nn.LSTM(comp_embed_layer_sizes[-1], embedding_size, batch_first=True)
        self.roots_lstm = nn.LSTM(comp_embed_layer_sizes[-1], embedding_size, batch_first=True)
        self.transformation_vectors_embed = nn.LSTM(
            MAX_TAGS, lstm_embedding_size, batch_first=True, bidirectional=bidirectional, num_layers=num_layers
        )
        self.exprs_embed = nn.LSTM(11, expr_embed_size, batch_first=True)
    
    def get_hidden_state(self, node, comps_embeddings, loops_tensor):
        nodes_list = []
        for n in node["child_list"]:
            nodes_list.append(self.get_hidden_state(n, comps_embeddings, loops_tensor))
        
        if nodes_list:
            nodes_tensor = torch.cat(nodes_list, 1)
            _, (nodes_h_n, _) = self.nodes_lstm(nodes_tensor)
            nodes_h_n = nodes_h_n.permute(1, 0, 2)
        else:
            nodes_h_n = torch.unsqueeze(self.no_nodes_tensor, 0).expand(comps_embeddings.shape[0], -1, -1)
        
        if node["has_comps"]:
            selected_comps_tensor = torch.index_select(
                comps_embeddings, 1, node["computations_indices"].to(self.device)
            )
            _, (comps_h_n, _) = self.comps_lstm(selected_comps_tensor)
            comps_h_n = comps_h_n.permute(1, 0, 2)
        else:
            comps_h_n = torch.unsqueeze(self.no_comps_tensor, 0).expand(comps_embeddings.shape[0], -1, -1)
        
        selected_loop_tensor = torch.index_select(loops_tensor, 1, node["loop_index"].to(self.device))
        x = torch.cat((nodes_h_n, comps_h_n, selected_loop_tensor), 2)
        
        for i in range(len(self.concat_layers)):
            x = self.concat_layers[i](x)
            x = self.concat_dropouts[i](self.ELU(x))
        return x
    
    def forward(self, tree_tensors):
        tree, comps_tensor_first_part, comps_tensor_vectors, comps_tensor_third_part, loops_tensor, functions_comps_expr_tree = tree_tensors
        
        batch_size, num_comps, len_sequence, len_vector = functions_comps_expr_tree.shape
        x = functions_comps_expr_tree.view(batch_size * num_comps, len_sequence, len_vector)
        _, (expr_embedding, _) = self.exprs_embed(x)
        expr_embedding = expr_embedding.permute(1, 0, 2).reshape(batch_size * num_comps, -1)
        
        batch_size, num_comps, _ = comps_tensor_first_part.shape
        first_part = comps_tensor_first_part.to(self.device).view(batch_size * num_comps, -1)
        vectors = comps_tensor_vectors.to(self.device)
        third_part = comps_tensor_third_part.to(self.device).view(batch_size * num_comps, -1)
        
        vectors = self.encode_vectors(vectors)
        _, (prog_embedding, _) = self.transformation_vectors_embed(vectors)
        prog_embedding = prog_embedding.permute(1, 0, 2).reshape(batch_size * num_comps, -1)
        
        x = torch.cat((first_part, prog_embedding, third_part, expr_embedding), dim=1).view(batch_size, num_comps, -1)
        for i in range(len(self.comp_embedding_layers)):
            x = self.comp_embedding_layers[i](x)
            x = self.comp_embedding_dropouts[i](self.ELU(x))
        comps_embeddings = x
        
        roots_list = []
        for root in tree["roots"]:
            roots_list.append(self.get_hidden_state(root, comps_embeddings, loops_tensor))
        
        roots_tensor = torch.cat(roots_list, 1)
        _, (roots_h_n, _) = self.roots_lstm(roots_tensor)
        roots_h_n = roots_h_n.permute(1, 0, 2)
        
        x = roots_h_n
        for i in range(len(self.regression_layers)):
            x = self.regression_layers[i](x)
            x = self.regression_dropouts[i](self.ELU(x))
        out = self.predict(x)
        return self.LeakyReLU(out[:, 0, 0])

def train_model(model, train_loader, test_loader, X_train, X_test, criterion, optimizer, scheduler, num_epochs=200, patience=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for indices, targets in train_loader:
            tree_tensors = [X_train[i] for i in indices]
            tree_tensors = tuple(
                torch.cat([t[i].to(device) if isinstance(t[i], torch.Tensor) else t[i] for i in indices], dim=0) 
                if isinstance(t[0], torch.Tensor) else t[0] 
                for t in tree_tensors
            )
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(tree_tensors)
            loss = criterion(outputs, targets.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * indices.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for indices, targets in test_loader:
                tree_tensors = [X_test[i] for i in indices]
                tree_tensors = tuple(
                    torch.cat([t[i].to(device) if isinstance(t[i], torch.Tensor) else t[i] for i in indices], dim=0) 
                    if isinstance(t[0], torch.Tensor) else t[0] 
                    for t in tree_tensors
                )
                targets = targets.to(device)
                outputs = model(tree_tensors)
                loss = criterion(outputs, targets.squeeze())
                val_loss += loss.item() * indices.size(0)
        
        val_loss /= len(test_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f'Early stopping after {epoch+1} epochs')
            model.load_state_dict(best_model_state)
            break
    
    if best_model_state is not None and epochs_no_improve > 0:
        model.load_state_dict(best_model_state)
    
    return model

def evaluate_model(model, X_test, y_test, y_scaler, file_names_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    y_pred_scaled = []
    with torch.no_grad():
        for i in range(len(X_test)):
            tree_tensors = tuple(
                t[i].unsqueeze(0).to(device) if isinstance(t[i], torch.Tensor) else t[i] 
                for t in X_test
            )
            output = model(tree_tensors)
            y_pred_scaled.append(output.item())
    
    y_pred_scaled = np.array(y_pred_scaled).reshape(-1, 1)
    y_test = y_test.numpy()
    
    y_test_actual = y_scaler.inverse_transform(y_test)
    y_pred_actual = y_scaler.inverse_transform(y_pred_scaled)
    y_pred_actual = np.clip(y_pred_actual, 0, None)
    
    print("\nPredicted vs Actual Execution Times for Test Files:")
    for i, file_name in enumerate(file_names_test):
        print(f"File: {file_name}")
        print(f"  Actual Execution Time: {y_test_actual[i][0]:.2f} ms")
        print(f"  Predicted Execution Time: {y_pred_actual[i][0]:.2f} ms")
    
    return y_test_actual, y_pred_actual

def main(main_dir):
    print(f"Processing main directory: {main_dir}")
    train_data, test_data, test_file_names = process_main_directory(main_dir)
    
    print(f"Total training samples: {len(train_data)}")
    print(f"Total test samples: {len(test_data)}")
    
    if len(train_data) == 0 or len(test_data) == 0:
        print("Error: No valid training or test data found")
        return None, None, None
    
    X_train, y_train, X_test, y_test, y_scaler = prepare_data_for_model(train_data, test_data)
    train_loader, test_loader, X_train, X_test = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=16)
    
    model = Model_Recursive_LSTM_v2(input_size=5, device="cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    print("Training Recursive LSTM model...")
    model = train_model(model, train_loader, test_loader, X_train, X_test, criterion, optimizer, scheduler)
    
    print("\nEvaluating the model...")
    y_test_actual, y_pred_actual = evaluate_model(model, X_test, y_test, y_scaler, test_file_names)
    
    return model, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "Output_Programs"
    model, y_test_actual, y_pred_actual = main(main_dir)
    if model is not None:
        print("\nModel training and prediction completed!")
