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

# Simulated constants (adjust based on your data if known)
MAX_TAGS = 50
MAX_NUM_TRANSFORMATIONS = 10

def get_execution_time(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw_content = f.read()
            content = raw_content.decode('utf-8', errors='replace').replace('\0', '')
            data = json.loads(content)
        
        if 'programming_details' not in data:
            print(f"Error: 'programming_details' key not found in {file_path}")
            return None
        
        schedules = data["scheduling_data"]
        for item in schedules:
            if isinstance(item, dict) and item.get('name') == 'total_execution_time_ms':
                execution_time = item.get('value')
                if execution_time is not None:
                    return float(execution_time)
        
        print(f"Warning: 'total_execution_time_ms' not found in 'Schedules' of {file_path}")
        return schedules[-1]["value"]
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {file_path}: {str(e)}")
        return None
    except UnicodeDecodeError as e:
        print(f"Error: Encoding issue in {file_path}: {str(e)}")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred while processing {file_path}: {str(e)}")
        return None

def extract_features_from_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    execution_time = get_execution_time(file_path)
    if execution_time is None:
        print(f"Warning: No execution time found in {file_path}")
        return None
    
    nodes_features = []
    edges_features = []
    scheduling_features = []
    
    programming_details = data.get("programming_details", {})
    scheduling_data = data.get("scheduling_data", [])
    
    for node in programming_details.get('Nodes', []):
        node_feature = {'Name': node.get('Name', '')}
        if 'Details' in node and 'Op histogram' in node['Details']:
            op_hist = node['Details']['Op histogram']
            for op_line in op_hist:
                parts = op_line.strip().split(':')
                if len(parts) == 2:
                    op_name = parts[0].strip().lower()
                    op_count = int(parts[1].strip())
                    node_feature[f'op_{op_name}'] = op_count
        nodes_features.append(node_feature)
    
    for edge in programming_details.get('Edges', []):
        edge_feature = {
            'From': edge.get('From', ''),
            'To': edge.get('To', ''),
            'Name': edge.get('Name', '')
        }
        edges_features.append(edge_feature)
    
    for sched in scheduling_data:
        if isinstance(sched, dict) and 'Details' in sched and 'scheduling_feature' in sched['Details']:
            sched_feature = {'Name': sched.get('Name', '')}
            sched_feature.update(sched['Details']['scheduling_feature'])
            scheduling_features.append(sched_feature)
    
    features = {
        'execution_time': execution_time,
        'nodes_count': len(nodes_features),
        'edges_count': len(edges_features),
        'scheduling_count': len(scheduling_features),
        'node_edge_ratio': len(nodes_features) / len(edges_features) if edges_features else 0
    }
    
    op_counts = {}
    for node in nodes_features:
        for key, value in node.items():
            if key.startswith('op_'):
                op_counts[key] = op_counts.get(key, 0) + value
    features.update(op_counts)
    
    important_metrics = [
        'bytes_at_production', 'bytes_at_realization', 'inner_parallelism', 
        'outer_parallelism', 'num_vectors', 'points_computed_total'
    ]
    if scheduling_features:
        for metric in important_metrics:
            features[f'sched_{metric}'] = sum(sf.get(metric, 0) for sf in scheduling_features)
    
    return features

def process_directory(directory_path):
    all_features = []
    file_names = []
    
    json_files = sorted([f for f in os.listdir(directory_path) if f.endswith('.json')])
    if len(json_files) < 32:
        print(f"Warning: Expected 32 files in {directory_path}, found {len(json_files)}")
    
    for filename in json_files:
        file_path = os.path.join(directory_path, filename)
        features = extract_features_from_file(file_path)
        if features is not None:
            all_features.append(features)
            file_names.append(filename)
    
    if len(all_features) < 32:
        print(f"Warning: Only {len(all_features)} valid files found in {directory_path}")
        return None, None, None
    
    train_features = all_features[:30]
    test_features = all_features[30:32]
    test_file_names = file_names[30:32]
    
    return train_features, test_features, test_file_names

def process_main_directory(main_dir):
    all_train_features = []
    all_test_features = []
    all_test_file_names = []
    
    subdirs = sorted([d for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))])
    if not subdirs:
        raise ValueError(f"No subdirectories found in {main_dir}")
    
    for subdir in subdirs:
        subdir_path = os.path.join(main_dir, subdir)
        train_features, test_features, test_file_names = process_directory(subdir_path)
        if train_features is None:
            continue
        all_train_features.extend(train_features)
        all_test_features.extend(test_features)
        all_test_file_names.extend([os.path.join(subdir, fname) for fname in test_file_names])
        print(f"Processed {subdir}: 30 train, 2 test")
    
    return all_train_features, all_test_features, all_test_file_names

def prepare_data_for_model(train_features, test_features):
    all_features_df = pd.DataFrame(train_features + test_features)
    all_features_df = all_features_df.fillna(0)
    
    constant_columns = [col for col in all_features_df.columns if all_features_df[col].nunique() == 1 and col != 'execution_time']
    all_features_df = all_features_df.drop(columns=constant_columns)
    
    all_features_df['execution_time_log'] = np.log1p(all_features_df['execution_time'])
    numeric_cols = all_features_df.select_dtypes(include=['number']).columns
    all_features_df = all_features_df[numeric_cols]
    
    train_df = all_features_df.iloc[:len(train_features)]
    test_df = all_features_df.iloc[len(train_features):]
    
    y_train = train_df['execution_time_log'].values.reshape(-1, 1)
    y_test = test_df['execution_time_log'].values.reshape(-1, 1)
    X_train_df = train_df.drop(['execution_time', 'execution_time_log'], axis=1)
    X_test_df = test_df.drop(['execution_time', 'execution_time_log'], axis=1)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_df)
    X_test_scaled = scaler_X.transform(X_test_df)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    
    input_size = X_train_scaled.shape[1]
    batch_size_train = X_train_scaled.shape[0]
    batch_size_test = X_test_scaled.shape[0]
    
    # Define tree structure
    tree = {"roots": [{"child_list": [], "has_comps": True, "computations_indices": torch.arange(input_size), "loop_index": torch.tensor([0])}]}
    
    # Prepare tensors
    lstm_embedding_size = 32
    expr_embed_size = 11
    num_layers = 1
    bidirectional = True
    expected_input_size = input_size + lstm_embedding_size * (2 if bidirectional else 1) * num_layers + expr_embed_size
    
    # Adjust comps_tensor_third_part to match expected size
    comps_tensor_first_part = torch.FloatTensor(X_train_scaled).unsqueeze(1)  # [batch_size, 1, input_size]
    comps_tensor_vectors = torch.zeros(batch_size_train, MAX_NUM_TRANSFORMATIONS, MAX_TAGS)
    comps_tensor_third_part = torch.zeros(batch_size_train, 1, expected_input_size - input_size - lstm_embedding_size * 2)  # Adjust size
    loops_tensor = torch.zeros(batch_size_train, 1, 8)
    functions_comps_expr_tree = torch.zeros(batch_size_train, 1, 1, 11)
    
    X_train_tensor_tuple = (comps_tensor_first_part, comps_tensor_vectors, comps_tensor_third_part, loops_tensor, functions_comps_expr_tree)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    
    comps_tensor_first_part_test = torch.FloatTensor(X_test_scaled).unsqueeze(1)
    comps_tensor_vectors_test = torch.zeros(batch_size_test, MAX_NUM_TRANSFORMATIONS, MAX_TAGS)
    comps_tensor_third_part_test = torch.zeros(batch_size_test, 1, expected_input_size - input_size - lstm_embedding_size * 2)
    loops_tensor_test = torch.zeros(batch_size_test, 1, 8)
    functions_comps_expr_tree_test = torch.zeros(batch_size_test, 1, 1, 11)
    
    X_test_tensor_tuple = (comps_tensor_first_part_test, comps_tensor_vectors_test, comps_tensor_third_part_test, loops_tensor_test, functions_comps_expr_tree_test)
    y_test_tensor = torch.FloatTensor(y_test_scaled)
    
    return tree, X_train_tensor_tuple, y_train_tensor, X_test_tensor_tuple, y_test_tensor, scaler_y, input_size

def initialization_function_xavier(x):
    return nn.init.xavier_uniform_(x)

class Model_Recursive_LSTM_v2(nn.Module):
    def __init__(
        self,
        input_size,
        comp_embed_layer_sizes=[128, 64, 32, 16],
        drops=[0.225, 0.225, 0.225, 0.225],
        output_size=1,
        lstm_embedding_size=32,
        expr_embed_size=11,
        loops_tensor_size=8,
        device="cpu",
        num_layers=1,
        bidirectional=True,
    ):
        super().__init__()
        self.device = device
        embedding_size = comp_embed_layer_sizes[-1]
        
        regression_layer_sizes = [embedding_size] + comp_embed_layer_sizes[-2:]
        concat_layer_sizes = [
            embedding_size * 2 + loops_tensor_size
        ] + comp_embed_layer_sizes[-2:]
        
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
            initialization_function_xavier(self.comp_embedding_layers[i].weight)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))
        for i in range(len(regression_layer_sizes) - 1):
            self.regression_layers.append(
                nn.Linear(regression_layer_sizes[i], regression_layer_sizes[i + 1], bias=True)
            )
            initialization_function_xavier(self.regression_layers[i].weight)
            self.regression_dropouts.append(nn.Dropout(drops[i]))
        for i in range(len(concat_layer_sizes) - 1):
            self.concat_layers.append(
                nn.Linear(concat_layer_sizes[i], concat_layer_sizes[i + 1], bias=True)
            )
            initialization_function_xavier(self.concat_layers[i].weight)
            nn.init.zeros_(self.concat_layers[i].weight)
            self.concat_dropouts.append(nn.Dropout(drops[i]))
        
        self.predict = nn.Linear(regression_layer_sizes[-1], output_size, bias=True)
        initialization_function_xavier(self.predict.weight)
        self.ELU = nn.ELU()
        self.LeakyReLU = nn.LeakyReLU(0.01)
        
        self.no_comps_tensor = nn.Parameter(torch.zeros(1, embedding_size))
        self.no_nodes_tensor = nn.Parameter(torch.zeros(1, embedding_size))
        initialization_function_xavier(self.no_comps_tensor)
        initialization_function_xavier(self.no_nodes_tensor)
        
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
            selected_comps_tensor = torch.index_select(comps_embeddings, 1, node["computations_indices"].to(self.device))
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

def create_data_loaders(tree, X_train_tensor_tuple, y_train, X_test_tensor_tuple, y_test, batch_size=16):
    train_dataset = TensorDataset(*X_train_tensor_tuple, y_train)
    test_dataset = TensorDataset(*X_test_tensor_tuple, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return tree, train_loader, test_loader

def train_model(model, tree, train_loader, test_loader, criterion, optimizer, num_epochs=150, patience=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            tensor_batch = tuple(t.to(device) for t in batch[:-1])
            targets = batch[-1].to(device)
            tree_tensors = (tree, *tensor_batch)
            optimizer.zero_grad()
            outputs = model(tree_tensors)
            loss = criterion(outputs, targets.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * targets.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                tensor_batch = tuple(t.to(device) for t in batch[:-1])
                targets = batch[-1].to(device)
                tree_tensors = (tree, *tensor_batch)
                outputs = model(tree_tensors)
                loss = criterion(outputs, targets.squeeze())
                val_loss += loss.item() * targets.size(0)
        
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

def evaluate_model(model, tree, X_test_tensor_tuple, y_test, y_scaler, file_names_test, is_log_transformed=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        tensor_tuple = tuple(t.to(device) for t in X_test_tensor_tuple)
        tree_tensors = (tree, *tensor_tuple)
        y_pred_scaled = model(tree_tensors)
    
    y_pred_scaled = y_pred_scaled.cpu().numpy()
    y_test = y_test.cpu().numpy()
    
    y_test_transformed = y_scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_transformed = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    
    if is_log_transformed:
        y_test_actual = np.expm1(y_test_transformed)
        y_pred_actual = np.expm1(y_pred_transformed)
    else:
        y_test_actual = y_test_transformed
        y_pred_actual = y_pred_transformed
    
    print("\nPredicted vs Actual Execution Times:")
    for i, file_name in enumerate(file_names_test):
        print(f"File: {file_name}")
        print(f"  Actual: {y_test_actual[i][0]:.2f} ms")
        print(f"  Predicted: {y_pred_actual[i][0]:.2f} ms")
        print(f"  Error: {abs(y_test_actual[i][0] - y_pred_actual[i][0]) / y_test_actual[i][0] * 100:.2f}%")
    
    mse = np.mean((y_test_actual - y_pred_actual) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_actual - y_pred_actual))
    print(f"\nMSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
    
    return y_test_actual, y_pred_actual

def main(main_dir):
    print(f"Processing main directory: {main_dir}")
    train_features, test_features, test_file_names = process_main_directory(main_dir)
    
    print(f"Total training samples: {len(train_features)}")
    print(f"Total test samples: {len(test_features)}")
    
    if not train_features or not test_features:
        print("Error: No valid data found")
        return None
    
    tree, X_train_tensor_tuple, y_train, X_test_tensor_tuple, y_test, y_scaler, input_size = prepare_data_for_model(train_features, test_features)
    
    tree, train_loader, test_loader = create_data_loaders(tree, X_train_tensor_tuple, y_train, X_test_tensor_tuple, y_test, batch_size=16)
    
    model = Model_Recursive_LSTM_v2(
        input_size=input_size,
        comp_embed_layer_sizes=[128, 64, 32, 16],
        drops=[0.225, 0.225, 0.225, 0.225],
        output_size=1,
        lstm_embedding_size=32,
        expr_embed_size=11,
        loops_tensor_size=8,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    print("Training Model_Recursive_LSTM_v2...")
    model = train_model(model, tree, train_loader, test_loader, criterion, optimizer)
    
    print("\nEvaluating model:")
    y_test_actual, y_pred_actual = evaluate_model(model, tree, X_test_tensor_tuple, y_test, y_scaler, test_file_names)
    
    return model, y_test_actual, y_pred_actual

if __name__ == "__main__":
    main_dir = "Output_Programs"
    model, y_test_actual, y_pred_actual = main(main_dir)
    if model is not None:
        print("\nTraining and prediction completed!")
