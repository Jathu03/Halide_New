import os
import jsonimport os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings

# Define the Model_Recursive_LSTM_v2 (unchanged)
class Model_Recursive_LSTM_v2(nn.Module):
    def __init__(
        self,
        input_size,
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
        
        self.encode_vectors = nn.Linear(3, 3, bias=True)
        for i in range(len(comp_embed_layer_sizes) - 1):
            self.comp_embedding_layers.append(nn.Linear(comp_embed_layer_sizes[i], comp_embed_layer_sizes[i + 1], bias=True))
            nn.init.xavier_uniform_(self.comp_embedding_layers[i].weight)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))
        for i in range(len(regression_layer_sizes) - 1):
            self.regression_layers.append(nn.Linear(regression_layer_sizes[i], regression_layer_sizes[i + 1], bias=True))
            nn.init.xavier_uniform_(self.regression_layers[i].weight)
            self.regression_dropouts.append(nn.Dropout(drops[i]))
        for i in range(len(concat_layer_sizes) - 1):
            self.concat_layers.append(nn.Linear(concat_layer_sizes[i], concat_layer_sizes[i + 1], bias=True))
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
        self.transformation_vectors_embed = nn.LSTM(3, lstm_embedding_size, batch_first=True, bidirectional=bidirectional, num_layers=num_layers)
        self.exprs_embed = nn.LSTM(11, expr_embed_size, batch_first=True)

    def get_hidden_state(self, node, comps_embeddings, loops_tensor):
        nodes_list = []
        for n in node["child_list"]:
            nodes_list.append(self.get_hidden_state(n, comps_embeddings, loops_tensor))
        
        if nodes_list:
            nodes_tensor = torch.cat(nodes_list, 1)
            lstm_out, (nodes_h_n, nodes_c_n) = self.nodes_lstm(nodes_tensor)
            nodes_h_n = nodes_h_n.permute(1, 0, 2)
        else:
            nodes_h_n = torch.unsqueeze(self.no_nodes_tensor, 0).expand(comps_embeddings.shape[0], -1, -1)
        
        if node["has_comps"]:
            selected_comps_tensor = torch.index_select(comps_embeddings, 1, node["computations_indices"].to(self.device))
            lstm_out, (comps_h_n, comps_c_n) = self.comps_lstm(selected_comps_tensor)
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
        lstm_out, (roots_h_n, roots_c_n) = self.roots_lstm(roots_tensor)
        roots_h_n = roots_h_n.permute(1, 0, 2)
        
        x = roots_h_n
        for i in range(len(self.regression_layers)):
            x = self.regression_layers[i](x)
            x = self.regression_dropouts[i](self.ELU(x))
        out = self.predict(x)
        return self.LeakyReLU(out[:, 0, 0])

# Custom Dataset Class
class HalideDataset(Dataset):
    def __init__(self, data_dir, scaler=None):
        self.data_dir = data_dir
        self.scaler = scaler or StandardScaler()
        self.samples = []
        self._load_data()

    def _load_data(self):
        for program_folder in os.listdir(self.data_dir):
            program_path = os.path.join(self.data_dir, program_folder)
            if not os.path.isdir(program_path):
                continue
            for schedule_file in os.listdir(program_path):
                if not schedule_file.endswith(".json"):
                    continue
                with open(os.path.join(program_path, schedule_file), "r") as f:
                    data = json.load(f)
                    tree_tensors, execution_time = self._process_schedule(data)
                    self.samples.append((tree_tensors, execution_time))

    def _process_schedule(self, data):
        # Extract nodes and edges
        nodes = data["programming_details"]["Nodes"]
        edges = data["programming_details"]["Edges"]
        scheduling_data = {item["Name"]: item["Details"]["scheduling_feature"] for item in data["scheduling_data"] if "Name" in item}
        execution_time = next(item["value"] for item in data["scheduling_data"] if item.get("name") == "total_execution_time_ms")

        # Build node map (only base function names)
        node_map = {node["Name"]: i for i, node in enumerate(nodes)}

        # Helper function to strip update suffixes
        def clean_name(name):
            return name.split(".update")[0]

        # Helper function to parse fractions or numbers, handling invalid values
        def parse_value(val):
            val = val.strip()
            if '/' in val:
                try:
                    num, denom = map(float, val.split('/'))
                    return num / denom
                except (ValueError, ZeroDivisionError):
                    warnings.warn(f"Invalid fraction '{val}' in Load Jacobians, defaulting to 0.0")
                    return 0.0
            try:
                return float(val)
            except ValueError:
                warnings.warn(f"Invalid value '{val}' in Load Jacobians, defaulting to 0.0")
                return 0.0

        # Filter edges and determine dependencies
        child_map = {i: [] for i in range(len(nodes))}
        valid_edges = []
        to_nodes = set()
        for edge in edges:
            from_name = clean_name(edge["From"])
            to_name = clean_name(edge["To"])
            if from_name in node_map and to_name in node_map:
                from_idx = node_map[from_name]
                to_idx = node_map[to_name]
                child_map[from_idx].append(to_idx)
                valid_edges.append(edge)
                to_nodes.add(to_idx)

        # Dynamically find the root (node with no outgoing edges)
        all_indices = set(range(len(nodes)))
        root_candidates = all_indices - to_nodes
        if not root_candidates:
            raise ValueError("No root node found in the schedule (no node without outgoing edges)")
        root_idx = max(root_candidates)

        def build_node(idx):
            return {
                "name": nodes[idx]["Name"],
                "has_comps": True,
                "computations_indices": torch.tensor([idx]),
                "loop_index": torch.tensor([idx]),
                "child_list": [build_node(child_idx) for child_idx in child_map[idx]]
            }
        tree = {"roots": [build_node(root_idx)]}

        # Computation tensors
        num_comps = len(nodes)
        comps_tensor_first_part = torch.zeros(num_comps, 26)  # 26 scheduling features
        comps_tensor_vectors = torch.zeros(num_comps, 3)     # Load Jacobians (3D)
        comps_tensor_third_part = torch.zeros(num_comps, 24) # 24 op histogram features
        
        for i, node in enumerate(nodes):
            name = node["Name"]
            if name in scheduling_data:
                features = list(scheduling_data[name].values())
                comps_tensor_first_part[i] = torch.tensor(features[:26])  # Truncate/pad to 26
            for edge in valid_edges:
                if clean_name(edge["To"]) == name:
                    jacobians = edge["Details"]["Load Jacobians"]
                    if jacobians and isinstance(jacobians, list) and len(jacobians) > 0:
                        values = [parse_value(x) for x in jacobians[0].split()]
                        # Ensure we have at least 3 values, padding with 0 if needed
                        values = values + [0.0] * (3 - len(values)) if len(values) < 3 else values[:3]
                        comps_tensor_vectors[i] = torch.tensor(values)
                    break
            op_hist = [int(x.split()[-1]) for x in node["Details"]["Op histogram"]]
            comps_tensor_third_part[i] = torch.tensor(op_hist[:24])  # Ensure exactly 24 elements

        # Loops tensor
        loops_tensor = torch.zeros(num_comps, 8)
        for i, node in enumerate(nodes):
            name = node["Name"]
            if name in scheduling_data:
                loops = [
                    scheduling_data[name]["innermost_loop_extent"],
                    scheduling_data[name]["vector_size"],
                    scheduling_data[name]["unrolled_loop_extent"],
                    scheduling_data[name]["inner_parallelism"],
                    scheduling_data[name]["outer_parallelism"],
                    0, 0, 0  # Padding
                ]
                loops_tensor[i] = torch.tensor(loops)

        # Expression tensor (simplified)
        functions_comps_expr_tree = torch.zeros(num_comps, 10, 11)  # 10 ops, 11 features
        for i, node in enumerate(nodes):
            op_hist = [int(x.split()[-1]) for x in node["Details"]["Op histogram"][:10]]  # First 10 ops
            functions_comps_expr_tree[i, :, :10] = torch.tensor(op_hist).float().view(10, 1).expand(10, 10)

        # Normalize features
        if not hasattr(self.scaler, "mean_"):
            self.scaler.fit(comps_tensor_first_part.numpy())
        comps_tensor_first_part = torch.tensor(self.scaler.transform(comps_tensor_first_part.numpy()))

        tree_tensors = (
            tree,
            comps_tensor_first_part.unsqueeze(0),
            comps_tensor_vectors.unsqueeze(0),
            comps_tensor_third_part.unsqueeze(0),
            loops_tensor.unsqueeze(0),
            functions_comps_expr_tree.unsqueeze(0)
        )
        return tree_tensors, execution_time

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Training Function (unchanged)
def train_model(model, train_loader, val_loader, num_epochs=50, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for tree_tensors, execution_time in train_loader:
            execution_time = execution_time.to(device).float()
            optimizer.zero_grad()
            output = model(tree_tensors)
            loss = criterion(output, execution_time)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for tree_tensors, execution_time in val_loader:
                execution_time = execution_time.to(device).float()
                output = model(tree_tensors)
                val_loss += criterion(output, execution_time).item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
    
    return model

# Evaluation and Speedup Prediction (unchanged)
def evaluate_model(model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for tree_tensors, execution_time in test_loader:
            execution_time = execution_time.to(device).float()
            pred = model(tree_tensors)
            predictions.append(pred.item())
            actuals.append(execution_time.item())
    
    baseline = np.mean(actuals)
    speedups_pred = [baseline / pred for pred in predictions]
    speedups_actual = [baseline / actual for actual in actuals]
    errors = [abs(pred - actual) / actual * 100 for pred, actual in zip(speedups_pred, speedups_actual)]
    return predictions, actuals, speedups_pred, speedups_actual, errors

# Main Execution (unchanged)
def main():
    data_dir = "synthetic_data"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = HalideDataset(data_dir)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    input_size = 26 + 3 + 24 + 100
    model = Model_Recursive_LSTM_v2(input_size=input_size, device=device)
    
    model = train_model(model, train_loader, val_loader)
    
    test_subset, _ = torch.utils.data.random_split(test_dataset, [10, len(test_dataset) - 10])
    test_subset_loader = DataLoader(test_subset, batch_size=1, shuffle=False)
    
    predictions, actuals, speedups_pred, speedups_actual, errors = evaluate_model(model, test_subset_loader)
    
    print("\nEvaluation Results for 10 Test Schedules:")
    for i in range(10):
        print(f"Schedule {i+1}:")
        print(f"  Predicted Execution Time: {predictions[i]:.2f} ms")
        print(f"  Actual Execution Time: {actuals[i]:.2f} ms")
        print(f"  Predicted Speedup: {speedups_pred[i]:.4f}")
        print(f"  Actual Speedup: {speedups_actual[i]:.4f}")
        print(f"  Error Percentage: {errors[i]:.2f}%")
    
    avg_error = np.mean(errors)
    print(f"\nAverage Error Percentage: {avg_error:.2f}%")

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler

# Define the Model_Recursive_LSTM_v2 (unchanged)
class Model_Recursive_LSTM_v2(nn.Module):
    def __init__(
        self,
        input_size,
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
        
        self.encode_vectors = nn.Linear(3, 3, bias=True)
        for i in range(len(comp_embed_layer_sizes) - 1):
            self.comp_embedding_layers.append(nn.Linear(comp_embed_layer_sizes[i], comp_embed_layer_sizes[i + 1], bias=True))
            nn.init.xavier_uniform_(self.comp_embedding_layers[i].weight)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))
        for i in range(len(regression_layer_sizes) - 1):
            self.regression_layers.append(nn.Linear(regression_layer_sizes[i], regression_layer_sizes[i + 1], bias=True))
            nn.init.xavier_uniform_(self.regression_layers[i].weight)
            self.regression_dropouts.append(nn.Dropout(drops[i]))
        for i in range(len(concat_layer_sizes) - 1):
            self.concat_layers.append(nn.Linear(concat_layer_sizes[i], concat_layer_sizes[i + 1], bias=True))
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
        self.transformation_vectors_embed = nn.LSTM(3, lstm_embedding_size, batch_first=True, bidirectional=bidirectional, num_layers=num_layers)
        self.exprs_embed = nn.LSTM(11, expr_embed_size, batch_first=True)

    def get_hidden_state(self, node, comps_embeddings, loops_tensor):
        nodes_list = []
        for n in node["child_list"]:
            nodes_list.append(self.get_hidden_state(n, comps_embeddings, loops_tensor))
        
        if nodes_list:
            nodes_tensor = torch.cat(nodes_list, 1)
            lstm_out, (nodes_h_n, nodes_c_n) = self.nodes_lstm(nodes_tensor)
            nodes_h_n = nodes_h_n.permute(1, 0, 2)
        else:
            nodes_h_n = torch.unsqueeze(self.no_nodes_tensor, 0).expand(comps_embeddings.shape[0], -1, -1)
        
        if node["has_comps"]:
            selected_comps_tensor = torch.index_select(comps_embeddings, 1, node["computations_indices"].to(self.device))
            lstm_out, (comps_h_n, comps_c_n) = self.comps_lstm(selected_comps_tensor)
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
        lstm_out, (roots_h_n, roots_c_n) = self.roots_lstm(roots_tensor)
        roots_h_n = roots_h_n.permute(1, 0, 2)
        
        x = roots_h_n
        for i in range(len(self.regression_layers)):
            x = self.regression_layers[i](x)
            x = self.regression_dropouts[i](self.ELU(x))
        out = self.predict(x)
        return self.LeakyReLU(out[:, 0, 0])

# Custom Dataset Class
class HalideDataset(Dataset):
    def __init__(self, data_dir, scaler=None):
        self.data_dir = data_dir
        self.scaler = scaler or StandardScaler()
        self.samples = []
        self._load_data()

    def _load_data(self):
        for program_folder in os.listdir(self.data_dir):
            program_path = os.path.join(self.data_dir, program_folder)
            if not os.path.isdir(program_path):
                continue
            for schedule_file in os.listdir(program_path):
                if not schedule_file.endswith(".json"):
                    continue
                with open(os.path.join(program_path, schedule_file), "r") as f:
                    data = json.load(f)
                    tree_tensors, execution_time = self._process_schedule(data)
                    self.samples.append((tree_tensors, execution_time))

    def _process_schedule(self, data):
        # Extract nodes and edges
        nodes = data["programming_details"]["Nodes"]
        edges = data["programming_details"]["Edges"]
        scheduling_data = {item["Name"]: item["Details"]["scheduling_feature"] for item in data["scheduling_data"] if "Name" in item}
        execution_time = next(item["value"] for item in data["scheduling_data"] if item.get("name") == "total_execution_time_ms")

        # Build node map (only base function names)
        node_map = {node["Name"]: i for i, node in enumerate(nodes)}

        # Helper function to strip update suffixes
        def clean_name(name):
            return name.split(".update")[0]

        # Helper function to parse fractions or numbers
        def parse_value(val):
            val = val.strip()
            if '/' in val:
                num, denom = map(float, val.split('/'))
                return num / denom
            return float(val)

        # Filter edges and determine dependencies
        child_map = {i: [] for i in range(len(nodes))}
        valid_edges = []
        to_nodes = set()
        for edge in edges:
            from_name = clean_name(edge["From"])
            to_name = clean_name(edge["To"])
            if from_name in node_map and to_name in node_map:
                from_idx = node_map[from_name]
                to_idx = node_map[to_name]
                child_map[from_idx].append(to_idx)
                valid_edges.append(edge)
                to_nodes.add(to_idx)

        # Dynamically find the root (node with no outgoing edges)
        all_indices = set(range(len(nodes)))
        root_candidates = all_indices - to_nodes
        if not root_candidates:
            raise ValueError("No root node found in the schedule (no node without outgoing edges)")
        root_idx = max(root_candidates)

        def build_node(idx):
            return {
                "name": nodes[idx]["Name"],
                "has_comps": True,
                "computations_indices": torch.tensor([idx]),
                "loop_index": torch.tensor([idx]),
                "child_list": [build_node(child_idx) for child_idx in child_map[idx]]
            }
        tree = {"roots": [build_node(root_idx)]}

        # Computation tensors
        num_comps = len(nodes)
        comps_tensor_first_part = torch.zeros(num_comps, 26)  # 26 scheduling features
        comps_tensor_vectors = torch.zeros(num_comps, 3)     # Load Jacobians (3D)
        comps_tensor_third_part = torch.zeros(num_comps, 24) # 24 op histogram features
        
        for i, node in enumerate(nodes):
            name = node["Name"]
            if name in scheduling_data:
                features = list(scheduling_data[name].values())
                comps_tensor_first_part[i] = torch.tensor(features[:26])  # Truncate/pad to 26
            for edge in valid_edges:
                if clean_name(edge["To"]) == name:
                    jacobians = edge["Details"]["Load Jacobians"]
                    if jacobians and isinstance(jacobians, list) and len(jacobians) > 0:
                        values = [parse_value(x) for x in jacobians[0].split()]
                        comps_tensor_vectors[i] = torch.tensor(values[:3])  # Take first 3 values
                    break
            op_hist = [int(x.split()[-1]) for x in node["Details"]["Op histogram"]]
            comps_tensor_third_part[i] = torch.tensor(op_hist[:24])  # Ensure exactly 24 elements

        # Loops tensor
        loops_tensor = torch.zeros(num_comps, 8)
        for i, node in enumerate(nodes):
            name = node["Name"]
            if name in scheduling_data:
                loops = [
                    scheduling_data[name]["innermost_loop_extent"],
                    scheduling_data[name]["vector_size"],
                    scheduling_data[name]["unrolled_loop_extent"],
                    scheduling_data[name]["inner_parallelism"],
                    scheduling_data[name]["outer_parallelism"],
                    0, 0, 0  # Padding
                ]
                loops_tensor[i] = torch.tensor(loops)

        # Expression tensor (simplified)
        functions_comps_expr_tree = torch.zeros(num_comps, 10, 11)  # 10 ops, 11 features
        for i, node in enumerate(nodes):
            op_hist = [int(x.split()[-1]) for x in node["Details"]["Op histogram"][:10]]  # First 10 ops
            functions_comps_expr_tree[i, :, :10] = torch.tensor(op_hist).float().view(10, 1).expand(10, 10)

        # Normalize features
        if not hasattr(self.scaler, "mean_"):
            self.scaler.fit(comps_tensor_first_part.numpy())
        comps_tensor_first_part = torch.tensor(self.scaler.transform(comps_tensor_first_part.numpy()))

        tree_tensors = (
            tree,
            comps_tensor_first_part.unsqueeze(0),
            comps_tensor_vectors.unsqueeze(0),
            comps_tensor_third_part.unsqueeze(0),
            loops_tensor.unsqueeze(0),
            functions_comps_expr_tree.unsqueeze(0)
        )
        return tree_tensors, execution_time

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# Training Function (unchanged)
def train_model(model, train_loader, val_loader, num_epochs=50, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for tree_tensors, execution_time in train_loader:
            execution_time = execution_time.to(device).float()
            optimizer.zero_grad()
            output = model(tree_tensors)
            loss = criterion(output, execution_time)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for tree_tensors, execution_time in val_loader:
                execution_time = execution_time.to(device).float()
                output = model(tree_tensors)
                val_loss += criterion(output, execution_time).item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
    
    return model

# Evaluation and Speedup Prediction (unchanged)
def evaluate_model(model, test_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for tree_tensors, execution_time in test_loader:
            execution_time = execution_time.to(device).float()
            pred = model(tree_tensors)
            predictions.append(pred.item())
            actuals.append(execution_time.item())
    
    baseline = np.mean(actuals)
    speedups_pred = [baseline / pred for pred in predictions]
    speedups_actual = [baseline / actual for actual in actuals]
    errors = [abs(pred - actual) / actual * 100 for pred, actual in zip(speedups_pred, speedups_actual)]
    return predictions, actuals, speedups_pred, speedups_actual, errors

# Main Execution (unchanged)
def main():
    data_dir = "synthetic_data"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = HalideDataset(data_dir)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    input_size = 26 + 3 + 24 + 100
    model = Model_Recursive_LSTM_v2(input_size=input_size, device=device)
    
    model = train_model(model, train_loader, val_loader)
    
    test_subset, _ = torch.utils.data.random_split(test_dataset, [10, len(test_dataset) - 10])
    test_subset_loader = DataLoader(test_subset, batch_size=1, shuffle=False)
    
    predictions, actuals, speedups_pred, speedups_actual, errors = evaluate_model(model, test_subset_loader)
    
    print("\nEvaluation Results for 10 Test Schedules:")
    for i in range(10):
        print(f"Schedule {i+1}:")
        print(f"  Predicted Execution Time: {predictions[i]:.2f} ms")
        print(f"  Actual Execution Time: {actuals[i]:.2f} ms")
        print(f"  Predicted Speedup: {speedups_pred[i]:.4f}")
        print(f"  Actual Speedup: {speedups_actual[i]:.4f}")
        print(f"  Error Percentage: {errors[i]:.2f}%")
    
    avg_error = np.mean(errors)
    print(f"\nAverage Error Percentage: {avg_error:.2f}%")

if __name__ == "__main__":
    main()
