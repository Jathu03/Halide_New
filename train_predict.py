import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os

# Define constants
MAX_NUM_TRANSFORMATIONS = 4
MAX_TAGS = 16
MAX_COMPS = 10
NUM_SCHEDULES = 5  # Number of schedules to predict

class Model_Recursive_LSTM_v2(nn.Module):
    def __init__(
        self,
        input_size,
        comp_embed_layer_sizes=[600, 350, 200, 180],
        drops=[0.225, 0.225, 0.225, 0.225],
        output_size=NUM_SCHEDULES,
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
        
        total_embedding_size = lstm_embedding_size * (2 if bidirectional else 1) * num_layers + expr_embed_size
        comp_embed_layer_sizes = [input_size + total_embedding_size] + comp_embed_layer_sizes
        
        regression_layer_sizes = [embedding_size] + comp_embed_layer_sizes[-2:]
        concat_layer_sizes = [embedding_size * 2 + loops_tensor_size] + comp_embed_layer_sizes[-2:]
        
        self.comp_embedding_layers = nn.ModuleList()
        self.comp_embedding_dropouts = nn.ModuleList()
        self.regression_layers = nn.ModuleList()
        self.regression_dropouts = nn.ModuleList()
        self.concat_layers = nn.ModuleList()
        self.concat_dropouts = nn.ModuleList()
        
        self.encode_vectors = nn.Linear(MAX_TAGS, MAX_TAGS, bias=True)
        
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
        self.transformation_vectors_embed = nn.LSTM(MAX_TAGS, lstm_embedding_size, batch_first=True, bidirectional=bidirectional, num_layers=num_layers)
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
        trees, comps_tensor_first_part, comps_tensor_vectors, comps_tensor_third_part, loops_tensor, functions_comps_expr_tree = tree_tensors
        
        batch_size, num_comps, len_sequence, len_vector = functions_comps_expr_tree.shape
        x = functions_comps_expr_tree.view(batch_size * num_comps, len_sequence, len_vector)
        _, (expr_embedding, _) = self.exprs_embed(x)
        expr_embedding = expr_embedding.permute(1, 0, 2).reshape(batch_size * num_comps, -1)
        
        batch_size, num_comps, _ = comps_tensor_first_part.shape
        first_part = comps_tensor_first_part.to(self.device).view(batch_size * num_comps, -1)
        vectors = comps_tensor_vectors.to(self.device)
        third_part = comps_tensor_third_part.to(self.device).view(batch_size * num_comps, -1)
        
        vectors = vectors.view(batch_size * num_comps, MAX_NUM_TRANSFORMATIONS, MAX_TAGS)
        vectors = self.encode_vectors(vectors)
        _, (prog_embedding, _) = self.transformation_vectors_embed(vectors)
        prog_embedding = prog_embedding.permute(1, 0, 2).reshape(batch_size * num_comps, -1)
        
        x = torch.cat((first_part, prog_embedding, third_part, expr_embedding), dim=1).view(batch_size, num_comps, -1)
        
        for i in range(len(self.comp_embedding_layers)):
            x = self.comp_embedding_layers[i](x)
            x = self.comp_embedding_dropouts[i](self.ELU(x))
        comps_embeddings = x
        
        roots_list = []
        for batch_idx in range(batch_size):
            tree = trees[batch_idx]
            for root in tree["roots"]:
                roots_list.append(self.get_hidden_state(root, comps_embeddings[batch_idx:batch_idx+1], loops_tensor[batch_idx:batch_idx+1]))
        
        roots_tensor = torch.cat(roots_list, 1)
        _, (roots_h_n, _) = self.roots_lstm(roots_tensor)
        roots_h_n = roots_h_n.permute(1, 0, 2)
        
        x = roots_h_n
        for i in range(len(self.regression_layers)):
            x = self.regression_layers[i](x)
            x = self.regression_dropouts[i](self.ELU(x))
        out = self.predict(x)
        return out[:, 0, :]  # [batch_size, 5]

class TiramisuDataset(Dataset):
    def __init__(self, dataset_path="tiramisu_dataset.pt"):
        self.data = torch.load(dataset_path)
        # Use single exec_time and generate synthetic times for 5 schedules
        base_exec_times = torch.tensor([sample['exec_time'] for sample in self.data], dtype=torch.float32)
        # Simulate 5 schedules with scaling factors (e.g., 0.8, 0.9, 1.0, 1.1, 1.2)
        scaling_factors = torch.tensor([0.8, 0.9, 1.0, 1.1, 1.2], dtype=torch.float32)
        exec_times = base_exec_times.unsqueeze(1) * scaling_factors  # [num_samples, 5]
        self.exec_time_mean = exec_times.mean(dim=0)  # Mean for each schedule
        self.exec_time_std = exec_times.std(dim=0)    # Std for each schedule
        self.scaling_factors = scaling_factors
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        num_comps = sample['comps_tensor'].shape[0]
        
        comps_tensor = sample['comps_tensor']
        if num_comps < MAX_COMPS:
            padding = torch.zeros(MAX_COMPS - num_comps, comps_tensor.shape[1])
            comps_tensor = torch.cat([comps_tensor, padding], dim=0)
        elif num_comps > MAX_COMPS:
            comps_tensor = comps_tensor[:MAX_COMPS]
        
        expr_tensor = sample['expr_tensor']
        num_expr_comps = expr_tensor.shape[0]
        if num_expr_comps < MAX_COMPS:
            padding = torch.zeros(MAX_COMPS - num_expr_comps, expr_tensor.shape[1], expr_tensor.shape[2])
            expr_tensor = torch.cat([expr_tensor, padding], dim=0)
        elif num_expr_comps > MAX_COMPS:
            expr_tensor = expr_tensor[:MAX_COMPS]
        
        tree = {
            "roots": [{
                "child_list": [],
                "has_comps": True,
                "computations_indices": torch.tensor([i for i in range(min(num_comps, MAX_COMPS))], dtype=torch.long),
                "loop_index": torch.tensor([0], dtype=torch.long)
            }]
        }
        # Generate and normalize execution times for 5 schedules
        base_exec_time = sample['exec_time']
        exec_times = base_exec_time * self.scaling_factors
        normalized_exec_times = (exec_times - self.exec_time_mean) / self.exec_time_std
        
        tree_tensors = (
            tree,
            comps_tensor[:, :10],
            comps_tensor[:, 10:74],
            comps_tensor[:, 74:],
            sample['loops_tensor'],
            expr_tensor
        )
        return tree_tensors, normalized_exec_times

def custom_collate_fn(batch):
    tree_tensors_list = []
    targets_list = []
    
    for tree_tensors, target in batch:
        tree_tensors_list.append(tree_tensors)
        targets_list.append(target)
    
    comps_first = torch.stack([t[1] for t in tree_tensors_list])
    comps_vectors = torch.stack([t[2] for t in tree_tensors_list])
    comps_third = torch.stack([t[3] for t in tree_tensors_list])
    
    max_loops = max(t[4].shape[0] for t in tree_tensors_list)
    loops_tensor = torch.stack([
        torch.cat([t[4], torch.zeros(max_loops - t[4].shape[0], t[4].shape[1])], dim=0) if t[4].shape[0] < max_loops else t[4][:max_loops]
        for t in tree_tensors_list
    ])
    
    expr_tensor = torch.stack([t[5] for t in tree_tensors_list])
    
    targets = torch.stack(targets_list)  # [batch_size, 5]
    trees = [t[0] for t in tree_tensors_list]
    
    return (trees, comps_first, comps_vectors, comps_third, loops_tensor, expr_tensor), targets

def train_model(model, train_loader, val_loader, num_epochs=10, device="cuda" if torch.cuda.is_available() else "cpu", learning_rate=0.0001):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_path = "best_model.pt"
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_count = 0
        
        for batch_idx, (tree_tensors, targets) in enumerate(tqdm(train_loader)):
            trees, comps_first, comps_vectors, comps_third, loops_tensor, expr_tensor = tree_tensors
            tree_tensors_device = (
                trees,
                comps_first.to(device),
                comps_vectors.to(device),
                comps_third.to(device),
                loops_tensor.to(device),
                expr_tensor.to(device)
            )
            targets = targets.to(device).float()  # [batch_size, 5]
            
            optimizer.zero_grad()
            outputs = model(tree_tensors_device)  # [batch_size, 5]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * targets.size(0)
            train_count += targets.size(0)
        
        avg_train_loss = train_loss / train_count
        
        model.eval()
        val_loss = 0
        val_count = 0
        
        with torch.no_grad():
            for tree_tensors, targets in val_loader:
                trees, comps_first, comps_vectors, comps_third, loops_tensor, expr_tensor = tree_tensors
                tree_tensors_device = (
                    trees,
                    comps_first.to(device),
                    comps_vectors.to(device),
                    comps_third.to(device),
                    loops_tensor.to(device),
                    expr_tensor.to(device)
                )
                targets = targets.to(device).float()
                
                outputs = model(tree_tensors_device)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * targets.size(0)
                val_count += targets.size(0)
        
        avg_val_loss = val_loss / val_count
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with validation loss: {best_val_loss:.6f}")
    
    return model

def calculate_error_percentage(model, data_loader, dataset, device):
    model.eval()
    with torch.no_grad():
        for tree_tensors, targets in data_loader:
            trees, comps_first, comps_vectors, comps_third, loops_tensor, expr_tensor = tree_tensors
            tree_tensors_device = (
                trees,
                comps_first.to(device),
                comps_vectors.to(device),
                comps_third.to(device),
                loops_tensor.to(device),
                expr_tensor.to(device)
            )
            targets = targets.to(device).float()  # [batch_size, 5]
            outputs = model(tree_tensors_device)  # [batch_size, 5]
            
            # Denormalize predictions and targets
            pred_exec_times = outputs * dataset.exec_time_std.to(device) + dataset.exec_time_mean.to(device)
            true_exec_times = targets * dataset.exec_time_std.to(device) + dataset.exec_time_mean.to(device)
            
            # Calculate error percentage for each schedule
            error_percentage = torch.abs(pred_exec_times - true_exec_times) / true_exec_times * 100
            
            # Print results for the first sample in the batch
            print("\nExecution Time Predictions and Error Percentages for First Sample:")
            for i in range(NUM_SCHEDULES):
                print(f"Schedule {i+1} (Scaling Factor {dataset.scaling_factors[i]:.1f}):")
                print(f"  Predicted: {pred_exec_times[0, i].item():.6f}")
                print(f"  True: {true_exec_times[0, i].item():.6f}")
                print(f"  Error Percentage: {error_percentage[0, i].item():.2f}%")
            break  # Only process the first batch

def main():
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0001
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = TiramisuDataset("tiramisu_dataset.pt")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
    test_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
    
    first_part_size = dataset[0][0][1].shape[-1]  # 10
    third_part_size = dataset[0][0][3].shape[-1]  # Verify this in your data
    input_size = first_part_size + third_part_size
    
    model = Model_Recursive_LSTM_v2(
        input_size=input_size,
        comp_embed_layer_sizes=[600, 350, 200, 180],
        drops=[0.225, 0.225, 0.225, 0.225],
        output_size=NUM_SCHEDULES,
        lstm_embedding_size=100,
        expr_embed_size=100,
        loops_tensor_size=8,
        device=device,
        num_layers=1,
        bidirectional=True
    )
    
    expected_input_size = first_part_size + 200 + third_part_size + 100
    print(f"Expected input size to first comp_embedding_layer: {expected_input_size}")
    print(f"Actual first layer input size: {model.comp_embedding_layers[0].weight.shape[1]}")
    print(f"Execution time means: {dataset.exec_time_mean}")
    print(f"Execution time stds: {dataset.exec_time_std}")
    
    trained_model = train_model(model, train_loader, val_loader, num_epochs, device, learning_rate)
    calculate_error_percentage(trained_model, test_loader, dataset, device)

if __name__ == "__main__":
    main()
