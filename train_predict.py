import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os

# Define constants (these should match your data creation script)
MAX_NUM_TRANSFORMATIONS = 4
MAX_TAGS = 16

# Define the Model_Recursive_LSTM_v2 class (copied from your first document)
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
        
        # Create the transformation encoding layers
        self.encode_vectors = nn.Linear(MAX_TAGS, MAX_TAGS, bias=True)
        
        # Create the computation embedding layers
        for i in range(len(comp_embed_layer_sizes) - 1):
            self.comp_embedding_layers.append(
                nn.Linear(comp_embed_layer_sizes[i], comp_embed_layer_sizes[i + 1], bias=True)
            )
            nn.init.xavier_uniform_(self.comp_embedding_layers[i].weight)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))
            
        # Create the final regression layers
        for i in range(len(regression_layer_sizes) - 1):
            self.regression_layers.append(
                nn.Linear(regression_layer_sizes[i], regression_layer_sizes[i + 1], bias=True)
            )
            nn.init.xavier_uniform_(self.regression_layers[i].weight)
            self.regression_dropouts.append(nn.Dropout(drops[i]))
            
        # Create the feed forward network for embedding loop levels
        for i in range(len(concat_layer_sizes) - 1):
            self.concat_layers.append(
                nn.Linear(concat_layer_sizes[i], concat_layer_sizes[i + 1], bias=True)
            )
            nn.init.xavier_uniform_(self.concat_layers[i].weight)
            nn.init.zeros_(self.concat_layers[i].weight)
            self.concat_dropouts.append(nn.Dropout(drops[i]))
            
        # Output layer
        self.predict = nn.Linear(regression_layer_sizes[-1], output_size, bias=True)
        nn.init.xavier_uniform_(self.predict.weight)
        
        self.ELU = nn.ELU()
        self.LeakyReLU = nn.LeakyReLU(0.01)
        
        # Parameter tensors
        self.no_comps_tensor = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, embedding_size)))
        self.no_nodes_tensor = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(1, embedding_size)))
        
        # LSTM layers
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
        
        # Embed expressions
        batch_size, num_comps, len_sequence, len_vector = functions_comps_expr_tree.shape
        x = functions_comps_expr_tree.view(batch_size * num_comps, len_sequence, len_vector)
        _, (expr_embedding, _) = self.exprs_embed(x)
        expr_embedding = expr_embedding.permute(1, 0, 2).reshape(batch_size * num_comps, -1)
        
        # Embed computations
        batch_size, num_comps, __dict__ = comps_tensor_first_part.shape
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

# Dataset class
class TiramisuDataset(Dataset):
    def __init__(self, dataset_path="tiramisu_dataset.pt"):
        self.data = torch.load(dataset_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        tree_tensors = (
            {"roots": [{"child_list": [], "has_comps": True, "computations_indices": torch.tensor([i for i in range(sample['comps_tensor'].shape[0])]), "loop_index": torch.tensor([0])}]},
            sample['comps_tensor'][:, :10],  # Adjust slicing based on your features
            sample['comps_tensor'][:, 10:74],  # Transformation vectors (64 elements)
            sample['comps_tensor'][:, 74:],  # Remaining features
            sample['loops_tensor'],
            sample['expr_tensor']
        )
        return tree_tensors, sample['exec_time']

def train_model(model, train_loader, val_loader, num_epochs=100, device="cuda" if torch.cuda.is_available() else "cpu", learning_rate=0.001):
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
            tree_tensors = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in tree_tensors)
            targets = targets.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(tree_tensors)
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
                tree_tensors = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in tree_tensors)
                targets = targets.to(device).float()
                
                outputs = model(tree_tensors)
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

def main():
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = TiramisuDataset("tiramisu_dataset.pt")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    input_size = dataset[0][0][1].shape[-1] + dataset[0][0][2].shape[-1] + dataset[0][0][3].shape[-1]
    model = Model_Recursive_LSTM_v2(
        input_size=input_size,
        comp_embed_layer_sizes=[600, 350, 200, 180],
        drops=[0.225, 0.225, 0.225, 0.225],
        output_size=1,
        lstm_embedding_size=100,
        expr_embed_size=100,
        loops_tensor_size=8,
        device=device,
        num_layers=1,
        bidirectional=True
    )
    
    train_model(model, train_loader, val_loader, num_epochs, device, learning_rate)

if __name__ == "__main__":
    main()
