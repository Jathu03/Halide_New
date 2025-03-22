import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os

# Assuming Model_Recursive_LSTM_v2 is in a separate file called model.py
from model import Model_Recursive_LSTM_v2  # You'll need to put the model class in a separate file

class TiramisuDataset(Dataset):
    def __init__(self, dataset_path="tiramisu_dataset.pt"):
        self.data = torch.load(dataset_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        # The model expects a tuple of tensors in forward()
        tree_tensors = (
            {"roots": [{"child_list": [], "has_comps": True, "computations_indices": torch.tensor([i for i in range(sample['comps_tensor'].shape[0])]), "loop_index": torch.tensor([0])}]},
            sample['comps_tensor'][:, :10],  # First part (adjust slicing based on your features)
            sample['comps_tensor'][:, 10:74],  # Transformation vectors (adjust based on MAX_TAGS * MAX_NUM_TRANSFORMATIONS)
            sample['comps_tensor'][:, 74:],  # Third part
            sample['loops_tensor'],
            sample['expr_tensor']
        )
        return tree_tensors, sample['exec_time']

def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    device="cuda" if torch.cuda.is_available() else "cpu",
    learning_rate=0.001
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_path = "best_model.pt"
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_count = 0
        
        for batch_idx, (tree_tensors, targets) in enumerate(tqdm(train_loader)):
            # Move data to device
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
        
        # Validation phase
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
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with validation loss: {best_val_loss:.6f}")

def main():
    # Hyperparameters
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load dataset
    dataset = TiramisuDataset("tiramisu_dataset.pt")
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    # Adjust input_size based on your computation tensor features
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
    
    # Train the model
    train_model(model, train_loader, val_loader, num_epochs, device, learning_rate)

if __name__ == "__main__":
    main()
