import torch.nn as nn

class HalideLSTM(nn.Module):
    def __init__(self, edge_input_dim, node_input_dim, sched_input_dim, hidden_dim=128, num_layers=2):
        super(HalideLSTM, self).__init__()
        
        self.edge_lstm = nn.LSTM(edge_input_dim, hidden_dim, num_layers, batch_first=True)
        self.node_lstm = nn.LSTM(node_input_dim, hidden_dim, num_layers, batch_first=True)
        
        self.context_fc = nn.Linear(sched_input_dim, hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, edge_seq, node_seq, sched_context):
        # Process edge sequence
        edge_out, (edge_h, _) = self.edge_lstm(edge_seq)
        edge_out = edge_h[-1]  # Take last hidden state
        
        # Process node sequence
        node_out, (node_h, _) = self.node_lstm(node_seq)
        node_out = node_h[-1]
        
        # Process scheduling context
        context_out = self.context_fc(sched_context)
        
        # Combine all features
        combined = torch.cat([edge_out, node_out, context_out], dim=1)
        out = self.fc(combined)
        return out

# Initialize model
edge_dim = dataset[0]['edge_seq'].shape[1]
node_dim = dataset[0]['node_seq'].shape[1]
sched_dim = dataset[0]['sched_context'].shape[0]
model = HalideLSTM(edge_dim, node_dim, sched_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

def train_model(model, train_loader, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_loader:
            edge_seq = batch['edge_seq'].to(device)
            node_seq = batch['node_seq'].to(device)
            sched_context = batch['sched_context'].to(device)
            target = batch['exec_time'].to(device)
            
            optimizer.zero_grad()
            output = model(edge_seq, node_seq, sched_context)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

train_model(model, train_loader)

def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch in test_loader:
            edge_seq = batch['edge_seq'].to(device)
            node_seq = batch['node_seq'].to(device)
            sched_context = batch['sched_context'].to(device)
            target = batch['exec_time'].to(device)
            
            output = model(edge_seq, node_seq, sched_context)
            predictions.extend(output.cpu().numpy())
            actuals.extend(target.cpu().numpy())
    
    predictions = np.expm1(predictions)  # Reverse log transform
    actuals = np.expm1(actuals)
    
    # Calculate error percentages
    errors = [abs(pred - act) / act * 100 for pred, act in zip(predictions, actuals)]
    
    # Print results for first 10 samples
    print("\nPredictions vs Actuals (First 10):")
    for i in range(min(10, len(predictions))):
        print(f"Schedule {i+1}:")
        print(f"Predicted: {predictions[i]:.2f} ms")
        print(f"Actual: {actuals[i]:.2f} ms")
        print(f"Error: {errors[i]:.2f}%")
    
    # Overall metrics
    mean_error = np.mean(errors)
    print(f"\nMean Absolute Percentage Error: {mean_error:.2f}%")

evaluate_model(model, test_loader)
