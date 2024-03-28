import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from Prepare import CustomGraphDataset  # Import your graph dataset class
import model 
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# Load dataset
dataset = CustomGraphDataset(root='data', file_path = r'E:\NCKH 2023 Mal\Graph\Adware')  # Provide appropriate parameters
num_classes = dataset.num_classes
num_features = dataset.num_features

# Split dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
model = GraphClassifier(num_features, hidden_dim=64, output_dim=num_classes).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for data in train_loader:
        x, adj, labels = data
        x, adj, labels = x.to(device), adj.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(x, adj)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Print training loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        x, adj, labels = data
        x, adj, labels = x.to(device), adj.to(device), labels.to(device)
        
        outputs = model(x, adj)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total:.2f}%')
