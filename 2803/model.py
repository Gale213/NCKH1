import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.weight.size(1) ** 0.5)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        return output

class GraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphClassifier, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = self.dropout(x)
        x = self.gcn2(x, adj)
        return F.log_softmax(x, dim=1)

# Example usage
if __name__ == "__main__":
    # Suppose you have the following data
    num_nodes = 10
    num_features = 20
    num_classes = 2
    num_edges = 50
    X = torch.rand((num_nodes, num_features))  # Feature matrix
    A = torch.randint(2, (num_nodes, num_nodes))  # Adjacency matrix
    y = torch.randint(num_classes, (num_nodes,))  # Class labels

    # Initialize the GCN-based graph classifier
    model = GraphClassifier(num_features, 16, num_classes)

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X, A)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
