####Training Pipeline########

#1. Design model (input, output, forward pass)
#2. Construct loss and optimizer
#3. Training loop
#   - Forward pass : Compute Prediction
#   - Backward pass : Gradients
#   - Update weights
# Note: Just need to define the model and use loss = nn.MSELoss, optimizer = torch.optim.SGD([W], lr=learning_rate)

import torch 
import torch.nn as nn
import time

X = torch.tensor([[1],[2],[3],[4],[5]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8],[10]], dtype=torch.float32)

X_test = torch.tensor([[8]], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features
#model = nn.Linear(input_size, output_size)

class LinearRegression(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.lin(x)
    
model = LinearRegression(input_size, output_size)

print(f"Prediction before training: f(8) = {model(X_test).item():.3f}")

learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#Training Loop

for epoch in range(n_iters):
    y_pred = model(X)
    l = loss(Y, y_pred)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % 5 == 0:
        [W,b] = model.parameters()
        print(f'epoch = {epoch+1}: W = {W[0][0].item():.3f}, loss = {l:.8f}')
        #time.sleep(1)
        
print(f"Prediction after training: f(8) = {model(X_test).item():.3f}")
