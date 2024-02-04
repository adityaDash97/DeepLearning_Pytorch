import torch 
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

#0. Prepare Data
X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
Y = Y.view(Y.shape[0],1)

n_samples, n_features = X.shape
#1. Model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)
#2. loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#3. Training loop
num_epochs = 100
for epoch in range(num_epochs):
    #forward pass and loss
    y_pred = model(X)
    loss = criterion(y_pred, Y)
    #backward pass 
    loss.backward()
    #update
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch+1) % 10 == 0:
        print(f'eopch: {epoch+1}, loss = {loss.item():.4f}')
        
#4. Plot
predicted = model(X).detach().numpy()
#print(predicted)
plt.plot(X_numpy, Y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.savefig('plot.png')