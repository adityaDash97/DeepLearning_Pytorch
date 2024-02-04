import torch 
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#0. Prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target 
n_samples, n_features = X.shape
#print(n_samples,n_features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
#scale
sc = StandardScaler() # make our features to have 0 mean and unit variance. Always recommended when we deal with logistic regression##
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)
#1. Model
#f = wx + b , sigmoid at the end
class LogisticRegression(nn.Module):
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1) # 1 is output size here
        
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
model = LogisticRegression(n_features)

#2. Loss and Optimizer

learning_rate = 0.01
criterion = nn.BCELoss() # Binary cross entropy loss used here in Logistic regression
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#3.Training Loop

num_epochs = 200
for epoch in range(num_epochs):
    #forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    #backward pass
    loss.backward()
    #update
    optimizer.step()
    # zero the gradients
    optimizer.zero_grad()
    
    if (epoch+1) % 10 == 0:
        print(f'epoch = {epoch+1}, loss = {loss.item():.4f}')
        
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_classes = y_pred.round()
    acc = y_pred_classes.eq(y_test).sum() / float(y_test.shape[0])
    print(f'Accuracy = {acc:.4f}')
    
    
    


# Assuming X and Y are your original data in numpy arrays
X_numpy = X_test.numpy()
Y_numpy = y_test.numpy()

# Convert X to a PyTorch tensor
X_tensor = torch.from_numpy(X_numpy.astype(np.float32))

# Make predictions on the entire training set
with torch.no_grad():
    predicted = model(X_tensor).numpy()

# Flatten Y_numpy to remove the singleton dimension
Y_numpy = Y_numpy.flatten()

# Plot the actual data points for class 0 in red ('ro')
plt.plot(X_numpy[Y_numpy == 0][:, 0], Y_numpy[Y_numpy == 0], 'ro', label='Class 0 (Actual)')

# Plot the actual data points for class 1 in blue ('bo')
plt.plot(X_numpy[Y_numpy == 1][:, 0], Y_numpy[Y_numpy == 1], 'bo', label='Class 1 (Actual)')

# Add labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# Save the plot
plt.savefig('plot_LR_binary_classification.png')