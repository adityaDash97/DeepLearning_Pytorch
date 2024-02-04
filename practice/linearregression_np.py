import numpy as np
import time 

X = np.array([1, 2, 3, 4, 5], dtype=np.float32)
Y = np.array([2, 4, 6, 8, 10], dtype=np.float32)

W = 0.0

#Model Prediction
def forward(x):
    return W * x

#Loss = Mean Squared Error
def loss(y, y_prediction):
    return ((y-y_prediction)**2).mean()

#Gradient
def gradient(x, y, y_prediction):
    return np.dot(2*x, y_prediction-y).mean()

print(f"Prediction before training: f(8) = {forward(8):.3f}")

#Training Loop
learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):
    #Forward pass
    y_pred = forward(X)
    
    #loss
    l = loss(Y, y_pred)
    
    #gradient 
    dw = gradient(X,Y,y_pred)
    
    #Update weighs
    W = W - (learning_rate * dw)
    
    if epoch % 1 == 0:
        print(f'epoch = {epoch+1}: W = {W:.3f}, loss = {l:.8f}')
        time.sleep(1)
        
print(f"Prediction after training: f(8) = {forward(8):.3f}")