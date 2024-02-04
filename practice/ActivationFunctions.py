#With non-linear transformations our netwrok can learn better and perform more complex tasks.
#After each layer we typically use an activation function

#Most popular activation functions:
#    1. Step function (1 if input is gt threshold otherwise 0)
#    2. Sigmoid (1/(1+e^-x), typically used in the last layer of a binary classifn problem)
#    3. TanH (Used in hidden layers, value between -1 and +1, f(x)=2/(1+e^-2x)-1)
#    4. ReLU (f(x)=max(0,x), if you dont know what to use use a ReLU for hidden layers, 0 for negative values and output same as input for positive values)
#    5. Leaky ReLU (Imporved version of ReLU, Tries to solve vanishing gradient problem, for positive values output same as input but for negativ values a small 'a' is multiplied ex 0.001)
#    6. Softmax (Good in last layer for multi class problems)

import torch 
import torch.nn as nn
import torch.nn.functional as F

#Option 1 (Create nn modules)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        #nn.Sigmoid
        #nn.Softmax
        #nn.TanH
        #nn.LeakyReLU
    
        return out
    
#Option 2 (Use activation functions directly in forward pass)
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(input_size, hidden_size)
        
    def forward(self, x):
        out = torch.relu(self.linear1(x))#Calling relu from torch api
        out = torch.sigmoid(self.linear2(out))# " "
        #torch.softmax
        #torch.tanh MAY NOT BE AVAILABLE FROM API   
        return out
        