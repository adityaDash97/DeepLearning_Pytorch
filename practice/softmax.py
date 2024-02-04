import torch 
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0,1.0,0.1])
outputs = softmax(x)
print("softmax numpy: ", outputs)

x_tensor = torch.from_numpy(x)
outputs = torch.softmax(x_tensor, dim=0)
print("Torch softmax: ",outputs)