import torch 
import torch.nn as nn
import numpy as np

#numpy version

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

#Y must me one hot encoded 
# class-0 : [0,0,0]
# class-1 : [0,1,0]
# class-2 : [0,0,1]
Y = np.array([1,0,0])
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)

print(f'Loss-1-good: {l1:.4f}')
print(f'Loss-2-bad: {l2:.4f}')

# Torch version

loss = nn.CrossEntropyLoss()
y_pred_good = torch.from_numpy(Y_pred_good).reshape(1,-1)
y_pred_bad = torch.from_numpy(Y_pred_bad).reshape(1,-1)
y = torch.tensor([0])


l1 = loss(y_pred_good, y)
l2 = loss(y_pred_bad, y)

print(l1, l2)
print(l1.item(), l2.item())

_, prediction1 = torch.max(y_pred_good, 1)
_, prediction2 = torch.max(y_pred_bad, 1)

print(prediction1, prediction2)

# Torch version with multiple samples
#3 samples
Z = torch.tensor([2, 0 ,1])
#nsamples x nclasses = 3x3
Z_pred_good = torch.tensor([[0.1, 1.0, 2.1],[2.0, 1.0, 0.1], [0.1, 3.0, 0.1]])
Z_pred_bad = torch.tensor([[2.1, 1.0, 0.1],[0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])

l3 = loss(Z_pred_good, Z)
l4 = loss(Z_pred_bad, Z)

print(l3.item())
print(l4.item())

_, prediction3= torch.max(Z_pred_good, 1)
_, prediction4 = torch.max(Z_pred_bad, 1)
print(prediction3, prediction4)


#Note:
#For multiclass problems use nn.CrossEntropyLoss() and no softmax at the end as its already done by Crossentropyloss in pytorch
#For single class use nn.BCELoss() and sigmoid at the end.