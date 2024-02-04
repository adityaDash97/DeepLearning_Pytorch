#ImageFolder
#Scheduler
#Transfer Learning
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import time 
import os
import copy

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

data_transforms = {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

data_dir = 'data/hymenoptera'
sets = ['train', 'val']
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
#print(image_datasets)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(class_names)

def imshow(inp, title):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp +mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.savefig("transfer.png")
    
inputs, classes = next(iter(dataloaders['train']))

out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch{}/{}'.format(epoch, num_epochs - 1))
        print('-'*10)
        
        #Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # set model to training mode
            else:
                model.eval() # set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            #Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                #forward
                #track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    #backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                #statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step()
                
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
num_features = model.fc.in_features

model.fc = nn.Linear(num_features, 2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

#scheduler

step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=20)



#Result

# Epoch0/19
# ----------
# train Loss: 0.5862 Acc: 0.6967
# val Loss: 0.4307 Acc: 0.8693

# Epoch1/19
# ----------
# train Loss: 0.5322 Acc: 0.7787
# val Loss: 0.4006 Acc: 0.8758

# Epoch2/19
# ----------
# train Loss: 0.5227 Acc: 0.7418
# val Loss: 0.3344 Acc: 0.9150

# Epoch3/19
# ----------
# train Loss: 0.4484 Acc: 0.7992
# val Loss: 0.3214 Acc: 0.8954

# Epoch4/19
# ----------
# train Loss: 0.4302 Acc: 0.8033
# val Loss: 0.2886 Acc: 0.9542

# Epoch5/19
# ----------
# train Loss: 0.4422 Acc: 0.7951
# val Loss: 0.2616 Acc: 0.9542

# Epoch6/19
# ----------
# train Loss: 0.4128 Acc: 0.8197
# val Loss: 0.2527 Acc: 0.9412

# Epoch7/19
# ----------
# train Loss: 0.4099 Acc: 0.8525
# val Loss: 0.2647 Acc: 0.9216

# Epoch8/19
# ----------
# train Loss: 0.4568 Acc: 0.7664
# val Loss: 0.2470 Acc: 0.9477

# Epoch9/19
# ----------
# train Loss: 0.3518 Acc: 0.8811
# val Loss: 0.2466 Acc: 0.9412

# Epoch10/19
# ----------
# train Loss: 0.4057 Acc: 0.8115
# val Loss: 0.2431 Acc: 0.9412

# Epoch11/19
# ----------
# train Loss: 0.4020 Acc: 0.8402
# val Loss: 0.2458 Acc: 0.9412

# Epoch12/19
# ----------
# train Loss: 0.3908 Acc: 0.8156
# val Loss: 0.2424 Acc: 0.9477

# Epoch13/19
# ----------
# train Loss: 0.3820 Acc: 0.8607
# val Loss: 0.2524 Acc: 0.9412

# Epoch14/19
# ----------
# train Loss: 0.3676 Acc: 0.8607
# val Loss: 0.2491 Acc: 0.9346

# Epoch15/19
# ----------
# train Loss: 0.4077 Acc: 0.8156
# val Loss: 0.2388 Acc: 0.9477

# Epoch16/19
# ----------
# train Loss: 0.3893 Acc: 0.8484
# val Loss: 0.2391 Acc: 0.9542

# Epoch17/19
# ----------
# train Loss: 0.4402 Acc: 0.7623
# val Loss: 0.2438 Acc: 0.9477

# Epoch18/19
# ----------
# train Loss: 0.4178 Acc: 0.8115
# val Loss: 0.2350 Acc: 0.9477

# Epoch19/19
# ----------
# train Loss: 0.3938 Acc: 0.8115
# val Loss: 0.2397 Acc: 0.9346

# Training complete in 1m 6s
# Best val Acc: 0.954248
