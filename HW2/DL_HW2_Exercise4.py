#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Stu no: 99210259
#stu name: Amir Pourmand


# In[16]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


# In[25]:


cuda = torch.device('cuda')


# In[2]:


get_ipython().system('wget https://github.com/mralisoltani/CNN_Tumor/raw/main/Tumor.zip')


# In[ ]:


get_ipython().system('unzip /content/Tumor.zip')


# In[ ]:





# In[74]:


# -*- coding: utf-8 -*-
"""

@author: Ali Soltani
"""


import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import pandas as pds
from torch.utils.data import DataLoader,TensorDataset,random_split

############################################################## Loading Data
n = 3762
image=[]
cw = os.getcwd().replace(os.sep, '/')
trans = transforms.Compose([transforms.ToTensor()])
for i in range(n):
#    image.append(np.asarray(Image.open(cw + "/Brain_Tumor/Image" + str(i+1) + ".jpg")))
    image.append(np.array(Image.open(cw + "/Brain_Tumor/Image" + str(i+1) + ".jpg").resize((48,48))))

temp = pds.read_csv(cw + "/Brain_Tumor.csv",index_col=None, header=None).to_numpy()
temp = temp[1:,1]
targets = np.zeros((n,1),dtype=int)
targets = []
for i in range(n):
    targets.append(int(temp[i]))

data = np.array(image)
data = data/255
data = torch.from_numpy(data).permute((0,3,2,1))
data = data.float().to(cuda)
targets = torch.tensor(targets).to(cuda)
dataset = TensorDataset(data,targets)
batch_size = 4
val_size = int(np.ceil(len(dataset)*0.2))
train_size = len(dataset) - val_size 

train_data,test_data = random_split(dataset,[train_size,val_size])


train_loader = DataLoader(train_data,batch_size = batch_size,shuffle=True)
test_loader = DataLoader(test_data,batch_size = batch_size,shuffle=True)


# In[29]:


import matplotlib.pyplot as plt
from torchvision import transforms


# In[30]:


print("count of data:", len(train_data))
train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

for i in range(batch_size):
    img = train_features[i].squeeze()
    label = train_labels[i]
    im = transforms.ToPILImage()(img).convert("RGB")
    plt.imshow(im)
    plt.show()
    print(f"Label: {label}")


# In[7]:


import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*48*48, 512),
            nn.ReLU(),
            nn.Linear(512, 24),
            nn.ReLU(),
            nn.Linear(24, 2),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# In[8]:





# In[56]:


model = NeuralNetwork().to(device)
print(model)


# In[57]:


learning_rate = 1e-3
batch_size = 64
epochs = 100

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# In[58]:


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


# In[59]:


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

losses=[]
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer)
    loss = test_loop(test_loader, model, loss_fn)
    losses.append(loss)
print("Done!")


# In[60]:


plt.plot(np.arange(0,100),losses)


# In[88]:


class NeuralNetwork2(nn.Module):
    def __init__(self):
        super(NeuralNetwork2, self).__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(36864,512),
            nn.ReLU(),
            nn.Linear(512,24),
            nn.ReLU(),
            nn.Linear(24,2)
        )

    def forward(self, x):
        logits = self.stack(x)
        return logits


# In[89]:


model2 = NeuralNetwork2().to(cuda)
print(model2)


# In[90]:


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model2.parameters(), lr=learning_rate)

losses=[]
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_loader, model2, loss_fn, optimizer)
    loss = test_loop(test_loader, model2, loss_fn)
    losses.append(loss)
print("Done!")


# In[81]:





# In[91]:


plt.plot(np.arange(0,100),losses)


# In[92]:


# accuracy is better. and in first one the model is overfitted and it will not get better accuracy 
# but in second one as time goes, it is possible that it gets better accuracy and lower loss. 


# In[ ]:




