#!/usr/bin/env python
# coding: utf-8

# # CE-40719: Deep Learning
# ## HW5 - GAN (100 points)
# 
# #### Name: Amir Pourmand
# #### Student No.: 99210259

# ### 1) Import Libraries

# In[1]:


# !pip install colabcode > /dev/null
# from colabcode import ColabCode
# ColabCode(port=10000)
import sys


# In[2]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10, 3) # set default size of plots


# ### 2) Loading Dataset (10 points)
# 
# In this notebook, you will use `MNIST` dataset to train your GAN. You can see more information about this dataset [here](http://yann.lecun.com/exdb/mnist/). This dataset is a 10 class dataset. It contains 60000 grayscale images (50000 for train and 10000 for test or validation) each with shape (3, 28, 28). Every image has a corresponding label which is a number in range 0 to 9.

# In[3]:


# MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor(), download=True)


# In[4]:


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################ Problem 01 (5 pts) ################
# define hyper parameters
batch_size = 64
d_lr = 1e-4
g_lr = 1e-4
n_epochs = 100
####################### End ########################
z_dim = 100


# In[5]:



################ Problem 02 (5 pts) ################
# Define Dataloaders
changed_dataset = torch.utils.data.TensorDataset(train_dataset.data.float()/255, train_dataset.targets)
changed_dataset = torch.utils.data.TensorDataset(test_dataset.data.float()/255, test_dataset.targets)

train_loader = torch.utils.data.DataLoader(dataset=changed_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=changed_dataset, batch_size=batch_size, shuffle=False)
####################### End ########################


# ### 3) Defining Network (30 points)
# At this stage, you should define a network that improves your GAN training and prevents problems such as mode collapse and vanishing gradients.

# In[6]:


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminator = nn.Sequential(
            ################ Problem 03 (15 pts) ################
            # use linear or convolutional layer
            # use arbitrary techniques to stabilize training
            nn.Dropout(),
            nn.Linear(784, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.Dropout(),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
            ####################### End ########################
        )

    def forward(self, x):
        return self.discriminator(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.generator = nn.Sequential(
            ################ Problem 04 (15 pts) ################
            # use linear or convolutional layer
            # use arbitrary techniques to stabilize training
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 784),
            nn.Sigmoid()
            ####################### End ########################
        )

    def forward(self, z):
        return self.generator(z)


# ### 4) Train the Network 
# At this step, you are going to train your network.

# In[7]:


################ Problem 05 (5 pts) ################
# Create instances of modules (discriminator and generator)
# don't forget to put your models on device
discriminator = Discriminator().to(device)
generator = Generator().to(device)
####################### End ########################


# In[8]:


################ Problem 06 (5 pts) ################
# Define two optimizer for discriminator and generator
d_optimizer = optim.Adam(discriminator.parameters(),lr=d_lr)
g_optimizer = optim.Adam(generator.parameters(),lr=g_lr)
####################### End ########################


# In[15]:


plot_frequency = 8

for epoch in range(n_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        ################ Problem 07 (15 pts) ################
        # put your inputs on device
        # Prepare what you need for training, like inputs for modules and variables for computing loss
        images = images.flatten(start_dim=1)

        real_img = images.to(device)

        fake_labels = torch.zeros(images.shape[0], 1).to(device)
        real_labels = torch.ones(images.shape[0], 1).to(device)
        z = torch.randn(images.shape[0], 128).to(device)

        generated_images = generator(z)

        d_optimizer.zero_grad()

        ####################### End ########################



        ################ Problem 08 (10 pts) ################
        # calculate discriminator loss and update it

        z = torch.randn(images.shape[0], 128).to(device)
        generated_images = generator(z)

        g_optimizer.zero_grad()

        d_loss = (F.binary_cross_entropy(discriminator(generated_images.detach()), fake_labels) +
                  F.binary_cross_entropy(discriminator(real_img), real_labels))
        d_loss.backward()
        d_optimizer.step()

        ####################### End ########################
        
        

        ################ Problem 09 (10 pts) ################
        # calculate generator loss and update it

        
        g_loss = F.binary_cross_entropy(discriminator(generated_images), real_labels)
        g_loss.backward()
        g_optimizer.step()


        ####################### End ########################


    ################ Problem 10 (10 pts) ################
    # plot some of the generated pictures based on plot frequency variable

    if (epoch % plot_frequency == 0):
        plt.subplots(1,10)
        for j in range(10):
            plt.subplot(1,10,j+1)
            plt.imshow(generated_images[j].detach().cpu().view(28, 28).numpy())
        plt.show()

    ####################### End ########################
    
    print("epoch: {} \t discriminator last batch loss: {} \t generator last batch loss: {}".format(epoch + 1, 
                                                                                            d_loss.item(), 
                                                                                            g_loss.item())
    )


# ### 5) Save Generator
# Save your final generator parameters. Upload it with your other files.

# In[18]:


################ Problem 11 (5 pts) ################
# save state dict of your generator
path = "/content/my_parameters"
torch.save(generator.state_dict(), path)
####################### End ########################


# In[ ]:




