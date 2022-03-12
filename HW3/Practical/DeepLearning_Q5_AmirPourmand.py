#!/usr/bin/env python
# coding: utf-8

# # Importing the dependencies

# In[50]:


get_ipython().system('pip install clean-text[gpl]')


# In[48]:


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.datasets import YelpReviewPolarity
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
from torchtext.vocab import GloVe
from nltk import word_tokenize, sent_tokenize, RegexpTokenizer
import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from tqdm.notebook import tqdm



nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


# # Downloading and preparing dataset

# In[2]:


# run this cell to prepare your data


# sample
def sample_k_array(mat, k, labels=2):
  data = []
  for label in range(1, labels + 1):
    temp_mat = mat[mat[:,0] == label]
    temp_array = temp_mat[np.random.choice(temp_mat.shape[0], k, replace=False), :]
    for item in temp_array:
      data.append(item)
  return np.array(data)

# download dataset
YelpReviewPolarity(root='.', split=('train', 'test'))

# reading train & test data
train_dataframe = pd.read_csv('YelpReviewPolarity/yelp_review_polarity_csv/train.csv')
val_dataframe = pd.read_csv('YelpReviewPolarity/yelp_review_polarity_csv/test.csv')

# renaming columns
train_dataframe = train_dataframe.rename(columns={    train_dataframe.columns[0]: 'label', train_dataframe.columns[1]: 'text'})

val_dataframe = val_dataframe.rename(columns={    val_dataframe.columns[0]: 'label', val_dataframe.columns[1]: 'text'})


train_mat = train_dataframe.values
val_mat = val_dataframe.values
train_data = sample_k_array(train_mat, 5000)
val_data = sample_k_array(val_mat, 1000)
train_data = pd.DataFrame({
    'text': train_data[:, 1],
    'label': train_data[:, 0]
})
val_data = pd.DataFrame({
    'text': val_data[:, 1],
    'label': val_data[:, 0]
})
train_data['label'] -= 1
val_data['label'] -= 1


# In[3]:


# download Glove 100-dim vectors
glove_embedding = GloVe(name='6B', dim=100)


# In[4]:


train_data


# In[5]:


val_data


# In[21]:


def count_words(sentence):
    return len(sentence.split(' '))


# In[7]:


def convert_to_glove(tokenized):
    temp = []
    for word in tokenized:
        temp.append(glove_embedding[word])
    return temp

def remove_stop_words(tokenized):
    return [word for word in tokenized if not word in stopwords.words()]


# In[ ]:





# In[51]:


import re
from nltk.corpus import stopwords

def remove_stop_words_quick(sentence):
    cachedStopWords = stopwords.words("english")
    pattern = re.compile(r'\b(' + r'|'.join(cachedStopWords) + r')\b\s*')
    return pattern.sub('', sentence)


# In[67]:


from cleantext import clean
def clean_sentences(sentence):
    return clean(sentence,
        fix_unicode=True,               # fix various unicode errors
        to_ascii=True,                  # transliterate to closest ASCII representation
        lower=True,                     # lowercase text
        no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
        no_urls=True,                  # replace all URLs with a special token
        no_emails=True,                # replace all email addresses with a special token
        no_phone_numbers=False,         # replace all phone numbers with a special token
        no_numbers=True,               # replace all numbers with a special token
        no_digits=True,                # replace all digits with a special token
        no_currency_symbols=False,      # replace all currency symbols with a special token
        no_punct=True,                 # remove punctuations
        lang="en"    )                   # set to 'de' for German special handling

def remove_small_len(sentence):
    return str.join(' ',[item if len(item)>2 else '' for item in word_tokenize(sentence) ])


# In[54]:


train_data['text_cleaned'] = train_data['text'].apply(clean_sentences)
train_data['text_cleaned'] = train_data['text_cleaned'].apply(remove_stop_words_quick)
train_data['text_cleaned'] = train_data['text_cleaned'].apply(remove_small_len)


# In[75]:


val_data['text_cleaned'] = val_data['text'].apply(clean_sentences)
val_data['text_cleaned'] = val_data['text_cleaned'].apply(remove_stop_words_quick)
val_data['text_cleaned'] = val_data['text_cleaned'].apply(remove_small_len)


# In[69]:


train_data['text_cleaned'].apply(count_words).hist(bins=100)


# In[10]:


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device


# In[11]:


batch_size = 128


# In[71]:


train_data['text_cleaned'].apply(word_tokenize)


# In[ ]:





# In[76]:


from tqdm.notebook import tqdm

class CustomDataset(Dataset):

    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        data_frame_row = self.data.loc[idx]
        raw_text = data_frame_row['text_cleaned'].lower()
        splitted_text = word_tokenize(raw_text)
        label = data_frame_row['label']

        number_of_words = 200
        seq_embbeding = np.zeros((number_of_words, 100))
        for idx, token in enumerate(splitted_text):
            if idx>= number_of_words:
              break
            seq_embbeding[idx, :] = glove_embedding[token]
        
        seq_embbeding = torch.Tensor(seq_embbeding)
        return seq_embbeding, label
        


train_loader = DataLoader(CustomDataset(train_data[['text_cleaned','label']]), batch_size = batch_size, shuffle = True)
val_loader = DataLoader(CustomDataset(val_data[['text_cleaned','label']]), batch_size = batch_size, shuffle = True)


# In[77]:


x,y=next(iter(train_loader))
x.shape


# # Defining Model

# In[81]:


class YelpClassifier(nn.Module):

    def __init__(self):
        super(YelpClassifier,self).__init__()


        self.lstm = nn.LSTM(input_size=100,hidden_size=64, num_layers=2,
                            batch_first=True)
        
        self.fc2 = nn.Sequential(
            nn.Linear(64,16),
            nn.ReLU(),

            nn.Linear(16,2)
        )

        self.loss_ = nn.CrossEntropyLoss()


    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:,-1,:]
        x = self.fc2(x)
        return x

  
    def loss(self, outputs, targets):
        return self.loss_(outputs, targets)


# In[82]:


print(YelpClassifier())


# In[82]:





# # Training & Evaluation

# In[88]:


from sklearn.metrics import f1_score
# your code

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def eval_model(model, data_loader, device):

    n = len(data_loader.dataset)
    model.eval()

    sum =0 
    with torch.no_grad():
        for x, y in data_loader:
            x= x.to(device)
            y=y.to(device)
            y_pred = model(x)
            y_pred = torch.argmax(y_pred, axis=-1)
            sum = sum + f1_score(y,y_pred)*x.shape[0]
    
    return sum/n

def train(model, train_loader, val_loader, optimizer, num_epochs, device):

    train_loss_history = np.zeros((num_epochs,))
    val_loss_history = np.zeros((num_epochs,))
    train_f1_history = np.zeros((num_epochs,))
    val_f1_history = np.zeros((num_epochs,))

    for epoch in range(num_epochs):
        train_loss = 0
        model.train()
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = model.loss(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.shape[0]
        
        train_loss = train_loss / len(train_loader.dataset)

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x,y= x.to(device),y.to(device)
                n = x.shape[0]
                y_pred = model(x).to(device)
                loss = model.loss(y_pred, y)
                val_loss += loss.item() * x.shape[0]

        val_loss = val_loss / len(val_loader.dataset)

        train_f1,val_f1 = eval_model(model, train_loader, device),eval_model(model, val_loader, device)

        train_loss_history[epoch] = train_loss
        val_loss_history[epoch] = val_loss
        train_f1_history[epoch] = train_f1
        val_f1_history[epoch] = val_f1

        print(f"Epoch {epoch + 1} / {num_epochs} Training Loss = {train_loss:.5f} Test Loss = {val_loss:.5f}")
        print(f"Training F1 score = {train_f1:.5f} F1 score = {val_f1:.5f}")
    
    return train_loss_history, val_loss_history, train_f1_history, val_f1_history

n_epochs = 15
model = YelpClassifier()
model=model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train_loss,test_loss,train_f1,test_f1 = train(model, train_loader, val_loader, optimizer, n_epochs, device)


# In[ ]:





# # Draw Loss & F1-score

# In[89]:


# your code
plt.plot(np.arange(0, n_epochs), train_loss, color = 'r', label = 'train')
plt.plot(np.arange(0, n_epochs), test_loss, color = 'g', label = 'test')
plt.title("loss")

plt.plot(np.arange(0, n_epochs), train_f1, color = 'r', label = 'train')
plt.plot(np.arange(0, n_epochs), test_f1, color = 'g', label = 'test')
plt.title("f1")


# In[ ]:




