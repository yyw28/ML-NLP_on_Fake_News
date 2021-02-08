#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification,BertModel
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.optim import AdamW
from tqdm import tqdm
from argparse import ArgumentParser
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# In[ ]:
import pandas as pd
from sklearn.model_selection import train_test_split

file_path = '/u/erdos/csga/yyu149/research'
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[2]:


# Model parameter
#first_n_words = 1420
MAX_SEQ_LEN = 128
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
tweet_field = Field(tokenize=tokenizer.encode, use_vocab=False,lower=False, include_lengths=True, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
fields = [('label', label_field),('tweet', tweet_field)]

# TabularDataset
train, valid, test = TabularDataset.splits(path=file_path, train='Constraint_English_Train.csv', validation='Constraint_English_Val.csv',test='Constraint_English_Test.csv', format='CSV', fields=fields, skip_header=True)

# Iterators
train_iter = BucketIterator(train, batch_size=64, sort_key=lambda x: len(x.tweet),device=device, train=True, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=64, sort_key=lambda x: len(x.tweet),device=device, train=True, sort=True, sort_within_batch=True)
test_iter = Iterator(test, batch_size=64, device=device, train=False, shuffle=False, sort=False)


# In[ ]:
#for (label,inputs),_ in train_iter:
#    print(len(inputs))



print('add_lstm')
#
class BERT(nn.Module):
    def __init__(self):
        super(BERT,self).__init__()
        options_name = 'bert-base-uncased'
        self.encoder = BertModel.from_pretrained(options_name)#output_hidden_states=True)
        self.dropout = nn.Dropout(0.5)
        self.hidden_size=self.encoder.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size,2)
        #torch.nn.init.xavier_normal_(self.classifier.weight)
        #self.LSTM = nn.LSTM(input_size=self.hidden_size,hidden_size=self.hidden_size,num_layers=1,batch_first=True,bidirectional=False) 

    def forward(self,label,inputs,attention_mask):
        last_hidden_state,pooled_output = self.encoder(input_ids=inputs[0],attention_mask=attention_mask)
        #print(len(pooled_output))

        #pack_padded = pack_padded_sequence(last_hidden_state,inputs[1],batch_first=True)
        #enc_hiddens,(h,n) = self.LSTM(last_hidden_state) 
        #output = pad_packed_sequence(enc_hiddens)
        #print(len(output))
        #print(len(h))
        #print(h[0].shape)

        output=self.dropout(F.relu(pooled_output)) # BERT + DENSE LAYER only
        #output = self.dropout(F.relu(h[-1]))     # BERT + LSTM
        output = self.classifier(output)
        #output = nn.Sigmoid(output)
        return output


# Save and Load Functions
def save_checkpoint(save_path, model,valid_loss):
    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):
    
    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

def binary_accuracy(preds, y):
    #rounded_preds = torch.round(preds) 
    _, predicted = torch.max(preds, 1)
    correct = (predicted == y).float() 
    acc = correct.sum() / len(preds)
    return acc


# Training Function
def train(model,
          optimizer,
          criterion = nn.CrossEntropyLoss(), #nn.BCELoss(),
          train_loader = train_iter,
          valid_loader = valid_iter,
          eval_every = len(train_iter)//2,
          num_epochs = 8, 
          file_path = file_path,
          best_valid_loss = float("Inf")):
    
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    epoch_acc = 0
    valid_epoch_acc = 0

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (label,inputs),_ in train_loader:
            attention_mask = (inputs[0]!= PAD_INDEX).type(torch.ByteTensor)

            #attention_mask = attention_mask.to(device)

            label = label.type(torch.LongTensor)           
            label = label.to(device)

            output = model(label,inputs,attention_mask) #.squeeze  #calling forward method
            
            loss =  criterion(output,label)
            acc = binary_accuracy(output,label)  

            optimizer.zero_grad()
            loss.backward()
            #acc.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            epoch_acc += acc.item()    
            global_step += 1

    #print('running_loss',running_loss / len(train_loader))
    #print('acc',epoch_acc / len(train_loader))

            # evaluation step
            if global_step % eval_every == 0:

                model.eval()
                with torch.no_grad():                    

                    # validation loop
                    for (label,inputs),_ in valid_loader:
                        label = label.type(torch.LongTensor)           
                        label = label.to(device)

                        attention_mask = (inputs[0]!=PAD_INDEX).type(torch.ByteTensor)  
                        #attention_mask = attention_mask.to(device)

                        output = model(label,inputs,attention_mask) #.squeeze
                        
                        loss = criterion(output,label)
                        acc = binary_accuracy(output,label)
                        
                        valid_running_loss += loss.item()
                        valid_epoch_acc += acc.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                ave_train_acc = epoch_acc/eval_every
                ave_valid_acc = valid_epoch_acc/len(valid_loader)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, train acc: {:.4f}, valid acc:{:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss,ave_train_acc,ave_valid_acc))
                
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/' + 'model_b_den.pt', model, best_valid_loss)
                    save_metrics(file_path + '/' + 'metrics_b_den.pt', train_loss_list, valid_loss_list, global_steps_list)
    
    save_metrics(file_path + '/' + 'metrics_b_den.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')

model = BERT().to(device)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-5)

train(model=model, optimizer=optimizer)

# Evaluation Function
def evaluate(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (label,inputs), _ in test_loader:
            label = label.type(torch.LongTensor)           
            label = label.to(device)

            attention_mask = (inputs[0]!=PAD_INDEX).type(torch.ByteTensor)  
                #attention_mask = attention_mask.to(device)

            output = model(label,inputs,attention_mask)

            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(label.tolist())
    
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))
    
    #cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    #ax= plt.subplot()
    #sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    #ax.set_title('Confusion Matrix')

    #ax.set_xlabel('Predicted Labels')
    #ax.set_ylabel('True Labels')

    #ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    #ax.yaxis.set_ticklabels(['FAKE', 'REAL'])
    
    
best_model = BERT().to(device)
optimizer = optim.Adam(best_model.parameters(), lr=0.001)
load_checkpoint(file_path + '/model_b_den.pt', best_model)
evaluate(best_model, test_iter)

