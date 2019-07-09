
# coding: utf-8

# In[1]:

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import os
import gc

# Any results you write to the current directory are saved as output.


# In[2]:

train_data=pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")
test_data[test_data.id_code=="980b5ca190ce"]


# In[3]:

from PIL import Image
import torchvision.transforms as transforms
image_dim=64
transform_ori=transforms.Compose([transforms.Resize((image_dim,image_dim)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.225,0.225,0.225]),])
train_data["image_data"]=train_data.id_code.map(lambda x:transform_ori(Image.open("../input/train_images/"+x+".png")))


# In[4]:

test_data["image_data"]=test_data.id_code.map(lambda x:transform_ori(Image.open("../input/test_images/"+x+".png")))


# In[5]:

#Note that transform can dirextly act on the open object
import torch
print(type(train_data.image_data[1:5].values))
train_input_tensor=torch.stack(train_data.image_data.values.tolist())
test_input_tensor=torch.stack(test_data.image_data.values.tolist())
train_target=torch.tensor(train_data.diagnosis.values)
# train_data.head()


# In[6]:

# print(len(train_input_tensor[0:40]))


# In[7]:

from torch import nn
class ConvNN(nn.Module):
	def __init__(self):
		super(ConvNN,self).__init__()
		self.cnn1=nn.Conv2d(in_channels=3,out_channels=8,kernel_size=5,stride=1,padding=2)

		self.batchnorm1=nn.BatchNorm2d(8)
		self.relu=nn.ReLU()
		self.maxpool1=nn.MaxPool2d(kernel_size=8)

		self.cnn2=nn.Conv2d(in_channels=8,out_channels=32,kernel_size=5,stride=1,padding=2)
		self.batchnorm2=nn.BatchNorm2d(32)
		self.maxpool2=nn.MaxPool2d(kernel_size=8)

		self.fc1=nn.Linear(in_features=int((image_dim/64)**2*32),out_features=int((image_dim/64)**2*16))
		self.dropout=nn.Dropout(p=0.5)
		self.fc2=nn.Linear(in_features=int((image_dim/64)**2*16),out_features=int((image_dim/64)**2*8))
		self.dropout=nn.Dropout(p=0.5)
# 		self.fc3=nn.Linear(in_features=int((image_dim/64)**2*8),out_features=int((image_dim/64)**2*4))
# 		self.dropout=nn.Dropout(p=0.5)
# 		self.fc4=nn.Linear(in_features=int((image_dim/64)**2*4),out_features=int((image_dim/64)**2*1))
# 		self.dropout=nn.Dropout(p=0.5)
		self.fc5=nn.Linear(in_features=int((image_dim/64)**2*8),out_features=5)
		self.softmaxl=nn.Softmax(5)

	def forward(self,x):
		out=self.cnn1(x)
		out=self.batchnorm1(out)
		out=self.relu(out)
		out=self.maxpool1(out)
# 		print(out.shape)
		out=self.cnn2(out)
		out=self.batchnorm2(out)
		out=self.maxpool2(out)
		out=out.view(-1,int((image_dim/64)**2*32))
		out=self.fc1(out)
		out=self.relu(out)
		out=self.dropout(out)
		out=self.fc2(out)
		out=self.relu(out)
		out=self.dropout(out)
# 		out=self.fc3(out)
# 		out=self.relu(out)
# 		out=self.dropout(out)
# 		out=self.fc4(out)
# 		out=self.relu(out)
# 		out=self.dropout(out)
		out=self.fc5(out)
		# out=self.softmax(out)
		return out


# In[8]:

model = ConvNN()
cuda_yes = torch.cuda.is_available()
if cuda_yes:
    print("yippy cuda available!")
    model = model.cuda()    
loss_fn = nn.CrossEntropyLoss()        
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)


# In[9]:

import time
from torch.autograd import Variable
num_epochs=10000
batch_size=100
#the function needs an iterable of tensors
train_load=torch.utils.data.DataLoader(dataset=train_input_tensor,
										batch_size=batch_size,
										shuffle=False)
train_target_load=torch.utils.data.DataLoader(dataset=train_target,
                                             batch_size=batch_size,
                                             shuffle=False)
# print(train_input_tensor.size())
train_loss=[]
test_loss=[]
train_accuracy=[]
test_accuracy=[]

for epoch in range(num_epochs):
    start=time.time()
    correct=0
    iterations=0
    iter_loss=0

    model.train()
    i=0
    for inputs,labels in zip(train_load,train_target_load):
        inputs=Variable(inputs)
        labels=Variable(labels)

        if cuda_yes:
            inputs=inputs.cuda()
            labels=labels.cuda()
#         print(inputs.shape,labels.shape)

        optimizer.zero_grad()
#             print('hello here')
        outputs=model(inputs)
        loss=loss_fn(outputs,labels)
#             print(loss.item())
        iter_loss+=loss.item()
        loss.backward()
        optimizer.step()
#             print('hello there')

        _,predicted=torch.max(outputs,1)
        correct+=(predicted==labels).sum()

        iterations+=1
        i+=1

    train_loss.append(iter_loss/iterations)
    train_accuracy.append((100*correct/len(train_input_tensor)))

    iter_loss=0.0
    correct=0
    iterations=0
    if(epoch%1000==0):
        print('Epoch {}/{}, Training Loss: {:.3f}, Train Accuracy: {:.3f}'.format(epoch+1,num_epochs,train_loss[-1],train_accuracy[-1]))


# In[10]:

test_input_tensor=Variable(test_input_tensor).cuda()
results=model(test_input_tensor)
numerical_results=np.argmax(results.cpu().detach().numpy(),axis=1)
id_series=test_data.id_code.as_matrix()
# print(type(id_series))
id_array=id_series.reshape((len(id_series),1))
numerical_results=numerical_results.reshape((len(numerical_results),1))
ans=np.hstack((id_array,numerical_results))
ans_df=pd.DataFrame(ans,columns=['id_code','diagnosis'])
ans_df.to_csv('submission.csv',index=False)


# In[ ]:



