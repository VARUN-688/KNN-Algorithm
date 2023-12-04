#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud


# In[2]:


import pandas as pd

path = "your path to dataset in this the dataset is fishiris  'https://raw.githubusercontent.com/VN-688/KNN-Algorithm/main/fishiris.csv' "
path = "https://raw.githubusercontent.com/VARUN-688/KNN-Algorithm/main/fishiris.csv"
df = pd.read_csv(path)


# In[3]:


df.describe()


# In[4]:


df.dtypes


# In[5]:


x=df.loc[:,['SepalLength','SepalWidth','PetalLength','PetalWidth']]
y=df.loc[:,'Name']
print(x,y)


# In[6]:


le=LabelEncoder()
y=le.fit_transform(y)
y


# In[7]:


xtrain, xtest, ytrain, ytest = train_test_split( 
             x, y, test_size = 0.25, random_state = 0) 


# In[8]:


model = KNeighborsClassifier(n_neighbors = 5,p=2) 
model.fit(xtrain, ytrain)
pred=model.predict(xtest)


# In[9]:


k=[]
a=[]
for i in range(1,10):
    model = KNeighborsClassifier(n_neighbors = i,p=2)
    model.fit(xtrain,ytrain)
    pred=model.predict(xtest)
    cm=confusion_matrix(ytest,pred)
    acc=(cm[0,0]+cm[1,1]+cm[2,2])/np.sum(cm)
    k.append(i)
    a.append(acc)
print(k,a)


# In[10]:


plt.scatter(k,a)
plt.show()

