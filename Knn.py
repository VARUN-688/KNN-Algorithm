from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Users\varun\OneDrive\Desktop\workshop\fishiris.csv")
x=df.loc[:,['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y=df.loc[:,'Name']

l=LabelEncoder()
y=l.fit_transform(y)

knn=KNeighborsClassifier(n_neighbors=5,p=2)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)

knn.fit(xtrain, ytrain)
pred=knn.predict(xtest)
print("the accuracy score = %.2f"%(accuracy_score(ytest, pred)*100))
