import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

data = pd.read_csv('BankNote_Authentication.csv')
data.head()

#Independent and dependent features
X=data.iloc[:,:-1]
y=data.iloc[:,-1]

X.head()

#Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

#Implement random forest classifier
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)

#Prediction
y_pred=classifier.predict(X_test)

#Checking Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)
score

pickle_out = open("BankNote.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()

classifier.predict([[2,3,4,1]])