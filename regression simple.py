import pandas as pd
import matplotlib as plt
import numpy as np 
from pandas import read_csv
data = read_csv('C:\LoyersMaisons.csv',delimiter=";")
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import classification_report
data
data.head()
data.info()
sns.countplot(x="surface",data=data)
sns.countplot(x="loyer",data=data)
sns.displot(data["surface"])
data.plot(kind="scatter", x="loyer", y="surface" , color='red')
loyer = data[data['loyer']>10000]
print(loyer)
loyer.plot(kind="scatter", x="surface", y="loyer" , color='red')
x=data.iloc[: ,:-1].values
y=data.iloc[: ,-1].values
x
y
x_tarin,x_test,y_tarin,y_test=train_test_split(x,y,test_size=0.2)
x_tarin
x_test
confusion_matrix(y_test,y_pred)
calassifier.predict([[5,110,75,35,0,34,0.125,70]])
calassifier.predict([[5,110,75,35,0,34,0.125,70]])
plt.plot(test_scores, 'data')
plt.title('Evolution des loyer par surface')
plt.xlabel('loyer')
plt.ylabel('surface')
plt.show()