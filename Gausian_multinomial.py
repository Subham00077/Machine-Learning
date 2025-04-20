import numpy as np
import pandas as pd
import matplotlib as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


from google.colab import files
uploaded = files.upload()


headers = ['preg', 'plas', 'pres', 'skin', 'insu', 'bmi', 'pedi', 'age', 'class']
data = pd.read_csv('pima-indians-diabetes.data.csv', names=headers)
data.head()


data.shape

feature_df=data[data.columns[0:-1]]
x=np.asarray(feature_df)
y=np.asarray(data[data.columns[-1]])


x1 = preprocessing.normalize(x, axis=0)


x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.2,random_state=4)


pd.DataFrame(data=x1, columns=headers[:-1]).hist()


x_train.shape

gnb = GaussianNB()
y_pred = gnb.fit(x_train, y_train).predict(x_test)


accuracy_score(y_test, y_pred)*100

f1_score(y_test, y_pred)*100

mnb = MultinomialNB()
y_pred_mnb = mnb.fit(x_train, y_train).predict(x_test)


accuracy_score(y_test, y_pred_mnb)*100


f1_score(y_test,y_pred)*100


dtree=DecisionTreeClassifier(criterion='entropy',max_depth=2,min_samples_split=4,ccp_alpha=0.05)


