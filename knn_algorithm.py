import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import roc_curve



from google.colab import files
uploaded = files.upload()


data=pd.read_csv('teleCust.csv')


raw=data.copy()

data.shape

data.head()


data.describe()


x=np.asarray(data[data.columns[0:-1]])
y=np.asarray(data[data.columns[-1]])




x0=preprocessing.StandardScaler().fit(x)
x1=x0.transform(x)
x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.20,random_state=10)




knn = KNeighborsClassifier(n_neighbors=41,metric='minkowski')
y_knn = knn.fit(x_train,y_train)
y_knn_pred = y_knn.predict(x_test)



knn_acc=accuracy_score(y_test,y_knn_pred)
knn_f1=f1_score(y_test,y_knn_pred,average='micro')
print("accuracy score ", knn_acc*100, "%")
print("f1_score ", knn_f1*100, "%")



knn_cr=classification_report(y_test,y_knn_pred)
print('confusion matrix\n' ,knn_cm)
print('classification report\n', knn_cr)


