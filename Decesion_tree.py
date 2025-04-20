import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


from google.colab import files
uploaded = files.upload()


df=pd.read_csv('drug.csv')



le_age=LabelEncoder()
le_sex = LabelEncoder()
le_bp = LabelEncoder()
le_chol = LabelEncoder()
le_drug = LabelEncoder()
df['Age']=le_age.fit_transform(df['Age'])
df['Sex'] = le_sex.fit_transform(df['Sex'])
df['BP'] = le_bp.fit_transform(df['BP'])
df['Cholesterol'] = le_chol.fit_transform(df['Cholesterol'])
df['Drug'] = le_drug.fit_transform(df['Drug'])


X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']]
y = df['Drug']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


dtree=DecisionTreeClassifier(criterion='entropy', max_depth=None)
dtree_y_pred = dtree.fit(X_train, y_train).predict(X_test)



dtree_acc = accuracy_score(y_test, dtree_y_pred)
print(f'Accuracy of Decision Tree Classifier: {dtree_acc * 100:.2f}%')




