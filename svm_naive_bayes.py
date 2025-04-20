import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, roc_curve, auc
)



from google.colab import files
uploaded = files.upload()




df = pd.read_csv('pima-indians-diabetes.data.csv', header=None)
df.head()




X = df.iloc[:, :-1]
y = df.iloc[:, -1]




scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


models = {
    "SVM": SVC(probability=True),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier()
}


metrics = {}




for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    metrics[name] = {
        "Accuracy": acc,
        "Recall": rec,
        "Precision": prec,
        "F1 Score": f1,
        "Confusion Matrix": cm,
        "y_prob": y_prob
    }




for name in metrics:
    print(f"\n🔍 {name}")
    print(f"Accuracy: {metrics[name]['Accuracy']:.4f}")
    print(f"Recall: {metrics[name]['Recall']:.4f}")
    print(f"Precision: {metrics[name]['Precision']:.4f}")
    print(f"F1 Score: {metrics[name]['F1 Score']:.4f}")





plt.figure(figsize=(14, 10))
for i, name in enumerate(metrics):
    plt.subplot(2, 2, i+1)
    sns.heatmap(metrics[name]["Confusion Matrix"], annot=True, fmt='d', cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
plt.tight_layout()
plt.show()






plt.figure(figsize=(10, 6))
for name in models:
    fpr, tpr, _ = roc_curve(y_test, metrics[name]["y_prob"])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')


    

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()


