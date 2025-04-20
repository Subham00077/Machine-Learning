
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    jaccard_score, confusion_matrix, roc_curve, auc
)



df = pd.read_csv('samples_cancer.csv')
print(df.head())




df['BareNuc'] = pd.to_numeric(df['BareNuc'], errors='coerce')



df.dropna(inplace=True)



df.drop(columns=['ID'], inplace=True)



df['Class'] = df['Class'].apply(lambda x: 1 if x == 4 else 0)



X = df.drop('Class', axis=1)
y = df['Class']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=4)
import seaborn as sns




kernels = ['linear', 'poly', 'rbf', 'sigmoid']
results = {}
fpr_dict, tpr_dict = {}, {}

for kernel in kernels:
    model = SVC(kernel=kernel, probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    jaccard = jaccard_score(y_test, y_pred)
    error = 1 - acc
    cm = confusion_matrix(y_test, y_pred)

    results[kernel] = {
        'Accuracy': acc,
        'Recall': rec,
        'Precision': prec,
        'F1-Score': f1,
        'Jaccard Score': jaccard,
        'Error Rate': error,
        'Confusion Matrix': cm
    }



    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fpr_dict[kernel] = fpr
    tpr_dict[kernel] = tpr

import seaborn as sns
import matplotlib.pyplot as plt





for kernel in results:
    print(f"\n📌 Kernel: {kernel.upper()}")
    for metric, value in results[kernel].items():
        if metric != "Confusion Matrix":
            print(f"{metric}: {value:.4f}")




    plt.figure(figsize=(5, 4))
    sns.heatmap(results[kernel]["Confusion Matrix"], annot=True, fmt='d', cmap="Blues")
    plt.title(f"{kernel.upper()} Kernel - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()





plt.figure(figsize=(10, 6))
for kernel in kernels:
    plt.plot(fpr_dict[kernel], tpr_dict[kernel], label=f'{kernel} (AUC = {auc(fpr_dict[kernel], tpr_dict[kernel]):.2f})')



plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison for SVM Kernels')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

