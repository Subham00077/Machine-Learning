import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.preprocessing import LabelEncoder



from google.colab import files
uploaded = files.upload()


df = pd.read_csv("samples_cancer.csv")


df.replace("?", np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors="coerce")
df.dropna(inplace=True)


label_encoders = {}
for col in ['Class']:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    label_encoders[col]=encoder


feature_df=df[df.columns[0:-1]]
X=np.asarray(feature_df)
y=np.asarray(df[df.columns[-1]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


lr = LogisticRegression(solver='liblinear', penalty='l2', C=1.0)
lr.fit(X_train, y_train)


df.head(10)


print("Classes after encoding:", encoder.classes_)

y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", np.round(accuracy * 100, 2), "%")


print("\nClassification Report:\n", classification_report(y_test, y_pred))


features = [[1000025, 5,  1,  1,  1,  2,  1,  3, 1, 1]]
predicted_category = lr.predict(features)


print("Predicted tumor for a patient:", label_encoders["Class"].inverse_transform(predicted_category))


y_train_prob = lr.predict_proba(X_train)
y_test_prob = lr.predict_proba(X_test)


train_log_loss = log_loss(y_train, y_train_prob)
test_log_loss = log_loss(y_test, y_test_prob)
print(f"Training Log Loss: {train_log_loss:.4f}")
print(f"Testing Log Loss: {test_log_loss:.4f}")


import numpy as np
import matplotlib.pyplot as plt
y_train_prob_pos = y_train_prob[:, 1] if y_train_prob.ndim > 1 else y_train_prob
y_test_prob_pos = y_test_prob[:, 1] if y_test_prob.ndim > 1 else y_test_prob


def compute_log_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


train_log_loss = []
test_log_loss = []

for i in range(1, 1001):
    train_log_loss.append(compute_log_loss(y_train, y_train_prob_pos))
    test_log_loss.append(compute_log_loss(y_test, y_test_prob_pos))




epochs = np.arange(1, len(train_log_loss) + 1)
plt.figure(figsize=(7, 5))
plt.plot(epochs, train_log_loss, label="Train Log-Loss", color="blue")
plt.plot(epochs, test_log_loss, label="Test Log-Loss", color="red")
plt.xlabel("Iterations")
plt.ylabel("Log-Loss")
plt.title("Log-Loss Curve for Logistic Regression")
plt.legend()
plt.show()
