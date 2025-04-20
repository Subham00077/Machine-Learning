import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy


iris = pd.read_csv('iris.csv', header=None)
iris.head()



iris = load_iris()
X = iris.data
y = iris.target




lb = LabelBinarizer()
y = lb.fit_transform(y)




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




def build_model(activation='relu', loss_fn='mse', optimizer='adam', learning_rate=0.01):
    model = Sequential()
    model.add(Dense(64, input_dim=4, activation=activation))
    for _ in range(4):  
        model.add(Dense(64, activation=activation))
    model.add(Dense(3, activation='softmax'))  

    


    if optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    else:
        opt = SGD(learning_rate=learning_rate)  

    # Choose loss function
    if loss_fn == 'mse':
        loss = MeanSquaredError()
    elif loss_fn == 'crossentropy':
        loss = CategoricalCrossentropy()
    else:
        loss = MeanSquaredError()
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    return model




activation = 'relu'           
loss_fn = 'crossentropy'        
optimizer = 'adam'               
learning_rate = 0.01
epochs = 300



model = build_model(activation, loss_fn, optimizer, learning_rate)
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), verbose=0)




plt.figure(figsize=(12, 5))



plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()




plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()




loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Final Test Accuracy: {accuracy*100:.2f}% | Loss: {loss:.4f}")



