import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam




X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0],
              [1],
              [1],
              [0]])



model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu')) 
model.add(Dense(1, activation='sigmoid')) 



model.compile(optimizer=Adam(learning_rate=0.1), loss='binary_crossentropy', metrics=['accuracy'])




model.fit(X, y, epochs=500, verbose=0)


predictions = model.predict(X).round()
print("\n🧠 XOR Predictions:")
for i in range(len(X)):
    print(f"Input: {X[i]} -> Predicted Output: {int(predictions[i][0])}")


