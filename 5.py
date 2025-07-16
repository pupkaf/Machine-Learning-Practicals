import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

x1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y2 = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
X = np.concatenate((x1, x2))
y = np.concatenate((y1, y2))
model = keras.Sequential([
    layers.Input(shape=(1,)),  
    layers.Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=1000, verbose=0)
predictions = model.predict(X)
plt.figure(figsize=(10, 6))
plt.scatter(x1, y1, color='blue', label='Группа 1')
plt.scatter(x2, y2, color='red', label='Группа 2')
plt.plot(X, predictions, color='green', label='Линейная регрессия', linewidth=2)
plt.title('Линейная регрессия по координатам двух групп точек')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()
