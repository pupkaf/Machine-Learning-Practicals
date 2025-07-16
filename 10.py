import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import tensorflow as tf 

train = pd.read_csv("sign_mnist_train.csv") 
train_labels = train[["label"]] 
train_images = train.loc[:, train.columns != 'label'] 
test = pd.read_csv("sign_mnist_test.csv") 
test_labels = test[["label"]] 
test_images = test.loc[:, test.columns != 'label'] 
x_train = train_images.values / 255.0  
x_test = test_images.values / 255.0  
X = x_train.reshape(-1, 28, 28, 1) 
x_test = x_test.reshape(-1, 28, 28, 1) 
model = tf.keras.Sequential([ 
    tf.keras.layers.Input(shape=(28, 28, 1)),  
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(128, activation='relu'), 
    tf.keras.layers.Dense(25, activation='softmax')  
]) 
model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
              metrics=['accuracy']) 
history = model.fit(X, train_labels, epochs=10, validation_split=0.2, verbose=0)  
test_loss, test_acc = model.evaluate(x_test, test_labels, verbose=0) 
print('\nTest accuracy:', test_acc)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('График потерь')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('График точности')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
f, ax = plt.subplots(2, 5) 
f.set_size_inches(10, 10) 
k = 0 
for i in range(2): 
    for j in range(5): 
        ax[i, j].imshow(X[k].reshape(28, 28), cmap="gray")   
        k += 1 
plt.tight_layout() 
plt.show() 
