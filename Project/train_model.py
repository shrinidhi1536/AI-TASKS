import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
import numpy as np

# Load dataset
(x_train, y_train), _ = cifar10.load_data()

# Animal classes only
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
animal_classes = ['bird','cat','deer','dog','frog','horse']
animal_ids = [class_names.index(a) for a in animal_classes]

mask = np.isin(y_train.flatten(), animal_ids)
x_train = x_train[mask] / 255.0
y_train = y_train[mask]

label_map = {animal_ids[i]: i for i in range(len(animal_ids))}
y_train = np.array([label_map[int(y)] for y in y_train])

# Model
model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64,activation='relu'),
    Dense(6,activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

# Save model
model.save("model.h5")
print("✅ model.h5 saved")