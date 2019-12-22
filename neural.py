import numpy
from keras import models
from keras import layers, losses
from numpy import loadtxt
from keras import optimizers
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras import regularizers

from keras import layers
from keras import models

from keras import backend as K
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

metrics = ['accuracy', precision_m, recall_m]

model.compile(loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(lr=1e-4),
            metrics= metrics)

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(r'C:\Users\admin\PycharmProjects\mlKeras\train',
                                                     target_size=(150, 150),# this is resize
                                                     batch_size=16,
                                                     class_mode='binary')
validation_generator = test_datagen.flow_from_directory(r'C:\Users\admin\PycharmProjects\mlKeras\validation',
                                                         target_size=(150, 150),
                                                         batch_size=16,
                                                         class_mode='binary')
test_generator = test_datagen.flow_from_directory(r'C:\Users\admin\PycharmProjects\mlKeras\test',
                                                  target_size=(150, 150),
                                                  batch_size=16,
                                                  class_mode='binary')
history = model.fit_generator(train_generator,
                              steps_per_epoch=10,
                              epochs=10,
                              validation_data = validation_generator,
                              validation_steps = 4,
                              )
print('\n----------------------------------------------MODEL 1----------------------------------------------\n')
print('\ntrain results:\n')
loss, accuracy, precision, recall = model.evaluate_generator(train_generator, verbose=0, steps = len(train_generator))
print("loss={:.2f}, accuracy={:.2f}, precision={:.2f}, recall={:.2f}".format(loss, accuracy, precision, recall))
print('\ntest results:\n')
loss, accuracy, precision, recall = model.evaluate_generator(test_generator, verbose=0, steps = len(test_generator))
print("loss={:.2f}, accuracy={:.2f}, precision={:.2f}, recall={:.2f}".format(loss, accuracy, precision, recall))

model2 = models.Sequential()
model2.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(128, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model2.add(layers.MaxPooling2D((2, 2)))
model2.add(layers.Flatten())
model2.add(layers.Dense(512, activation='relu'))
model2.add(layers.Dense(32, activation='relu'))
model2.add(layers.Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy',
                optimizer=optimizers.RMSprop(lr=1e-4),
                metrics= metrics)
history = model2.fit_generator(train_generator,
                              steps_per_epoch=10,
                              epochs=10,
                              validation_data = validation_generator,
                              validation_steps = 4,
                              )
print('\n----------------------------------------------MODEL 2----------------------------------------------\n')
print('\ntrain results:\n')
loss, accuracy, precision, recall = model2.evaluate_generator(train_generator, verbose=0, steps = len(train_generator))
print("loss={:.2f}, accuracy={:.2f}, precision={:.2f}, recall={:.2f}".format(loss, accuracy, precision, recall))
print('\ntest results:\n')
loss, accuracy, precision, recall = model2.evaluate_generator(test_generator, verbose=0, steps = len(test_generator))
print("loss={:.2f}, accuracy={:.2f}, precision={:.2f}, recall={:.2f}".format(loss, accuracy, precision, recall))

model3 = models.Sequential()
model3.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model3.add(layers.MaxPooling2D((2, 2)))
model3.add(layers.Conv2D(64, (3, 3), activation='relu'))
model3.add(layers.MaxPooling2D((2, 2)))
model3.add(layers.Conv2D(64, (3, 3), activation='relu'))
model3.add(layers.MaxPooling2D((2, 2)))
model3.add(layers.Flatten())
model3.add(layers.Dropout(0.4))
model3.add(layers.Dense(512, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
model3.add(layers.Dense(32, activation='relu'))
model3.add(layers.Dense(1, activation='sigmoid'))
model3.compile(loss='binary_crossentropy',
                optimizer=optimizers.RMSprop(lr=1e-4),
                metrics= metrics)
history = model3.fit_generator(train_generator,
                              steps_per_epoch=10,
                              epochs=10,
                              validation_data = validation_generator,
                              validation_steps = 4,
                              )
print('\n----------------------------------------------MODEL 3----------------------------------------------\n')
print('\ntrain results:\n')
loss, accuracy, precision, recall = model3.evaluate_generator(train_generator, verbose=0, steps = len(train_generator))
print("loss={:.2f}, accuracy={:.2f}, precision={:.2f}, recall={:.2f}".format(loss, accuracy, precision, recall))
print('\ntest results:\n')
loss, accuracy, precision, recall = model3.evaluate_generator(test_generator, verbose=0, steps = len(test_generator))
print("loss={:.2f}, accuracy={:.2f}, precision={:.2f}, recall={:.2f}".format(loss, accuracy, precision, recall))