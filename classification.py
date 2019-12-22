# import numpy
from keras import models
from keras import layers, losses
# from numpy import loadtxt
from keras import optimizers
import pandas as pd
# import numpy as np
from sklearn.preprocessing import StandardScaler

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
model.add(layers.Dense(8, activation='relu', input_shape=(33,)))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

file = 'Attribute DataSet.xlsx'

data = pd.read_csv("export_data_classifier1.csv")

from sklearn.model_selection import train_test_split

X, y = data.iloc[:,:].values, data.iloc[:, 4].values
# df.iloc [ : ,1].values давал слишком большую ошибку???
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
metrics = ['accuracy', precision_m, recall_m]



# 1
print('\n\nbinary_crossentropy:\n')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)
model.fit(X_train_std, y_train, epochs=5)

print('\ntrain results:\n')
loss, accuracy, precision, recall = model.evaluate(X_train_std, y_train, verbose=0)
print("loss={:.2f}, accuracy={:.2f}, precision={:.2f}, recall={:.2f}".format(loss, accuracy, precision, recall))

print('\ntest results:\n')
loss, accuracy, precision, recall = model.evaluate(X_test_std, y_test, verbose=0)
print("loss={:.2f}, accuracy={:.2f}, precision={:.2f}, recall={:.2f}".format(loss, accuracy, precision, recall))

# 2
print('\n\nRMSprop:\n')
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='mse',
              metrics = metrics)
model.fit(X_train_std, y_train,  batch_size=120, epochs= 10)

print('\ntrain results:\n')
loss, accuracy, precision, recall = model.evaluate(X_train_std, y_train, verbose=0)
print("loss={:.2f}, accuracy={:.2f}, precision={:.2f}, recall={:.2f}".format(loss, accuracy, precision, recall))

print('\ntest results:\n')
loss, accuracy, precision, recall = model.evaluate(X_test_std, y_test, verbose=0)
print("loss={:.2f}, accuracy={:.2f}, precision={:.2f}, recall={:.2f}".format(loss, accuracy, precision, recall))


# cccccccccccccccccccccccccccccccccccccccccc
model = models.Sequential()
model.add(layers.Dense(8, activation='relu', input_shape=(33,)))
model.add(layers.Dense(2, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = metrics)
model.fit(X_train_std, y_train,  batch_size=120, epochs= 20)

print('\ntrain results:\n')
loss, accuracy, precision, recall = model.evaluate(X_train_std, y_train, verbose=0)
print("loss={:.2f}, accuracy={:.2f}, precision={:.2f}, recall={:.2f}".format(loss, accuracy, precision, recall))

print('\ntest results:\n')
loss, accuracy, precision, recall = model.evaluate(X_test_std, y_test, verbose=0)
print("loss={:.2f}, accuracy={:.2f}, precision={:.2f}, recall={:.2f}".format(loss, accuracy, precision, recall))



# #  НАДО ЛИ ТАКОЕ ДЕЛАТЬ?
#
# from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier()
# model.fit(X_train_std, y_train)
#
# print('\ntrain results:\n')
# loss, accuracy, precision, recall = model.evaluate(X_train_std, y_train, verbose=0)
# print("loss={:.2f}, accuracy={:.2f}, precision={:.2f}, recall={:.2f}".format(loss, accuracy, precision, recall))
#
# print('\ntest results:\n')
# loss, accuracy, precision, recall = model.evaluate(X_test_std, y_test, verbose=0)
# print("loss={:.2f}, accuracy={:.2f}, precision={:.2f}, recall={:.2f}".format(loss, accuracy, precision, recall))
# старая оценка
# loss_and_metrics = model.evaluate(X_train_std, y_train, batch_size=128)
# for index, elem in enumerate(metrics):
#     print("%s: %.2f%%" % (model.metrics_names[index], loss_and_metrics[index]*100))
