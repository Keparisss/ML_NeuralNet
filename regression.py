import numpy
from keras import models, Sequential
from keras import layers, losses
from keras.layers import Dense
from numpy import loadtxt
from keras import optimizers
import pandas as pd
# import numpy as np
from sklearn.preprocessing import StandardScaler


def build_model():
  model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(data.keys())-1]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

data = pd.read_csv("export_data_regressor.csv")

model = build_model()

from sklearn.model_selection import train_test_split

X = data.iloc[:, :-1].values
y = data['APM'].values
from sklearn.preprocessing import MinMaxScaler
MMS = MinMaxScaler()
X = MMS.fit_transform(X)
y = y.reshape(-1, 1)
y = MMS.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# Compile the network :
# model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
model.fit(X_train, y_train, epochs = 3)
y_train_pred = model.predict(X_train)
y_test_pred= model.predict(X_test)


from sklearn.metrics import mean_squared_error
print('MSE тренировка: %.3f, тестирование: %.3f' % (
mean_squared_error(y_train, y_train_pred),
mean_squared_error(y_test, y_test_pred)))

from sklearn.metrics import r2_score
print('R2 тренировка: %.3f, тестирование: %.3f' %(
                        r2_score(y_train, y_train_pred),
                        r2_score(y_test, y_test_pred)))


print('\nМодель с 3 скрытыми слоями')

NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(128, kernel_initializer='normal', input_shape=[len(data.keys())-1], activation='relu'))
# The Hidden Layers :
NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
NN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.fit(X_train, y_train, epochs = 3)
y_train_pred = NN_model.predict(X_train)
y_test_pred= NN_model.predict(X_test)

print('MSE тренировка: %.3f, тестирование: %.3f' % (
mean_squared_error(y_train, y_train_pred),
mean_squared_error(y_test, y_test_pred)))

print('R2 тренировка: %.3f, тестирование: %.3f' %(
                        r2_score(y_train, y_train_pred),
                        r2_score(y_test, y_test_pred)))
