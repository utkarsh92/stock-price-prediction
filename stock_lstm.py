from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from create_dataset import get_dataset
from keras.callbacks import EarlyStopping

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=15):
	X, Y = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		X.append(a)
		Y.append(dataset[i + look_back, 0])
	return np.array(X), np.array(Y)

test_months = [12,1]
fig,ax = plt.subplots(len(test_months))
# look_back 
# test_months = [12]
graph_j = 0

for m in test_months:
	start_year = []
	train_mse = []
	test_mse = []
	for i in range(2005,2018,1):
		print(i,m)
		df_train, df_test = get_dataset(i,m)

		dataset_train = np.array(df_train['Close'])
		dataset_test = np.array(df_test['Close'])

		# scaler = MinMaxScaler(feature_range=(0,1))
		# dataset_train = scaler.fit_transform(df_train['Close'].values.reshape(-1, 1))
		# dataset_test = scaler.fit_transform(df_test['Close'].values.reshape(-1, 1))

		dataset_train = df_train['Close'].values.reshape(-1, 1)
		dataset_test = df_test['Close'].values.reshape(-1, 1)

		print(dataset_train.shape)
		print(dataset_test.shape)

		look_back = 10

		x_train, y_train = create_dataset(dataset_train, look_back = look_back)
		x_test, y_test = create_dataset(dataset_test, look_back = look_back)

		x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
		x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))

		print(x_train.shape,y_train.shape)
		print(x_test.shape,y_test.shape)

		#define model
		regressor = Sequential()
		regressor.add(LSTM(units = 80,activation='relu',input_shape = (1, look_back)))
		
		regressor.add(Dense(units = 1))

		#train
		callback = EarlyStopping(monitor='val_loss', patience=4,restore_best_weights=True)
		regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
		# regressor.fit(x_train,y_train, epochs=50, batch_size = 4,validation_split=0.25,shuffle=False)
		regressor.fit(x_train,y_train, epochs=100, batch_size = 4, callbacks=[callback],validation_split=0.25,shuffle=False)

		#predict
		train_predict = regressor.predict(x_train)
		test_predict = regressor.predict(x_test)

		# train_predict = scaler.inverse_transform(train_predict)
		# y_train = scaler.inverse_transform([y_train]).T
		# test_predict = scaler.inverse_transform(test_predict)
		# y_test = scaler.inverse_transform([y_test]).T

		# calculate loss
		train_mse_loss = mean_squared_error(y_train,train_predict)
		print("train error : ",train_mse_loss)
		test_mse_loss = mean_squared_error(y_test,test_predict)
		print("test error : ",test_mse_loss)
		start_year.append(i)
		test_mse.append(test_mse_loss)
		train_mse.append(train_mse_loss)

	ax[graph_j].plot(start_year,train_mse,color = 'green',label='train_mse')
	ax[graph_j].plot(start_year,test_mse,color = 'red',label='test_mse')
	ax[graph_j].set_title('Test Months = ' + str(m))
	ax[graph_j].set_xlabel('Training Starting Year')
	ax[graph_j].set_ylabel('Loss')

	graph_j+=1

plt.show()