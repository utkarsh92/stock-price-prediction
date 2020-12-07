import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout
import os
global df
from math import sqrt
from sklearn.metrics import mean_squared_error

df = pd.read_csv('GOOG_total.csv')
df.head()

df['Date'] = pd.to_datetime(df['Date'])

def get_dataset(train_year_start=2005,testmonths=1):
    global df
    train_start_date = str(train_year_start) + '-01-01'
    mask = (df['Date'] >= train_start_date)
    df = df[mask]
    # print('total = ',len(df.index))
    test_count =  testmonths * 21 # 252/12 = 21
    train_count = len(df.index) - test_count
    train_set = df.head(train_count)
    test_set = df.tail(test_count)

    # print(len(train_set.index),len(test_set.index), len(train_set.index)+len(test_set.index))
    
    return train_set, test_set

def split(train_data, test_data):
    x_train = np.array(train_data[['Close', 'High', 'Low', 'Volume']])
    y_train = np.array(train_data['Close'])
    x_test = np.array(test_data[['Close', 'High', 'Low', 'Volume']])
    y_test = np.array(test_data['Close'])
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    
        
    return x_train, y_train, x_test, y_test


test_list = [1]

for x in test_list:

    print("\n-------",x,"--------\n")

    train_loss = []
    test_loss = []
    
    for i in range(2005, 2019, 1):
        print(i , "Year")
        scaler = MinMaxScaler(feature_range=(0,1))
        train_data, test_data =  get_dataset(i, 1)
        #test_data =  get_dataset(2019, 12)
        x_train, y_train, x_test, y_test = split(train_data, test_data)
        #scaler(x_train, y_train, x_test, y_test)
        
        #Feature Scaling
        scaler = MinMaxScaler(feature_range=(0,1))
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.fit_transform(x_test)
        
        #print(x_train)
        model = Sequential()
        model.add(Dense(units=96, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=96, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=96, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=96, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        history=model.fit(x_train, y_train, epochs=50, batch_size=64)
    
    
        y_hat = model.predict(x_train)
        
        #y_hat = scaler.inverse_transform(y_hat)
        #y_train = scaler.inverse_transform(y_train)
        #RMSE
        loss = sqrt(mean_squared_error(y_train, y_hat))
        train_loss.append([i, loss])
        print("train:", loss)
    
        y_hat = model.predict(x_test)
        #y_hat = scaler.inverse_transform(y_hat)
        #y_test = scaler.inverse_transform(y_test)
        loss = sqrt(mean_squared_error(y_test, y_hat))
        test_loss.append([i, loss])
        print("test:", loss)

    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    
    plt.figure(figsize=(16, 8))
    plt.title('Train RMSE vs Test RMSE\nEpochs = 50, Batch Size = 64\nRMSE loss plot for 1 Month Test data ')
    plt.xlabel('Year')
    plt.ylabel('Loss')
    plt.plot(train_loss[:,0], train_loss[:,1])
    plt.plot(test_loss[:,0], test_loss[:,1])
    plt.legend(['train_rmse', 'test_rmse'])
    # plt.show()
    plt.savefig('/home/dharmendra/Desktop/FML Projects/' + str(x) + '.png')

