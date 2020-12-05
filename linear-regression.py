from create_dataset import get_dataset
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('bmh')


def get_sets(start_year, test_months):
    df_train, df_test = get_dataset(start_year, test_months)

    # print(df_train.shape)

    x_train = np.array(df_train[['Open', 'High', 'Low', 'Volume']])
    y_train = np.array(df_train['Close'])
    x_test = np.array(df_test[['Open', 'High', 'Low', 'Volume']])
    y_test = np.array(df_test['Close'])
    dates = np.array(df_test['Date'].dt.date)
    # print(type(df_test['Date']))
    # print(dates[0])

    return x_train, y_train, x_test, y_test, dates

test_list = [1, 6, 12]

for x in test_list:

    print("\n-------",x,"--------\n")

    train_loss = []
    test_loss = []

    # print(len(train_loss), len(test_loss))

    for i in range(2005, 2019, 1):
        # print(i)
        x_train, y_train, x_test, y_test, dates = get_sets(i, x)

        # print(x_train.shape)

        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        lreg = LinearRegression()
        lreg.fit(x_train, y_train)

        #train
        y_hat = lreg.predict(x_train)
        loss = mean_squared_error(y_train, y_hat)
        train_loss.append([i, loss])
        # print("train:", loss)

        #test
        y_hat = lreg.predict(x_test)
        loss = mean_squared_error(y_test, y_hat)
        test_loss.append([i, loss])
        # print("test:", loss)

        # del lreg, x_train, y_train, x_test, y_test 

    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    # print('Train:\n', train_loss)
    # print('Test:\n', test_loss)


    # print('Diff:\n', test_loss[:,1] / train_loss[:, 1])
    # exit()

    plt.figure(figsize=(16, 8))
    plt.title('MSE loss plot ' + str(x) + ' months')
    plt.xlabel('Year')
    plt.ylabel('Loss')
    # plt.plot(train_loss[:,0], train_loss[:,1], color='blue')
    plt.plot(test_loss[:,0], test_loss[:,1], color='red')
    # plt.legend(['train_mse', 'test_mse'])
    plt.legend(['test_mse'])
    # plt.show()
    plt.savefig('/mnt/c/Users/utin9/Desktop/test_stand/' + str(x) + '_.png')
    del train_loss, test_loss