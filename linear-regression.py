import pandas as pd
import numpy as np
# from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('bmh')

#load data from csv
df = pd.read_csv('GOOG.csv', skipinitialspace=True)
df = df.tail(504)
print(df)

#visualize data
# plt.figure(figsize=(16,8))
# plt.title('google')
# plt.xlabel('days')
# plt.ylabel('close price')
# plt.plot(df['Close'])
# plt.show()

test_days = 25
x = np.array(df[['Open', 'High', 'Low', 'Volume']])[:-test_days]
y = np.array(df['Close'])[:-test_days]

# x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.2)
x_train = x
y_train = y
x_test = np.array(df[['Open', 'High', 'Low', 'Volume']].tail(test_days))
y_test = np.array(df['Close'].tail(test_days))
dates = np.array(df['Date'].tail(test_days))

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# scalar = MinMaxScaler().fit(x_train)
# lreg = LinearRegression().fit(scalar.transform(x_train), y_train)
# y_hat = lreg.predict(scalar.transform(x_test))


# lreg = Ridge(alpha=.0001).fit(scalar.transform(x_train), y_train)
# y_hat = lreg.predict(scalar.transform(x_test))

lreg = LinearRegression().fit(x_train, y_train)
y_hat = lreg.predict(x_test)

print("mse loss: ", np.mean(np.power(y_hat - y_test, 2)))
# exit()

plt.figure(figsize=(16,8))
plt.title('google data for ' + dates[0] + ' to ' + dates[-1])
plt.xlabel('days')
plt.ylabel('close price')
# plt.plot(y_train)
plt.plot(y_test)
plt.plot(y_hat)
plt.legend(['y_test', 'y_hat'])
plt.show()