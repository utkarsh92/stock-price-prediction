import numpy as np
import pandas as pd



def get_dataset(train_year_start=2005,testmonths=12):
	
	df = pd.read_csv('GOOG_total.csv', skipinitialspace=True)
	df['Date'] = pd.to_datetime(df['Date'])

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

# to check
# get_dataset(2006,1)
