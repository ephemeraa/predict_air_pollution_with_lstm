import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical


def load_data(data_dir="data", filename="PRSA_data_2010.1.1-2014.12.31.csv")
	data_path = os.path.join(data_dir,filename)
	dataset = pd.read_csv(data_path)
	values  = dataset.values
	return values


def preprocess(values):
	#pollution, dew, temperature,pressure, wind_speed, snow, and rain, continuous variables
	numerical_values		= np.concatenate((values[:,1:5], values[:,6:]), axis=1).astype(np.float32)
	#wind direction, discrete variable
	categorical_values	  = values[:,5]

	#preprocess:
	#1. normalize continuous numerical values to have zero mean and unit std
	numerical_values -= np.mean(numerical_values, axis=0)
	numerical_values /= np.std(numerical_values, axis=0)

	#2. encode dicrete wind direction variables into one-hot encoding
	wnd_dir_items    = np.unique(categorical_values)
	wnd_dir_str2num	 = zip(wnd_dir_items, np.arange(wnd_dir_items.size))
	for s, i in wnd_dir_str2num:
		categorical_values[categorical_values==s] = i
	categorical_values = categorical_values.astype(np.int32)
	categorical_values = to_categorical(categorical_values)

	#3. concatentate variables together
	float_data = np.concatenate((numerical_values, categorical_values), axis=1)
	return float_data


def generator(data, lookback, delay, min_index, max_index,
			  shuffle=False, 
			  batch_size=128, 
			  step=1):
	
	if max_index is None:
		max_index = len(data) - delay - 1
	i = min_index + lookback
	
	while 1:
		if shuffle:
			rows = np.random.randint(
				min_index + lookback, max_index, size=batch_size)
		else:
			if i + batch_size >= max_index:
				i = min_index + lookback
			rows = np.arange(i, min(i + batch_size, max_index))
			i += len(rows)

		samples = np.zeros((len(rows),
						   lookback // step,
						   data.shape[-1]))
		targets = np.zeros((len(rows),))
		for j, row in enumerate(rows):
			indices = range(rows[j] - lookback, rows[j], step)
			samples[j] = data[indices]
			targets[j] = data[rows[j] + delay][1]
		yield samples, targets


def build_generator(float_data, lookback, delay, step, batch_size)
	train_gen = generator(float_data,
						  lookback=lookback,
						  delay=delay,
						  min_index=0,
						  max_index=20000,
						  shuffle=True,
						  step=step, 
						  batch_size=batch_size)
	val_gen = generator(float_data,
						lookback=lookback,
						delay=delay,
						min_index=20001,
						max_index=30000,
						step=step,
						batch_size=batch_size)
	test_gen = generator(float_data,
						 lookback=lookback,
						 delay=delay,
						 min_index=30001,
						 max_index=None,
						 step=step,
						 batch_size=batch_size)
	val_steps = (30000 - 20001 - lookback) // batch_size
	test_steps = (len(float_data) - 30001 - lookback) // batch_size	
	return (train_gen, val_gen, test_gent),(val_steps, test_steps)