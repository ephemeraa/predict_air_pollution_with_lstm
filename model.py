from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop


def build_model():
	model = Sequential()
	model.add(layers.GRU(32,
						 dropout=0.1,
						 recurrent_dropout=0.5,
						 input_shape=(None, float_data.shape[-1])))
	model.add(layers.Dense(1))
	model.compile(optimizer=RMSprop(), loss='mae')
	return model


def train_model(model, train_gen, val_gen, val_steps)
	history = model.fit_generator(train_gen,
								  steps_per_epoch=500,
								  epochs=40,
								  validation_data=val_gen,
								  validation_steps=val_steps)
	return model, history


def predict(model, x_test):
	return model.predict(x_test)
