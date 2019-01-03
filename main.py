import data
import model
import plot


def main():
	ori_data   = data.load_data()
	float_data = data.preprocess(ori_data)

	lookback = 240
	step = 1
	delay = 24
	batch_size = 512

	(train_gen, val_gen, test_gen), (val_steps, test_steps) = data.build_generator(
		float_data=float_data,
		lookback=lookback,
		step=step,
		delay=delay,
		batch_size=batch_size
	)

	m          = model.build_model()
	m, history = model.train_model(m, train_gen, val_gen, val_steps)

	plot.plot_history(history)

	test_x, test_y = next(test_gen)
	pred_y         = model.predict(m, test_x)
	plot.plot_single(test_y, pred_y)