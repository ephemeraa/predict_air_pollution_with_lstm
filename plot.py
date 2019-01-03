import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_history(history):
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	epochs = range(len(loss))
	plt.figure()
	plt.plot(epochs, loss, 'bo', label='Training loss')
	plt.plot(epochs, val_loss, 'b', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	sns.set(style="darkgrid")
	plt.savefig("train_curve.jpg",dpi=300)
	plt.show()


def plot_single(pred_y, test_y):
	time_steps = np.arange(test_y.size)

	dataset_pred = pd.DataFrame({
    "pred_y": test_y,
    "exp_idx": -np.ones(pred_y.shape),
    "time_steps": time_steps,
    "conditions": ["groundtruth"]*test_y.size
	})
	dataset_grnd = pd.DataFrame({
	    "pred_y": pred_y,
	    "exp_idx": np.zeros(pred_y.shape),
	    "time_steps": time_steps,
	    "conditions": ["predictions"]*test_y.size
	})


	dataset_single = pd.concat((dataset_pred, dataset_grnd))
	colors = {"groundtruth":"b", "predictions":"g"}
	sns.tsplot(data=dataset_single, 
	           time="time_steps", 
	           value="pred_y", 
	           unit="exp_idx", 
	           condition="conditions",
	           color=colors)
	plt.legend(loc='best').draggable()
	plt.savefig("single_re.jpg",dpi=300)
	plt.show()