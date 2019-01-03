# predict_air_pollution_with_lstm
This module created a complete workflow to use LSTM/GRU to predict air pollution based on various kinds of sensory data.


The core code module include four parts
1. data.py:
  a.helps load in data from .csv
  b.preprocess data(normalize numerical data and convert categorical data into one-hot encoding)
  c.create train/val/test generator
  
2. model.py(keras required)
  a.build computational graph
  b.train model
  c.predict
 
3. plot.py(seaborn required)
  a.plot train curve
  b.plot predictions and groundtruth
  
4. main.py
  entrance
