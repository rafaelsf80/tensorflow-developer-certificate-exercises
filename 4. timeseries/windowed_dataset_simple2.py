""" Sample script to show a windowed dataset for time series
    Inluding batch, shuffle and prefetching
"""
import tensorflow as tf

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)
for x,y in dataset:
  print("X = ", x.numpy())
  print("Y = ", y.numpy())


# X =  [[1 2 3 4] [4 5 6 7]]
# Y =  [[5] [8]]
# X =  [[3 4 5 6] [2 3 4 5]]
# Y =  [[7] [6]]
# X =  [[0 1 2 3][5 6 7 8]]
# Y =  [[4] [9]]




