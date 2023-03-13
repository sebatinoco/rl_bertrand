import numpy as np

def get_rolling(series, window_size):

  rolling_mean = np.convolve(series, np.ones(window_size)/window_size, mode = 'valid')

  fill = np.full([window_size - 1], np.nan)
  rolling_mean = np.concatenate((fill, rolling_mean))

  return rolling_mean