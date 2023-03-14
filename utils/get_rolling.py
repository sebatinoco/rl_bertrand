import numpy as np

def get_rolling(series, window_size):
  
  '''
  Returns the rolling average of actions using a fixed window_size.
  '''

  rolling_avg = np.convolve(series, np.ones(window_size)/window_size, mode = 'valid')

  fill = np.full([window_size - 1], np.nan)
  rolling_avg = np.concatenate((fill, rolling_avg))

  return rolling_avg