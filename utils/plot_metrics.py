import numpy as np
import matplotlib.pyplot as plt

def get_rolling(series, window_size):
  
  '''
  Returns the rolling average of actions using a fixed window_size.
  '''

  rolling_avg = np.convolve(series, np.ones(window_size)/window_size, mode = 'valid')

  fill = np.full([window_size - 1], np.nan)
  rolling_avg = np.concatenate((fill, rolling_avg))

  return rolling_avg

def plot_metrics(fig, axes, prices_history, monopoly_history, nash_history, rolling = 1000):

    prices = np.array(prices_history)
    
    [ax.cla() for ax in axes]

    for agent in range(prices.shape[1]):
        rolling_mean = get_rolling(prices[:, agent], rolling)
        axes[0].plot(range(rolling_mean.shape[0]), rolling_mean, label = f'Agent {agent}') # plot curve

    axes[0].plot(monopoly_history, label = 'Monopoly Price', linestyle = '--')
    axes[0].plot(nash_history, label = 'Nash Price', linestyle = '--')

    axes[0].set_ylabel('Price')
    axes[0].set_xlabel('Timesteps')
    
    [ax.grid('on') for ax in axes]
    
    plt.legend()
    fig.tight_layout()
    
    plt.pause(0.05)