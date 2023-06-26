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

def plot_metrics(fig, axes, prices_history, monopoly_history, nash_history, rewards_history,
                 rolling = 1000, actor_loss = None, Q_loss = None):
  
    prices = np.array(prices_history)
    rewards = np.array(rewards_history)  
    [ax.cla() for row in axes for ax in row]
        
    for agent in range(prices.shape[1]):
      rolling_price = get_rolling(prices[:, agent], rolling)
      axes[0, 0].plot(range(rolling_price.shape[0]), rolling_price, label = f'Agent {agent}') # plot rolling avg price
      
      rolling_mean = get_rolling(rewards[:, agent], rolling)
      axes[0, 1].plot(range(rolling_mean.shape[0]), rolling_mean, label = f'Agent {agent}') # plot rolling avg reward

    axes[0, 0].plot(monopoly_history, label = 'Monopoly Price', linestyle = '--')
    axes[0, 0].plot(nash_history, label = 'Nash Price', linestyle = '--')
    axes[0, 0].set_title(f'Rolling Avg of Prices (window = {rolling})')
    axes[0, 0].set_xlabel('Timesteps')

    axes[0, 1].set_title(f'Rolling Avg of Rewards (window = {rolling})')
    axes[0, 1].set_xlabel('Timesteps')
    
    if (actor_loss is not None) & (Q_loss is not None):
      
      # graficar loss por cada agente!!

      axes[1, 0].plot(actor_loss, label = 'Actor loss')
      axes[1, 0].set_title('Actor Q(s,a) (Agent 0)')
      axes[1, 0].set_xlabel('Update iteration')

      axes[1, 1].plot(Q_loss, label = 'Critic Loss')
      axes[1, 1].set_title('Critic MSE Loss (Agent 0)')
      axes[1, 1].set_xlabel('Update iteration')
    
    [ax.grid('on') for row in axes for ax in row]
    [ax.legend() for row in axes for ax in row]
    
    fig.tight_layout()
    plt.pause(0.05)
