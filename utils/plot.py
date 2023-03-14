import numpy as np
import matplotlib.pyplot as plt
from utils.get_rolling import get_rolling

def plot(env, plots_dir, exp_name, window_size = 1000):

    history = np.array(env.history)
    plt.figure(figsize = (12, 6))
    for agent in range(history.shape[1]): # for each agent
        series = history[:, agent] # actions of agent i
        series = np.array([env._action_to_price[agent][action] for action in series]) # action to price
        rolling_mean = get_rolling(series, window_size) # moving average
        
        plt.plot(range(history.shape[0]), rolling_mean, label = f'Agent {agent}') # plot curve
        
    plt.axhline(y = env.monopoly_price, color = 'r', linestyle = '--', label = 'Monopoly Price') # monopoly price
    plt.axhline(y = env.nash_price, color = 'g', linestyle = '--', label = 'Nash Price') # nash price
    plt.xlabel('Steps') # x axis label
    plt.ylabel('Price') # y axis label
    plt.title('Solving Bertrand Environment with Multi Agent Q-learning') # title
    plt.legend() # plot
        
    plt.savefig(f'{plots_dir}/{exp_name}.png', transparent = False, bbox_inches = 'tight') # save figure