import numpy as np
import matplotlib.pyplot as plt
from utils.get_rolling import get_rolling

def plot(env, plots_dir, exp_name, window_size = 100):

    history = np.array(env.history)
    plt.figure(figsize = (12, 6))
    for agent in range(history.shape[1]):
        series = history[:, agent] # acciones del agente i
        series = np.array([env._action_to_price[agent][action] for action in series]) # acciones a precios
        rolling_mean = get_rolling(series, window_size) # media móvil
        
        plt.plot(range(history.shape[0]), rolling_mean, label = f'Agent {agent}') # plot
        plt.xlabel('Steps')
        plt.ylabel('Price')
        plt.title('Solving Bertrand Environment with Multi Agent Q-learning')
        plt.legend()
        
        plt.savefig(f'{plots_dir}/{exp_name}.png', transparent = False, bbox_inches = 'tight')