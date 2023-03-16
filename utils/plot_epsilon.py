import numpy as np
import matplotlib.pyplot as plt
from utils.get_rolling import get_rolling

def plot_epsilon(epsilon_list, env, plots_dir, env_name, exp_name, window_size = 1000):

    '''
    Plot the curves of the experiment using a fixed window_size.
    '''

    history = np.array(env.history)
    fig, ax = plt.subplots(figsize = (12, 6))
    for agent in range(history.shape[1]):
      series = history[:, agent] # acciones del agente i
      series = np.array([env._action_to_price[agent][action] for action in series]) # acciones a precios
      rolling_mean = get_rolling(series, window_size) # media móvil
      ax.plot(range(history.shape[0]), rolling_mean, label = f'Agent {agent}') # plot
    ax.axhline(y = env.monopoly_price, color = 'r', linestyle = '--', label = 'Monopoly Price')
    ax.axhline(y = env.nash_price, color = 'g', linestyle = '--', label = 'Nash Price')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Price')

    ax2 = ax.twinx()
    ax2.plot(range(len(epsilon_list)), epsilon_list, color = 'tab:purple', label = 'Epsilon curve')
    ax2.set_ylabel('Epsilon')

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2)

    plt.savefig(f'{plots_dir}/{env_name}/{exp_name}_epsilon.png', transparent = False, bbox_inches = 'tight') # save figure