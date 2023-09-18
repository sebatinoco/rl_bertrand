import pandas as pd
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

def get_rolling_std(series, window_size):
    '''
    Returns the rolling standard deviation using a fixed window_size.
    '''
    rolling_mean = get_rolling(series, window_size)
    squared_diff = (series - rolling_mean) ** 2
    rolling_variance = get_rolling(squared_diff, window_size)
    rolling_std = np.sqrt(rolling_variance)

    return rolling_std

def get_plots(exp_name, window_size = 500):

    ###########################################
    df_plot = pd.read_csv(f'metrics/{exp_name}.csv', sep = ';', encoding = 'utf-8-sig')
    
    actions_cols = [col for col in df_plot.columns if 'actions' in col]
    price_cols = [col for col in df_plot.columns if 'prices' in col]
    rewards_cols = [col for col in df_plot.columns if 'rewards' in col]
    quantities_cols = [col for col in df_plot.columns if 'quantities' in col]

    n_agents = len(actions_cols)

    df_plot['avg_actions'] = df_plot[actions_cols].mean(axis = 1)
    df_plot['avg_prices'] = df_plot[price_cols].mean(axis = 1)
    df_plot['avg_rewards'] = df_plot[rewards_cols].mean(axis = 1)
    df_plot['avg_quantities'] = df_plot[quantities_cols].mean(axis = 1)
    avg_cols = [col for col in df_plot.columns if 'avg' in col]

    window_cols = price_cols + rewards_cols + quantities_cols + avg_cols + ['delta']
    for col in window_cols:
        df_plot[col] = get_rolling(df_plot[col], window_size = window_size)
        
    ############################################
    plt.figure(figsize = (12, 4))
    for agent in range(n_agents):
        price_serie = df_plot[f'prices_{agent}']
        plt.plot(price_serie, label = f'Agent {agent}')
    plt.plot(df_plot['p_monopoly'], color = 'red', label = 'Monopoly price')
    plt.plot(df_plot['p_nash'], color = 'green', label = 'Nash price')
    plt.xlabel('Timesteps')
    plt.ylabel('Prices')
    plt.legend()
    plt.savefig(f'figures/{exp_name}_prices.pdf')
    plt.close()
    
    ############################################
    plt.figure(figsize = (12, 4))
    plt.plot(df_plot['avg_prices'], label = 'Average prices')
    plt.plot(df_plot['p_monopoly'], color = 'red', label = 'Monopoly price')
    plt.plot(df_plot['p_nash'], color = 'green', label = 'Nash price')
    plt.xlabel('Timesteps')
    plt.ylabel('Prices')
    plt.legend()
    plt.savefig(f'figures/{exp_name}_avg_prices.pdf')
    plt.close()
    
    ############################################
    plt.figure(figsize = (12, 4))
    plt.plot(df_plot['avg_rewards'], label = 'Average profits')
    plt.plot(df_plot['pi_N'], label = 'Nash profits', color = 'green')
    plt.plot(df_plot['pi_M'], label = 'Monopoly profits', color = 'red')
    plt.xlabel('Timesteps')
    plt.ylabel('Profits')
    plt.legend()
    plt.savefig(f'figures/{exp_name}_rewards.pdf')
    plt.close()
    
    ############################################
    plt.figure(figsize = (12, 4))
    plt.plot(df_plot['delta'], label = 'Average profits')
    plt.axhline(1, color = 'red', label = 'Nash profits')
    plt.axhline(0, color = 'green', label = 'Monoply profits')
    plt.xlabel('Timesteps')
    plt.ylabel('Delta')
    plt.legend()
    plt.savefig(f'figures/{exp_name}_delta.pdf')
    plt.close()