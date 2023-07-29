import pandas as pd

def export_results(prices_history, quantities_history, monopoly_history, 
                   nash_history, rewards_history, metric_history, 
                   costs_history, exp_name):
    
    exp_dict = {'prices_history': prices_history, 'quantities_history': quantities_history,
                'rewards_history': rewards_history, 'costs_history': costs_history,
                'monopoly_history': monopoly_history, 'nash_history': nash_history, 
                'metric_history': metric_history}
    
    #print('prices_history:', len(prices_history))
    #print('quantities_history:', len(quantities_history))
    #print('rewards_history:', len(rewards_history))
    #print('costs_history:', len(costs_history))
    #print('monopoly_history:', len(monopoly_history))
    #print('nash_history:', len(nash_history))
    #print('metric_history:', len(metric_history))
    
    exp_metrics = pd.DataFrame(exp_dict)
    
    exp_metrics.to_csv(f'metrics/{exp_name}.csv', sep = '\t')