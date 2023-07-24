import pandas as pd

def export_results(prices_history, monopoly_history, nash_history, rewards_history, metric_history, exp_name):
    
    exp_dict = {'prices_history': prices_history, 'monopoly_history': monopoly_history, 
                  'nash_history': nash_history, 'rewards_history': rewards_history,
                  'metric_history': metric_history}
    
    exp_metrics = pd.DataFrame(exp_dict)
    
    exp_metrics.to_csv(f'metrics/{exp_name}.csv', sep = '\t')