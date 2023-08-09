import os
import yaml
import torch
import numpy as np
import time

from agents.ddpg import DDPGAgent
from agents.sac import SACAgent
from agents.dqn import DQNAgent

#from envs.BertrandInflation import BertrandEnv
#from envs.BertrandInflation_final import BertrandEnv
from envs.BertrandInflation_final2 import BertrandEnv
from envs.LinearBertrandInflation_final import LinearBertrandEnv
from replay_buffer import ReplayBuffer
from utils.run_args import run_args
from utils.train import train
from utils.get_results import get_results

models_dict = {'sac': SACAgent, 'ddpg': DDPGAgent, 'dqn': DQNAgent}
envs_dict = {'bertrand': BertrandEnv, 'linear': LinearBertrandEnv}

if __name__ == '__main__':
    
    configs = sorted(os.listdir('configs'))
    
    r_args = run_args() 
    
    train_agents = r_args['train_agents']
    filter_env = r_args['env']
    filter_config = r_args['filter_config']
    nb_experiments = r_args['nb_experiments']
    device = f"cuda:{r_args['gpu']}" if torch.cuda.is_available() else 'cpu'
    print(f'using {device}!')
    
    # filter configs if specified
    if filter_env or filter_config:
        
        env_configs = [config for config in configs if len(set(filter_env) & set(config.split('_'))) > 0] # filter by environment
        filtered_configs = [config for config in configs if config in filter_config] # filter by config
        
        final_configs = set(env_configs + filtered_configs) # filtered configs
        configs = [config for config in configs if config in final_configs] # filter configs

    print('Running experiments on the following configs: ', configs)
    
    for experiment_idx in range(1, nb_experiments + 1):
        start_time = time.time()
        # load config
        for config in configs:
            with open(f"configs/{config}", 'r') as file:
                args = yaml.safe_load(file)
                agent_args = args['agent']
                env_args = args['env']
                buffer_args = args['buffer']
                train_args = args['train']

            #train_args['timesteps'] = 1000

            # set experiment name
            exp_name = f"{args['exp_name']}_{experiment_idx}"
            
            # load model
            model = models_dict[args['model']] 
            
            # load environment, agent and buffer
            env = envs_dict[args['env_name']]
            #env = envs_dict['bertrand']
            env = env(**env_args)      
            
            dim_states = env.N + 1 if args['use_lstm'] else env.k * env.N + env.k + 1
            dim_actions = args['n_actions'] if args['model'] == 'dqn' else 1
            
            # limit prices
            expected_shocks = int((train_args['timesteps'] - train_args['inflation_start']) * env_args['rho'])
            print('\n' + 'Expected shocks:', expected_shocks)
            price_low, price_high = (np.log(env.price_low), np.log(env.price_high * (1.05 ** expected_shocks)))
            
            agents = [model(dim_states, dim_actions, env.price_low, env.price_high, **agent_args) for _ in range(env.N)]
            buffer = ReplayBuffer(N = env.N, **buffer_args)
            
            # train
            train(env, agents, buffer, env.N, exp_name = exp_name, **train_args)
            
        execution_time = time.time() - start_time

        print(f'{execution_time:.2f} seconds -- {(execution_time/60):.2f} minutes -- {(execution_time/3600):.2f} hours')  
    
    get_results(n_experiments = nb_experiments + 1)