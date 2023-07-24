import os
import yaml
import torch

from agents.ddpg import DDPGAgent
from agents.sac import SACAgent
from agents.dqn import DQNAgent

from envs.BertrandInflation import BertrandEnv
from replay_buffer import ReplayBuffer
from utils.run_args import run_args
from utils.train import train

if __name__ == '__main__':
    
    configs = sorted(os.listdir('configs'))
    
    r_args = run_args() 
    
    train_agents = r_args['train_agents']
    filter_env = r_args['env']
    filter_config = r_args['filter_config']
    nb_experiments = r_args['nb_experiments']
    device = f"cuda:{r_args['gpu']}" if torch.cuda.is_available() else 'cpu'
    print(f'using {device}!')
    
    models_dict = {'sac': SACAgent, 'ddpg': DDPGAgent, 'dqn': DQNAgent}
    
    for experiment_idx in range(nb_experiments):
        
        # load config
        for config in configs:
            with open(f"configs/{config}", 'r') as file:
                    args = yaml.safe_load(file)

        # set experiment name
        exp_name = f"{args['exp_name']}_{experiment_idx}"
        
        # load model
        model = models_dict[args['model']] 
        
        # load environment, agent and buffer
        env = BertrandEnv(**args['env'])      
        env.reset()
          
        dim_states = env.N + 1 if args['use_lstm'] else env.k * env.N + env.k + 1
        dim_actions = args['n_actions']
        
        agents = [model(dim_states, dim_actions, env.price_low, env.price_high, **args['agent']) for _ in range(env.N)]
        buffer = ReplayBuffer(N = env.N, **args['buffer'])
        
        # train
        train(env, agents, buffer, env.N, exp_name = exp_name, **args['train'])