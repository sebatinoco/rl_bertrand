import argparse
from distutils.util import strtobool

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # env arguments
    parser.add_argument('--env_name', type = str, default = 'BertrandDiff', help = 'environment name to run the experiment')
    parser.add_argument('--N', type = int, default = 2, help = 'number of agents')
    parser.add_argument('--k', type = int, default = 1, help = 'past periods observed by agents')
    parser.add_argument('--v', type = int, default = 3, help = 'past periods to predict next inflation value')
    parser.add_argument('--rho', type = float, default = 0.001, help = 'probability of changing prices')
    parser.add_argument('--xi', type = float, default = 0.2, help = 'term to amplify range of actions')
    
    # buffer arguments
    parser.add_argument('--sample_size', type = int, default = 512, help = 'batch size')
    parser.add_argument('--buffer_size', type = int, default = 200000, help = 'buffer size')
    
    # agent arguments
    parser.add_argument('--actor_lr', type = float, default = 0.01, help = 'learning rate of the agents')
    parser.add_argument('--Q_lr', type = float, default = 0.01, help = 'learning rate of the agents')
    parser.add_argument('--gamma', type = float, default = 0.99, help = 'gamma coeff of the agents')
    parser.add_argument('--tau', type = float, default = 0.001, help = 'tau coeff of the agents')
    parser.add_argument('--hidden_size', type = int, default = 256, help = 'hidden dim of the agents')
    
    # train arguments
    parser.add_argument('--timesteps', type = int, default = int(2e4), help = 'number of steps')
    parser.add_argument('--learning_start', type = int, default = 100, help = 'steps to start learning')
    parser.add_argument('--update_steps', type = int, default = 10, help = 'steps per update')
    parser.add_argument('--plot_steps', type = int, default = 50, help = 'steps per update')
    parser.add_argument('--rolling', type = int, default = 1000, help = 'rolling steps')
    #parser.add_argument("--seed", type = int, default = 3380, help = "seed of the experiment")
    
    # plot arguments
    parser.add_argument('--plots_dir', type = str, default = 'plots', help = 'folder dir to save plot results')
    parser.add_argument('--exp_name', type = str, default = 'bertrand', help = 'name of the experiment')
    parser.add_argument('--window_size', type = int, default = 100, help = 'window size of moving average')
    parser.add_argument('--plot_loss', type = lambda x: bool(strtobool(x)), default = False, help = 'enable plot loss')
    
    # consolidate args
    args = parser.parse_args()
    args = vars(args)
    
    return args