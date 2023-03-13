import argparse

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type = int, default = 3380,
        help = "seed of the experiment")
    
    # env arguments
    parser.add_argument('--N', type = int, default = 2, help = 'number of agents')
    parser.add_argument('--k', type = int, default = 1, help = 'past periods observed by agents')
    parser.add_argument('--m', type = int, default = 10, help = 'amount of actions available to each agent')
    parser.add_argument('--a_0', type = float, default = 1.0, help = 'base vertical differentiation index')
    parser.add_argument('--a_index', type = float, default = 1.0, help = 'vertical differentiation index by agent')
    parser.add_argument('--mu', type = float, default = 0.25, help = 'horizontal differentiation index')
    parser.add_argument('--c', type = float, default = 1.0, help = 'marginal cost (assumed equal to each agent)')
    parser.add_argument('--xi', type = float, default = 0.1, help = 'term to amplify range of actions')
    parser.add_argument('--convergence', type = int, default = 1000, help = 'min steps with repeated actions to conclude convergence')
    
    # agent arguments
    
    
    # training arguments
    parser.add_argument('--episodes', type = int, default = 1, help = 'number of episodes of the experiment')
    parser.add_argument('--n_steps', type = int, default = int(1e5), help = 'number of steps per episode')
    parser.add_argument('--decay_rate', type = float, default = 5e-4, help = 'decay rate of exploration phase')
    parser.add_argument('--max_epsilon', type = float, default = 1.0, help = 'max epsilon of the agent')
    parser.add_argument('--min_epsilon', type = float, default = 0.0, help = 'min epsilon of the agent')
    parser.add_argument('--lr', type = float, default = 0.05, help = 'learning rate of the agents')
    parser.add_argument('--gamma', type = float, default = 0.95, help = 'gamma coeff of the experiment')
    
    # plot arguments
    parser.add_argument('--plots_dir', type = str, default = 'plots', help = 'folder dir to save plot results')
    parser.add_argument('--exp_name', type = str, default = 'bertrand', help = 'name of the experiment')
    
    # consolidate args
    args = parser.parse_args()
    
    return args