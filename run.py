from env import BertrandEnv
from agent import QLearning
from utils.parse_args import parse_args
from train import train
from utils.plot import plot

if __name__ == '__main__':
    
    # consolidate train arguments
    args = parse_args()
    args = vars(args)
    
    # arguments required by each class
    env_arguments = BertrandEnv.__init__.__code__.co_varnames
    train_arguments = train.__code__.co_varnames
    plot_arguments = plot.__code__.co_varnames
    
    # filter arguments
    env_args = {arg_name: arg_value for arg_name, arg_value in args.items()
                if arg_name in env_arguments}
    train_args = {arg_name: arg_value for arg_name, arg_value in args.items()
                if arg_name in train_arguments}
    plot_args = {arg_name: arg_value for arg_name, arg_value in args.items()
                if arg_name in plot_arguments}
    
    # initialize environment
    env = BertrandEnv(**env_args)
    
    # initialize agents
    agents = [QLearning(env = env) for agent in range(env.N)]
    
    # train
    train(env = env, agents = agents, action_space = env.action_space.n, **train_args)
    
    plot(env = env, **plot_args)
    