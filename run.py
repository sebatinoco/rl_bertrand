from envs.bertrand import BertrandEnv
from envs.bertrand_diff import BertrandDiffEnv
from agent import QLearning
from utils.parse_args import parse_args
from trainer import Trainer
from utils.plot import plot
from utils.plot_epsilon import plot_epsilon

if __name__ == '__main__':
    
    # consolidate train arguments
    args = parse_args()
    args = vars(args)
    
    # envs available
    envs = {'bertrand': BertrandEnv, 'bertrand_diff': BertrandDiffEnv}
    
    # chosen env
    env = envs[args['env_name']]
    
    # arguments required by each class
    env_arguments = env.__init__.__code__.co_varnames
    trainer_arguments = Trainer.__init__.__code__.co_varnames
    plot_arguments = plot.__code__.co_varnames
    
    # filter arguments
    env_args = {arg_name: arg_value for arg_name, arg_value in args.items()
                if arg_name in env_arguments}
    trainer_args = {arg_name: arg_value for arg_name, arg_value in args.items()
                if arg_name in trainer_arguments}
    plot_args = {arg_name: arg_value for arg_name, arg_value in args.items()
                if arg_name in plot_arguments}
    
    # initialize environment
    env = env(**env_args)
    
    # initialize agents
    agents = [QLearning(env = env) for agent in range(env.N)]
    
    # trainer
    trainer = Trainer(**trainer_args)
    trainer.train(env = env, agents = agents, action_space = env.action_space.n)
    
    # plot
    plot(env = env, **plot_args)
    plot_epsilon(epsilon_list = trainer.epsilon_list, env = env, **plot_args)

    