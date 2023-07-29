import numpy as np
import matplotlib.pyplot as plt
#from tqdm import tqdm
from replay_buffer import ReplayBuffer
from utils.parse_args import parse_args
from utils.plot_metrics import plot_metrics
from utils.export_results import export_results
import sys

#from envs.BertrandInflation import BertrandEnv
#from envs.BertrandInflation_cost import BertrandEnv
#from envs.SimpleBertrandInflation import BertrandEnv
from envs.BertrandInflation_final import BertrandEnv
from envs.LinearBertrandInflation_final import LinearBertrandEnv

from agents.ddpg import DDPGAgent
from agents.dqn_linear import DQNAgent # precios no se estan graficando bien
from agents.sac import SACAgent
#from agents.ddpg_cost import DDPGAgent

models_dict = {'sac': SACAgent, 'ddpg': DDPGAgent, 'dqn': DQNAgent}
envs_dict = {'bertrand': BertrandEnv, 'linear': LinearBertrandEnv}

if __name__ == '__main__':
    
    # load args
    args = parse_args()

    # initiate environment
    env = envs_dict[args['env']]
    env = env(N = args['N'], k = args['k'], rho = args['rho'], v = args['v'], xi = args['xi'])
    ob_t = env.reset()
    
    # get dimensions
    dim_states = env.N if args['use_lstm'] else env.k * env.N + env.k + 1
    dim_actions = args['n_actions'] if args['model'] == 'dqn' else 1
    
    # limit prices
    expected_shocks = int((args['timesteps'] - args['inflation_start']) * args['rho'])
    print('Expected shocks:', expected_shocks)
    price_low, price_high = (np.log(env.price_low), np.log(env.price_high * (1.05 ** expected_shocks)))

    # initiate agents
    model = models_dict[args['model']]
    agents = [model(dim_states, dim_actions, price_low, price_high) for _ in range(args['N'])]
    
    # initiate buffer
    buffer = ReplayBuffer(N = args['N'], buffer_size = args['buffer_size'], sample_size = args['sample_size'])

    # initiate plot
    plot_dim = (2, 3) if args['plot_loss'] else (1, 3)
    fig, axes = plt.subplots(*plot_dim, figsize = (16, 6) if args['plot_loss'] else (16, 4))
    axes = np.array(axes, ndmin = 2)

    # train
    for t in range(args['timesteps']):
        # select action
        actions = [agent.select_action(ob_t) for agent in agents] 
        
        # trigger deviation
        if (t > args['timesteps'] // 2) & (args['trigger_deviation']):
            actions[0] = env.pN
        
        # start changing prices
        if t > args['inflation_start']:
            env.inflation_start = True
        
        #actions = [env.pM for _ in range(env.N)]
        
        # step
        ob_t1, rewards, done, info = env.step(actions)
        
        # store transition
        experience = (ob_t, actions, rewards, ob_t1, done)
        buffer.store_transition(*experience)
        
        # update and plot
        if (t % args['update_steps'] == 0) & (t >= args['sample_size']):
            # update
            for agent_idx in range(args['N']):
                agent = agents[agent_idx]
                sample = buffer.sample(agent_idx)
                agent.update(*sample)

            # plot
            if t % args['plot_steps'] == 0:
                plot_args = (fig, axes, env.prices_history, env.monopoly_history, env.nash_history, env.rewards_history, env.metric_history, args['window'])
                plot_metrics(*plot_args, agent.actor_loss, agent.Q_loss) if args['plot_loss'] else plot_metrics(*plot_args)     
        
        sys.stdout.write(f"\rTraining completion: {100 * t/args['timesteps']:.2f} % \t Delta: {info:.2f}")
        
        # update ob_t
        ob_t = ob_t1
    
    # save plot
    plt.savefig(f"figures/{args['exp_name']}.pdf")
    
    # export results
    export_results(env.prices_history[env.k:], env.quantities_history,
                   env.monopoly_history[1:], env.nash_history[1:], 
                   env.rewards_history, env.metric_history, 
                   env.costs_history[env.k:], args['exp_name'])