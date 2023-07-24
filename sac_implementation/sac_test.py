from sac2019 import SACAgent
from common.utils import mini_batch_train
import gym

env = gym.make("Pendulum-v1")

# SAC 2019 Params
gamma = 0.99
tau = 0.01
alpha = 0.2
a_lr = 3e-4
q_lr = 3e-4
p_lr = 3e-4
buffer_maxlen = 1000000

state, info = env.reset()

#2019 agent
agent = SACAgent(env, gamma, tau, alpha, q_lr, p_lr, a_lr, buffer_maxlen)

# train
episode_rewards = mini_batch_train(env, agent, max_episodes = 50, max_steps = 500, batch_size = 64)