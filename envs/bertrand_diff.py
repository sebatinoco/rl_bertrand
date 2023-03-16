import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.optimize import minimize, fsolve
import matplotlib.pyplot as plt
from utils.get_rolling import get_rolling

class BertrandDiffEnv(gym.Env):
  
  '''
  Environment representing the market dynamics of a canonical logit demand with vertical and horizontal differentiation.
  Contains the basic gym functions: step and reset.
  '''
  
  metadata = {'render_modes': None}

  def __init__(self, N = 2, k = 1, m = 10, xi = 0.1, a_0 = 0, a = None, mu = 0.25, c = 1, a_index = 1, convergence = 1000):

    self.N = N # number of agents
    self.k = k # past periods to observe
    self.m = m # number of actions 
    self.a_0 = a_0 # base vertical differentiation index
    self.mu = mu # horizontal differentiation index
    self.c = c # marginal cost
    self.a_index = a_index # vertical differentiation indexes
    self.convergence = convergence # number of steps to conclude convergence

    self.a = a
    if a is None:
      self.a = {agent: c + a_index for agent in range(N)}

    self.observation_space = spaces.Box(low = 0, high = m - 1, shape = (k, N), dtype = int)
    self.action_space = spaces.Discrete(m)

    # Monopoly Equilibrium Price
    self.monopoly_price = minimize(self.monopoly, x0 = 0).x[0]

    # Nash Equilibrium Price
    nash_solution = fsolve(func = self.nash, x0 = [1.0] * N)
    assert all(round(price, 4) == round(nash_solution[0], 4) for price in nash_solution), f"Nash price should be unique: {nash_solution}"

    self.nash_price = nash_solution[0]
    
    print(f'Monopoly Price: {self.monopoly_price}')
    print(f'Nash Price: {self.nash_price}')

    price_diff = self.monopoly_price - self.nash_price
    
    self._action_to_price = np.linspace(self.nash_price - xi * (self.monopoly_price - self.nash_price), self.monopoly_price + xi * (self.monopoly_price - self.nash_price), self.m)
    self._action_to_price = [{i: self._action_to_price[i] for i in range(len(self._action_to_price))} for agent in range(N)]

    self.reward_list = []

  def nash(self, p):
    
    '''
    Nash problem. Containes the derivatives of each agent with respecto its price.
    '''

    #assert len(a) == len(c), "a must be equal size to c"
    assert len(self.a) == len(p), "a must be equal size to p"
    #assert len(c) == len(p), "c must be equal size to p"

    sum_denominator = [np.exp((self.a[i] - p[i]) / self.mu) for i in range(len(p))]
    sum_denominator.append(np.exp(self.a_0 / self.mu))
    sum_denominator = sum(sum_denominator)

    result = []
    for i in range(len(p)):
      first_term = np.exp((self.a[i] - p[i]) / self.mu) / sum_denominator
      second_term = (np.exp((self.a[i] - p[i])/self.mu) * (p[i] - self.c)) / self.mu * sum_denominator
      third_term = (p[i] - self.c) / self.mu

      fn = first_term * (1 + second_term - third_term)
      result.append(fn)

    return result
  
  def monopoly(self, p):
    
    '''
    Monopoly maximization problem. 
    '''
    
    return -(p[0] - self.c) * self.demand(p, 0)

  def obs_sample(self):
    
    '''
    Returns a sample of the observation space.
    '''
    
    return list(self.observation_space.sample()[0])

  def _get_obs(self):
    
    '''
    Returns the current observation (k past actions) in state format (index)
    '''
    
    return str(self.history[-self.k:])

  def _get_info(self):
    return {'convergence_count': self.convergence_count}

  def get_revenue(self, p):
    
    '''
    Receives a list of prices, returns a list of rewards for each agent.
    '''

    r = [self.demand(p, agent) * (p[agent] - self.c) for agent in range(self.N)]

    return r

  def demand(self, p, agent):
    
      '''
      Returns the sold quantity in function of the vertical and horizontal differentiation, as per the prices set.
      p: Dictionary of prices offered by agents (dict)
      agent: Agent to obtain the quantity sold.
      '''

      numerator = np.exp((self.a[agent] - p[agent]) / self.mu)
      denominator = sum([np.exp((self.a[agent] - p[agent])/self.mu) for agent in range(len(p))]) + np.exp(self.a_0 / self.mu)

      return numerator / denominator

  def reset(self):

    '''
    Resets the environment. Returns a random state S_0 (vector of prices).
    '''

    self.history = [] # list of lists with chosen actions per step
    sample_action = list(self.observation_space.sample()[0])
    self.history.append(sample_action)

    self.convergence_count = 0 # reset convergence_count

    return self._get_obs()

  def step(self, action_n: list):

    '''
    Generates a step on the environment.
    Receives a list of indexed actions (one per agent).
    Returns a tuple (observation, reward, done, info)
    action_n: list with actions of the agents (list)
    '''

    if type(action_n) != list:
      raise ValueError(f'You must provide a list with actions for each of the {self.N} agents!')

    if len(action_n) != self.N:
      raise ValueError(f'Method receives a list of size {self.N}')

    # end loop if convergence_count > convergence
    if action_n == self.history[-1]:
      self.convergence_count += 1
    else:
      self.convergence_count = 0
    terminated = True if self.convergence_count >= self.convergence else False

    self.history.append(action_n)

    # indexed prices
    observation = self._get_obs()

    # actions to prices
    self.prices = [self._action_to_price[agent][action_n[agent]] for agent in range(self.N)]

    reward = self.get_revenue(self.prices)
    self.reward_list.append(reward)

    info = self._get_info()

    return observation, reward, terminated, info

  def plot(self, window_size):

    history = np.array(self.history)

    plt.figure(figsize = (12, 6))
    for agent in range(history.shape[1]):
      series = history[:, agent] # actions of agent i
      series = np.array([self._action_to_price[agent][action] for action in series]) # actions to prices
      rolling_mean = get_rolling(series, window_size) # moving average
      plt.plot(range(history.shape[0]), rolling_mean, label = f'Agent {agent}') # plot
    plt.xlabel('Steps')
    plt.ylabel('Price')
    plt.title('Solving Bertrand Environment with Multi Agent Q-learning')
    plt.legend()