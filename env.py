import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.optimize import minimize, root, fsolve
import matplotlib.pyplot as plt
from utils.get_rolling import get_rolling

class BertrandEnv(gym.Env):
  metadata = {'render_modes': None}

  def __init__(self, N = 2, k = 1, m = 10, xi = 0.1, a_0 = 0, a = None, mu = 0.25, c = 1, a_index = 1, convergence = 100):

    self.N = N # cantidad de agentes
    self.k = k # periodos hacia atrás que agentes observan
    self.m = m # cantidad de acciones
    self.a_0 = a_0 # indice diferenciacion vertical por default
    self.mu = mu # indice diferenciacion horizontal
    self.c = c # costo marginal
    self.a_index = a_index # indices de diferenciacion vertical
    self.convergence = convergence # cantidad de steps para concluir convergencia

    self.a = a
    if a is None:
      self.a = {agent: c + a_index for agent in range(N)}

    self.observation_space = spaces.Box(low = 0, high = m - 1, shape = (k, N), dtype = int)
    self.action_space = spaces.Discrete(m)

    # Monopoly Equilibrium Price
    def monopoly_func(p):
        return -(p[0] - self.c) * self.demand(p, 0)
      
    self.monopoly_price = minimize(monopoly_func, x0 = 0).x[0]

    # Nash Equilibrium Price
    nash_solution = fsolve(func = self.nash, x0 = (1.0, 1.0))
    assert all(round(price, 3) == round(nash_solution[0], 3) for price in nash_solution), f"Nash price should be unique: {nash_solution}"

    #for i in range(len(solution)):
      #  print(f'p{i}: {solution[i]}')

    self.nash_price = nash_solution[0]
    
    print(f'Monopoly Price: {self.monopoly_price}')
    print(f'Nash Price: {self.nash_price}')

    price_diff = self.monopoly_price - self.nash_price

    self._action_to_price = [{action: action * (price_diff + 2 * xi * price_diff)/(m - 1) for action in range(m)} 
                            for agent in range(N)]

    self.reward_list = []


  def nash(self, p):

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

  def obs_sample(self):
    '''
    Método que devuelve una muestra del espacio de observación.
    '''
    return list(self.observation_space.sample()[0])

  def _get_obs(self):
    '''
    Método que devuelve la observación actual, es decir, los k precios anteriores 
    en formato de state (en vez de los precios puros, su representación en índice).
    '''
    return str(self.history[-self.k:])

  def _get_info(self):
    pass

  def get_revenue(self, p):
    '''
    Método que recibe una lista de precios, 
    devuelve una lista con los rewards asociados a cada agente.
    '''

    r = [self.demand(p, agent) * (p[agent] - self.c) for agent in range(self.N)]

    return r

  def demand(self, p, agent):
      '''
      Retorna la cantidad vendida en función de la diferenciación vertical/horizontal y los precios.
      p: Diccionario de precios ofrecidos por los agentes (dict)
      agent: Agente a obtener la cantidad demandada (int)
      '''

      numerator = np.exp((self.a[agent] - p[agent]) / self.mu)
      denominator = sum([np.exp((self.a[agent] - p[agent])/self.mu) for agent in range(len(p))]) + np.exp(self.a_0 / self.mu)

      return numerator / denominator

  def reset(self):

    '''
    Método que resetea el environment. Devuelve un estado S_0 (vectores de precios) aleatorio.
    '''

    self.history = [] # lista de listas con las acciones elegidas para cada step
    sample_action = list(self.observation_space.sample()[0])
    self.history.append(sample_action)

    self.convergence_count = 0 # reset contador convergencia

    return self._get_obs()

  def step(self, action_n: list):

    '''
    Método que genera un step en el environment. 
    Recibe una lista de acciones indexadas (una por cada agente).
    Devuelve una tupla de (observation, reward, done, info)
    action_n: lista con acciones de los agentes (list)
    '''

    if type(action_n) != list:
      raise ValueError(f'Debes usar una lista con una acción por cada uno de los {self.N} agentes!')

    if len(action_n) != self.N:
      raise ValueError(f'Método recibe una lista de largo {self.N}')

    # terminar loop si precio se repite mas de convergence_count veces
    if action_n == self.history[-1]:
      self.convergence_count += 1
    else:
      self.convergence_count = 0
    terminated = True if self.convergence_count >= self.convergence else False

    self.history.append(action_n)

    # precios indexados
    observation = self._get_obs()

    # acciones a precios
    self.prices = [self._action_to_price[agent][action_n[agent]] for agent in range(self.N)]

    reward = self.get_revenue(self.prices)
    self.reward_list.append(reward)

    info = None

    return observation, reward, terminated, info

  def plot(self, window_size):

    history = np.array(self.history)

    plt.figure(figsize = (12, 6))
    for agent in range(history.shape[1]):
      series = history[:, agent] # acciones del agente i
      series = np.array([self._action_to_price[agent][action] for action in series]) # acciones a precios
      rolling_mean = get_rolling(series, window_size) # media móvil
      plt.plot(range(history.shape[0]), rolling_mean, label = f'Agent {agent}') # plot
    plt.xlabel('Steps')
    plt.ylabel('Price')
    plt.title('Solving Bertrand Environment with Multi Agent Q-learning')
    plt.legend()