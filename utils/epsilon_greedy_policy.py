import random
from utils.greedy_policy import greedy_policy

def epsilon_greedy_policy(Qtable, state, epsilon, action_space):

  '''
  Método que devuelve una acción según la política epsilon_greedy.
  Qtable: Q-table del algoritmo Q-learning (list)
  state: Estado observado (int?)
  epsilon: Epsilon a computar epsilon_greedy
  '''

  # Randomly generate a number between 0 and 1
  random_num = random.random()
  # if random_num > greater than epsilon --> exploitation
  if random_num > epsilon:
    # Take the action with the highest value given a state
    # np.argmax can be useful here
    action = greedy_policy(Qtable, state)
  # else --> exploration
  else:
    action = random.randint(0, action_space - 1) # Take a random action
  
  return action