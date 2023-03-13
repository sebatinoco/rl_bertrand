import numpy as np

def greedy_policy(Qtable, state):
  '''
  Método que devuelve la acción con el mayor state-action value
  '''
  action = np.argmax(Qtable[state])
  
  return action