import numpy as np

def greedy_policy(Qtable, state):
  
  '''
  Returns the action with the highest state-action value
  '''
  
  action = np.argmax(Qtable[state])
  
  return action