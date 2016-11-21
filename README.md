# Tropofy bridge torch problem

This is a tropofy Python implementation of SARSA Reinforcement learning to solve the classic Bridge Torch problem

The description of the problem can be found here: https://en.wikipedia.org/wiki/Bridge_and_torch_problem

The definition of SARSA can be found here: http://www.cse.unsw.edu.au/~cs9417ml/RL1/algorithms.html

The general idea of the algorithm is to generate a reference solution which uses the quickest person to hold the torch back and forth the bridge, SARSA RL then compares the new crossing bridge time with this reference solution time, if the new solution is shorter than that is a positive reward else it is a negative reward. SARSA learns from those experience and should finally converged to the near optimal solution.
