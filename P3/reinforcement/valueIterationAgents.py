# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):

        "*** YOUR CODE HERE question 1 ***"
        for k in range(self.iterations):
            newValues = self.values.copy()
            states = self.mdp.getStates()

            for state in states:
                actions = self.mdp.getPossibleActions(state)

                if actions:
                    qValue = max(self.computeQValueFromValues(state, action) for action in actions)
                    newValues[state] = qValue
                else:
                    newValues[state] = self.values[state]

            self.values = newValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE question 1 ***"
        qValue = 0

        if self.mdp.isTerminal(state):
            return None

        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            qValue += prob * (reward + (self.discount * self.values[nextState]))

        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit. Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE question 1 ***"
        actions = self.mdp.getPossibleActions(state)

        if not actions or self.mdp.isTerminal(state):
            return None

        result = max(actions, key = lambda action: self.getQValue(state, action))

        return result

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE question 4 ***"
        states = self.mdp.getStates()

        for k in range(self.iterations):
            state = states[k % len(states)] #the mod is $money$, keep that in your tool belt!

            newValues = self.values.copy()

            actions = self.mdp.getPossibleActions(state)

            if actions:
                qValue = max(self.getQValue(state, action) for action in actions)
                newValues[state] = qValue
            else: #case where MDP has no actions
                newValues[state] = self.values[state]

            self.values = newValues

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE question 5 ***"
        states = self.mdp.getStates()
        values = self.values
        stateValue = 0
        stateQueue = util.PriorityQueue()
        predecessors = dict()

        for state in states:
            if not self.mdp.isTerminal(state):
                sPrimes = []
                actions = self.mdp.getPossibleActions(state)

                for action in actions:
                    sPrimes = self.mdp.getTransitionStatesAndProbs(state, action)

                    for s, p in sPrimes:
                        if s in predecessors:
                            predecessors[s].add(state)
                        else:
                            predecessors[s] = set()
                            predecessors[s].add(state)

                stateValue = values[state]
                qValue = max(self.getQValue(state, action) for action in actions)
                diff = abs(stateValue - qValue)

                stateQueue.update(state, -diff)

        for iter in range(self.iterations):
            if stateQueue.isEmpty():
                return

            currentState = stateQueue.pop()

            if not self.mdp.isTerminal(currentState):
                actions = self.mdp.getPossibleActions(currentState)
                currentValue = max(self.getQValue(currentState, action) for action in actions)
                self.values[currentState] = currentValue

            for pred in predecessors[currentState]:
                actions = self.mdp.getPossibleActions(pred)
                predValue = self.values[pred]

                qValue = max(self.getQValue(pred, action) for action in actions)
                diff = abs(predValue - qValue)

                if diff > self.theta:
                    stateQueue.update(pred, -diff)
