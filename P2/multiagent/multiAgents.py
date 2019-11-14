# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        # Note: Remember that newFood has the function asList()
        # Note: As features, try the ***reciprocal*** of important values (such as distance to food) rather than just the values themselves.
        # Note: The evaluation function you’re writing is evaluating state-action pairs; in later parts of the project, you’ll be evaluating states.
        #"State-action pairs": does that mean the states of pacman vs ghosts one at a time?

        # Options: Default ghosts are random; you can also play for fun with slightly smarter directional ghosts using -g DirectionalGhost.
        # If the randomness is preventing you from telling whether your agent is improving, you can use -f to run with a fixed random seed
        # (same random choices every game). You can also play multiple games in a row with -n. Turn off graphics with -q to run lots of games quickly.
        "*** YOUR CODE HERE question 1***"
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        currentScore = currentGameState.getScore()
        powerPellet = currentGameState.getCapsules()
        foodList = [1]
        foodScore = 0

        ghostScore = 0
        foodScore = len(newFood.asList())
        foodList = [1]
        ghostList = [1]
        foodWeight = 1
        motivateWeight = 20

        for food in newFood.asList():
            foodList += [-1 * util.manhattanDistance(newPos, food)]

        for ghost in newGhostStates:
            ghostList += [util.manhattanDistance(newPos, ghost.getPosition())]

        ghostScore = max(ghostList)
        foodScore = max(foodList)

        if ghostScore > 2:
            ghostScore *= -20

        for times in newScaredTimes:
            if times > 0:
                ghostScore *= 700

        pacMotivate =  motivateWeight * (successorGameState.getScore() + (1 /len(foodList)) * foodWeight)
        evalSum = pacMotivate + ghostScore + (1 / foodScore)

        return evalSum
        # need to balance foodWeight and ghostScores
        # return successorGameState.getScore() #return linear weighted sum of game state features HIGHER better

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated. It's
    only partially specified, and designed to be extended. Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

#Score the leaves of your minimax tree with the supplied self.evaluationFunction, which defaults to scoreEvaluationFunction
#MinimaxAgent extends MultiAgentSearchAgent, which gives access to self.depth and self.evaluationFunction
#Important: A single search ply is considered to be one Pacman move and all the ghosts’ responses, so depth 2 search will involve Pacman and each ghost moving two times.
#As a result, the autograder will be very picky about how many times you call GameState.generateSuccessor.
#If you call it any more or less than necessary, the autograder will complain!

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def value(gameState, agentIndex, depth):

            if gameState.isLose() or gameState.isWin() or depth == 0:
                return (self.evaluationFunction(gameState), Directions.STOP)

            if agentIndex == 0:
                return maxAgent(gameState, agentIndex, depth - 1)

            else:
                return minAgent(gameState, agentIndex, depth)

        def minAgent(state, index, depth):

            minVal = v = float("inf")
            minActions = state.getLegalActions(index)
            bestAction = Directions.STOP

            for action in minActions:
                minState = state.generateSuccessor(index, action)
                minVal = min(v, value(minState, (index + 1) % state.getNumAgents(), depth)[0])

                if minVal < v: #DOH! was pointing the wrong way...
                    bestAction = action
                    v = minVal

            return (minVal, bestAction)

        def maxAgent(state, index, depth):

            if state.isLose() or state.isWin() or depth == 0:
                return (self.evaluationFunction(state), Directions.STOP)

            maxVal = v = float("-inf")
            maxActions = state.getLegalActions(index)
            bestAction = Directions.STOP

            for action in maxActions:
                maxState = state.generateSuccessor(index, action)
                maxVal = max(v, value(maxState, (index + 1) % state.getNumAgents(), depth)[0])

                if maxVal > v:
                    bestAction = action
                    v = maxVal

            return (maxVal, bestAction)

        return maxAgent(gameState, self.index, self.depth)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState): #returns an action
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        """
        "*** YOUR CODE HERE ***"

        def value(gameState, agentIndex, depth, alpha, beta):

            if gameState.isLose() or gameState.isWin() or depth == 0:
                return (self.evaluationFunction(gameState), Directions.STOP)

            if agentIndex == 0:
                return maxAgent(gameState, agentIndex, depth - 1, alpha, beta)

            else:
                return minAgent(gameState, agentIndex, depth, alpha, beta)

        def minAgent(state, index, depth, alpha, beta):

            minVal = v = float("inf")
            minActions = state.getLegalActions(index)
            bestAction = Directions.STOP

            for action in minActions:
                minState = state.generateSuccessor(index, action)
                minVal = min(v, value(minState, (index + 1) % state.getNumAgents(), depth, alpha, beta)[0])

                if minVal < v:
                    bestAction = action
                    v = minVal

                beta = min(beta, v)

                if beta < alpha: #You must not prune on equality in order to match the set of states explored by our autograder.
                    return (v, bestAction)

            return (minVal, bestAction)

        def maxAgent(state, index, depth, alpha, beta):

            if state.isLose() or state.isWin() or depth == 0:
                return (self.evaluationFunction(state), Directions.STOP)

            maxVal = v = float("-inf")
            maxActions = state.getLegalActions(index)
            bestAction = Directions.STOP

            for action in maxActions:
                maxState = state.generateSuccessor(index, action)
                maxVal = max(v, value(maxState, (index + 1) % state.getNumAgents(), depth, alpha, beta)[0])

                if maxVal > v:
                    bestAction = action
                    v = maxVal

                alpha = max(alpha, v)

                if beta < alpha: #You must not prune on equality in order to match the set of states explored by our autograder.
                    return (v, bestAction)

            return (maxVal, bestAction)

        return maxAgent(gameState, self.index, self.depth, float("-inf"), float("inf"))[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def value(gameState, agentIndex, depth):

            if gameState.isLose() or gameState.isWin() or depth == 0:
                return (self.evaluationFunction(gameState), Directions.STOP)

            if agentIndex == 0:
                return maxAgent(gameState, agentIndex, depth - 1)

            else:
                return expAgent(gameState, agentIndex, depth)

        def expAgent(state, index, depth):

            minVal = v = 0
            minActions = state.getLegalActions(index)
            bestAction = Directions.STOP

            for action in minActions:
                minState = state.generateSuccessor(index, action)
                p = 1 / len(minActions) #equal chance of each action
                minVal += (p * value(minState, (index + 1) % state.getNumAgents(), depth)[0])

                if minVal < v: #DOH! was pointing the wrong way...
                    bestAction = action
                    v = minVal

            return (minVal, bestAction)

        def maxAgent(state, index, depth):

            if state.isLose() or state.isWin() or depth == 0:
                return (self.evaluationFunction(state), Directions.STOP)

            maxVal = v = float("-inf")
            maxActions = state.getLegalActions(index)
            bestAction = Directions.STOP

            for action in maxActions:
                maxState = state.generateSuccessor(index, action)
                maxVal = max(v, value(maxState, (index + 1) % state.getNumAgents(), depth)[0])

                if maxVal > v:
                    bestAction = action
                    v = maxVal

            return (maxVal, bestAction)

        return maxAgent(gameState, self.index, self.depth)[1]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # state = currentGameState
    # numGhosts = currentGameState.get
    # return 1
    currentScore = currentGameState.getScore()
    pacPosition = currentGameState.getPacmanPosition()
    foodState = currentGameState.getFood()
    powerPellet = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    foodList = [1]
    foodScore = 0
    ghostValue = 0
    # print(foodList)

    for ghost in ghostStates:
        ghostPosition = util.manhattanDistance(ghost.getPosition(), pacPosition)
        scaredTime = ghost.scaredTimer

        if ghostPosition <= 2 and scaredTime == 0:
            ghostValue = float("-inf")

        if scaredTime > 0:
            ghostValue = 100 / ghostPosition

    for food in foodState.asList():

        foodList += [util.manhattanDistance(pacPosition, food)]

    return currentScore + (ghostValue + 1/max(foodList))

# Abbreviation
better = betterEvaluationFunction
