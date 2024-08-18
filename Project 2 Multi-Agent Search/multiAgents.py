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
from pacman import GameState
import numpy as np

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood() #remaining food
        newGhostStates = successorGameState.getGhostStates()

        # print("My_pos: ", newPos)
        # print("Food: ", currentGameState.getFood().asList())

        # for i in newGhostStates:
        #     print("Ghost: ", i.getPosition())

         # Distance to the closest food
        foodDistances = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        closestFoodDistance = min(foodDistances) if foodDistances else 0

        num_food = len(newFood.asList())

        # Adjust the ghost score to ensure Pacman avoids non-scared ghosts effectively
        ghostPenalty = 0
        ghost_zones = []
        danger = 0
        for i in newGhostStates:
            distance = manhattanDistance(newPos, i.getPosition())
            if i.scaredTimer > 0 and distance < 10:  
                ghostPenalty -= distance * 2  
            else:
                if distance < 3:
                    ghostPenalty = -100000000    
        
        foodScore = -2 * closestFoodDistance  

        return successorGameState.getScore() + foodScore + ghostPenalty + danger - (num_food * 100)

def scoreEvaluationFunction(currentGameState: GameState):
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

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
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

    def getAction(self, gameState: GameState):
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

        def minimax(agentIndex, depth, gameState):
            #if the state is a terminal state: return the state’s uUlity
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            # if the next agent is MAX: return max-value(state)
            if agentIndex == 0:  # Pac-Man
                return max_value(agentIndex, depth, gameState)
            else:  
                return min_value(agentIndex, depth, gameState)

        def max_value(agentIndex, depth, gameState):
            v = -100000000000
            for action in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                v = max(v, minimax(1, depth, successorGameState))
            return v

        def min_value(agentIndex, depth, gameState):
            v = 1000000000000
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth + 1 if nextAgentIndex == 0 else depth
            for action in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                v = min(v, minimax(nextAgentIndex, nextDepth, successorGameState))
            return v

        # Collect legal moves and successor states
        # agentIndex=0 means Pacman, ghosts are >= 1
        legalMoves = gameState.getLegalActions(0)
        
        # Choose one of the best actions
        scores = [minimax(1, 0, gameState.generateSuccessor(0, action)) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  

        return legalMoves[chosenIndex]
        # def value(state):
        #     if GameState.isWin():
        #         return self.evaluationFunction(GameState)
            
        #     if next agennt is pacman:
        #         return max_value(state)
            
        #     if next agemt is ghost:
        #         return min_value(state)
    
    
        # def max_value(state):
        #     v = -100000000000

        #     for sucessor of state:
        #         v = max(v, value(sucessor) )
        #     return v
        
        # def min_value(state):
        #     v = 100000000000

        #     for sucessor of state:
        #         v = min(v, value(sucessor) )
        #     return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alpha_beta(agentIndex, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agentIndex == 0:  # Pac-Man, maximize score
                return max_value(agentIndex, depth, gameState, alpha, beta)
            else:  # Ghosts, minimize score
                return min_value(agentIndex, depth, gameState, alpha, beta)

        def max_value(agentIndex, depth, gameState, alpha, beta):
            v = -10000000000000
            for action in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                v = max(v, alpha_beta(1, depth, successorGameState, alpha, beta))
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(agentIndex, depth, gameState, alpha, beta):
            v = 100000000000000
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth + 1 if nextAgentIndex == 0 else depth
            for action in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                v = min(v, alpha_beta(nextAgentIndex, nextDepth, successorGameState, alpha, beta))
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        # Initial values for alpha and beta
        alpha, beta = float("-inf"), float("inf")
        bestAction = None
        for action in gameState.getLegalActions(0):
            score = alpha_beta(1, 0, gameState.generateSuccessor(0, action), alpha, beta)
            if score > alpha:
                alpha = score
                bestAction = action
        return bestAction
    
    # def max-value(state, α, β):
    #     iniUalize v = -∞
    #     for each successor of state:
    #         v = max(v, value(successor, α, β))
    #         if v ≥ β return v
    #         α = max(α, v)
    #     return v

    # def min-value(state , α, β):
    #     iniUalize v = +∞
    #     for each successor of state:
    #         v = min(v, value(successor, α, β))
    #         if v ≤ α return v
    #         β = min(β, v)
    #     return v


    


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
       getLegalActions uniformly at random.
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        def expectimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            # if the next agent is MAX: return max-value(state)
            if agentIndex == 0:  # Pac-Man
                return max_value(agentIndex, depth, gameState)
            else:  
                return exp_value(agentIndex, depth, gameState)

        def max_value(agentIndex, depth, gameState):
            v = -10000000000000
            bestAction = None
            for action in gameState.getLegalActions(agentIndex):
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                value = expectimax(1, depth, successorGameState)
                if value > v:
                    v = value
                    bestAction = action
            if depth == 0:  
                return bestAction
            return v

        def exp_value(agentIndex, depth, gameState):
            actions = gameState.getLegalActions(agentIndex)
            if len(actions) == 0: 
                return self.evaluationFunction(gameState)
            v = 0
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            nextDepth = depth if nextAgentIndex != 0 else depth + 1
            for action in actions:
                successorGameState = gameState.generateSuccessor(agentIndex, action)
                v += expectimax(nextAgentIndex, nextDepth, successorGameState)
            return v / len(actions)

        return expectimax(0, 0, gameState)  
    
        # agentIndex=0 means Pacman, ghosts are >= 1
        # legalMoves = gameState.getLegalActions(0)
        
        # # Choose one of the best actions
        # scores = [minimax(1, 0, gameState.generateSuccessor(0, action)) for action in legalMoves]
        # bestScore = max(scores)
        # bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # chosenIndex = random.choice(bestIndices)  
        
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
     # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood() #remaining food
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()

        # Distance to the closest food
    foodDistances = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
    closestFoodDistance = min(foodDistances) if foodDistances else 0

    num_food = len(newFood.asList())
    num_caps = len(newCapsules)

    # Adjust the ghost score to ensure Pacman avoids non-scared ghosts effectively
    ghostPenalty = 0
    ghost_zones = []
    danger = 0
    for i in newGhostStates:
        distance = manhattanDistance(newPos, i.getPosition())
        if i.scaredTimer > 0 and distance < 10:  
            ghostPenalty -= distance * 2  
        else:
            if distance < 3:
                ghostPenalty = -100000000    
    
    foodScore = -2 * closestFoodDistance  

    return currentGameState.getScore() + foodScore + ghostPenalty + danger - (num_food * 100) - (num_caps * 10000)

# Abbreviation
better = betterEvaluationFunction
