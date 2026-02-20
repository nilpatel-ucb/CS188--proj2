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
        # Main take away is for each new position calculate a fxn which rewards or doesn't reward an action,
        #have it pick the max action based on the eval function

        # evalulates actions rather than state
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newFoodGrid = newFood.asList()
        # manhattanDistance(new)
        newGhostStates = successorGameState.getGhostStates()
        # min distance to ghost
        minGhostDistance = []
        for i in newGhostStates:
            minGhostDistance.append([manhattanDistance(newPos, i.getPosition()), i.getPosition()])
        
        minFoodCoordiante = []
        
        for i in newFoodGrid:
            minFoodCoordiante.append([ manhattanDistance(newPos, i), i])

        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        if len(minFoodCoordiante) == 0:
            return successorGameState.getScore()
        minimum = min(minFoodCoordiante)

        "*** YOUR CODE HERE ***"
       
        if len(minGhostDistance) == 0:
            return successorGameState.getScore() -  manhattanDistance(newPos, minimum[1])
        minimumGhost = min(minGhostDistance)

        if minimumGhost[0] <= 2:
            return -100
        return successorGameState.getScore() -  manhattanDistance(newPos, minimum[1]) + manhattanDistance(newPos, minimumGhost[1])

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
        "*** YOUR CODE HERE ***"

        # how do we keep track of depth?
        # how does recursssion work

        totalNumOfAgents = gameState.getNumAgents()
        actionToReturn = None
        num = float('-inf')

        # for theAction in gameState.getLegalActions(0):
        #     succssor = gameState.generateSuccessor(0, theAction)

        #     val = value(succssor, 0, 1)
        #     if val > num or actionToReturn is None:
        #         actionToReturn = theAction
        #         num = val
        def value(state, depth, numOfAgent):
            
            # listOfActions = gameState.getLegalActions(gameState)
            # allSuccessors = []
            # for action in listOfActions:
            #     allSuccessors.append(gameState.generateSuccessor(gameState, action))

            if state.isWin() == True or state.isLose() == True or self.depth == depth:
                return self.evaluationFunction(state)
            if numOfAgent == 0:
                return maxValueOfState(state, depth)
            elif numOfAgent > 0:
                return minValueOfState(state, depth, numOfAgent)
            
            
        #write down maxState
        def maxValueOfState(state, depth):
            listOfActions = state.getLegalActions(0)
            num = float('-inf')

            #so we don't get any index errors
            if not listOfActions:
                return self.evaluationFunction(state)
            for theAction in listOfActions:
                #not evaluating actions, we are evaluating states
                # num = max(num, minValueOfState(succ))
                succ = state.generateSuccessor(0,theAction)
                # need to pass in value, as this is what will allow us to make a list of all 
                # min values for us to choose from at the maxLevel
                num = max(num, value(succ, depth, 1))
            return num
        #write down minState
        def minValueOfState(state, depth, numOfAgent):
            num =  float('inf')
            listOfActions = state.getLegalActions(numOfAgent)
            if not listOfActions:
                return self.evaluationFunction(state)
            lastAgent = totalNumOfAgents - 1
            # for succ in allSuccessors:

            #     num = min(num, maxValueOfState(succ))

            for action in listOfActions:
                successor = state.generateSuccessor(numOfAgent, action)

                if numOfAgent == lastAgent:
                    #send it back to pacman if we are on last agent
                    num = min(num, value(successor, depth + 1, 0))
                else:
                    num = min(num, value(successor, depth, numOfAgent + 1))
            return num
        
        for theAction in gameState.getLegalActions(0):
            succssor = gameState.generateSuccessor(0, theAction)

            val = value(succssor, 0, 1)
            if val > num or actionToReturn is None:
                actionToReturn = theAction
                num = val
        return actionToReturn





        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        totalNumOfAgents = gameState.getNumAgents()
        actionToReturn = None
        num = float('-inf')
        

        # for theAction in gameState.getLegalActions(0):
        #     succssor = gameState.generateSuccessor(0, theAction)

        #     val = value(succssor, 0, 1)
        #     if val > num or actionToReturn is None:
        #         actionToReturn = theAction
        #         num = val
        def value(state, depth, numOfAgent, alpha, beta):
            
            # listOfActions = gameState.getLegalActions(gameState)
            # allSuccessors = []
            # for action in listOfActions:
            #     allSuccessors.append(gameState.generateSuccessor(gameState, action))

            if state.isWin() == True or state.isLose() == True or self.depth == depth:
                return self.evaluationFunction(state)
            if numOfAgent == 0:
                return maxValueOfState(state, depth, alpha, beta)
            elif numOfAgent > 0:
                return minValueOfState(state, depth, numOfAgent, alpha, beta)
            
            
        #write down maxState
        def maxValueOfState(state, depth, alpha, beta):
            listOfActions = state.getLegalActions(0)
            num = float('-inf')

            #so we don't get any index errors
            if not listOfActions:
                return self.evaluationFunction(state)
            for theAction in listOfActions:
                #not evaluating actions, we are evaluating states
                # num = max(num, minValueOfState(succ))
                succ = state.generateSuccessor(0,theAction)
                # need to pass in value, as this is what will allow us to make a list of all 
                # min values for us to choose from at the maxLevel
                num = max(num, value(succ, depth, 1, alpha, beta))
                if num > beta:
                    return num
                alpha = max(alpha, num)
            return num
        #write down minState
        def minValueOfState(state, depth, numOfAgent, alpha, beta):
            num =  float('inf')
            listOfActions = state.getLegalActions(numOfAgent)
            if not listOfActions:
                return self.evaluationFunction(state)
            lastAgent = totalNumOfAgents - 1
            # for succ in allSuccessors:

            #     num = min(num, maxValueOfState(succ))

            for action in listOfActions:
                successor = state.generateSuccessor(numOfAgent, action)

                if numOfAgent == lastAgent:
                    #send it back to pacman if we are on last agent
                    num = min(num, value(successor, depth + 1, 0, alpha, beta))
                    if num < alpha:
                        return num
                    beta = min(beta, num)
                else:

                    num = min(num, value(successor, depth, numOfAgent + 1, alpha, beta))
                    if num < alpha:
                        return num
                    beta = min(beta, num)
            return num
        alpha = float('-inf')
        beta= float('inf')
        for theAction in gameState.getLegalActions(0):
            succssor = gameState.generateSuccessor(0, theAction)

            val = value(succssor, 0, 1, alpha, beta)
            if val > num or actionToReturn is None:
                actionToReturn = theAction
                num = val
            alpha = max(num,alpha )
        return actionToReturn


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
