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
import random
import util

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
        #Collect allowed moves and successor states
        legalMoves = gameState.getLegalActions()

        #Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        #Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

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
        #Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        food_weight = 10
        ghost_weight = 30
        scared_weight = 20
        foodPositions = newFood.asList()
        if foodPositions:
            nearestFood = min([manhattanDistance(newPos, food)
                              for food in foodPositions])
            score += food_weight / max(1, nearestFood)
        for ghostState in newGhostStates:
            ghostDistance = manhattanDistance(newPos, ghostState.getPosition())
            if ghostState.scaredTimer > 0:
                score += scared_weight / max(1, ghostDistance)
            else:
                if ghostDistance == 0:
                    return float("-inf")
                score -= ghost_weight/ghostDistance
        return score


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  #Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def value(self, state, agentIndex, depth):
        #stops at any terminal node whether it has been won already or not
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        legalActions = state.getLegalActions(agentIndex)

        #next player in order, wraps back to pacman after the last ghost
        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = depth
        #a full round is pacman plus every ghost, so only bump depth when pacman goes again
        if nextAgent == 0:
            nextDepth += 1

        if agentIndex == 0:
            #(pacman)best val path
            v = float("-inf")
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = max(v, self.value(successor, nextAgent, nextDepth))
            return v
        else:
            #(ghosts)take the worst value path
            v = float("inf")
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                v = min(v, self.value(successor, nextAgent, nextDepth))
            return v

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
        bestValue = float("-inf")
        bestAction = None
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            #pacman already moved -> ghost turn now
            value = self.value(successor, 1, 0)
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #alpha is the best pacman can already force, beta is the worst a ghost will still allow
        alpha = float("-inf")
        beta = float("inf")
        bestScore = float("-inf")
        bestMove = None
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            #pacman turn done, so ghost moves in round now
            if gameState.getNumAgents() == 1:
                score = self.maxValue(nextState, 1, alpha, beta)
            else:
                score = self.minValue(nextState, 1, 0, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestMove = action
            #later root moves can get cut if they can't beat what pacman already found
            alpha = max(alpha, bestScore)
        return bestMove

    def maxValue(self, state, depth, alpha, beta):
        #game is over, or we've already looked ahead self.depth rounds
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        score = float("-inf")
        for action in state.getLegalActions(0):
            nextState = state.generateSuccessor(0, action)
            if state.getNumAgents() == 1:
                nextScore = self.maxValue(nextState, depth + 1, alpha, beta)
            else:
                #ghosts still move before this round is finished
                nextScore = self.minValue(nextState, 1, depth, alpha, beta)
            score = max(score, nextScore)
            if score > beta:
                return score
            alpha = max(alpha, score)
        return score

    def minValue(self, state, ghostIndex, depth, alpha, beta):
        #a ghost can also land on a win, a loss, or the depth limit
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        #start above every real score so the first move can take over
        score = float("inf")
        lastGhost = ghostIndex == state.getNumAgents() - 1
        for action in state.getLegalActions(ghostIndex):
            nextState = state.generateSuccessor(ghostIndex, action)
            if lastGhost:
                #ghosts turns done, now pacman turn
                nextScore = self.maxValue(nextState, depth + 1, alpha, beta)
            else:
                nextScore = self.minValue(nextState, ghostIndex + 1, depth, alpha, beta)
            #ghost picks whatever hurts pacman more
            score = min(score, nextScore)
            #if pacman already has better move, get rid of this branch
            if score < alpha:
                return score
            beta = min(beta, score)
        return score


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
        bestScore = float("-inf")
        bestMove = None
        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            #pacman already moved, so the ghosts answer randomly inside this round
            if gameState.getNumAgents() == 1:
                score = self.maxValue(nextState, 1)
            else:
                score = self.averageValue(nextState, 1, 0)
            if score > bestScore:
                bestScore = score
                bestMove = action
        return bestMove

    def maxValue(self, state, depth):
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        score = float("-inf")
        for action in state.getLegalActions(0):
            nextState = state.generateSuccessor(0, action)
            if state.getNumAgents() == 1:
                nextScore = self.maxValue(nextState, depth + 1)
            else:
                #ghosts still move before current round finished, but randomly
                nextScore = self.averageValue(nextState, 1, depth)
            score = max(score, nextScore)
        return score

    def averageValue(self, state, ghostIndex, depth):
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)

        #each legal ghost move equally likely--> add them up and divide
        actions = state.getLegalActions(ghostIndex)
        total = 0
        lastGhost = ghostIndex == state.getNumAgents() - 1
        for action in actions:
            nextState = state.generateSuccessor(ghostIndex, action)
            if lastGhost:
                #all ghost turns done, so the next round pacman's
                total += self.maxValue(nextState, depth + 1)
            else:
                total += self.averageValue(nextState, ghostIndex + 1, depth)
        return total/len(actions)


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Starts from the game score. closer food scores higher and
    leftover food hurts, and a nearby capsule should be grabbed. Chase a
    scared ghost only if pacman can reach it before the timer runs out.
    otherwise stay away, especially when a ghost is right next to pacman.
    """
    "*** YOUR CODE HERE ***"
    #start from current score
    score = currentGameState.getScore()
    pos = currentGameState.getPacmanPosition()
    foodPositions = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()

    food_weight = 10
    food_left_weight = 4
    capsule_weight = 15
    ghost_weight = 30
    scared_weight = 100

    if foodPositions:
        nearestFood = min([manhattanDistance(pos, food)
                          for food in foodPositions])
        #closer food is worth more
        score += food_weight / max(1, nearestFood)
        #leftover food is decentivized
        score -= food_left_weight * len(foodPositions)

    if capsules:
        nearestCapsule = min([manhattanDistance(pos, capsule)
                             for capsule in capsules])
        score += capsule_weight/ max(1, nearestCapsule)

    for ghostState in currentGameState.getGhostStates():
        ghostDistance = manhattanDistance(pos, ghostState.getPosition())
        #only chase if pacman can get there before the ghost stops getting scared
        if ghostState.scaredTimer > ghostDistance:
            score += scared_weight/ max(1, ghostDistance)
        else:
            #standing next to a normal ghost->  a loss
            if ghostDistance <= 1:
                score -= 500
            else:
                #farther ghosts ar weighted less
                score -= ghost_weight /ghostDistance
    return score


#Abbreviation
better = betterEvaluationFunction
