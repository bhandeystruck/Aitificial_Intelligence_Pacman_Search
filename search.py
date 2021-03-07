# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # need to initialize the variables

    # nodesVisited will contain nodes that are popped back from the stack,
    # it will also hold the directions from which they were obtained
    nodesVisited = {}

    # varaible solitonSteps will have the directions pacman needs to reach the goal state
    solutionSteps = []

    # variable stack will have triplets of node, direction and cost
    stack = util.Stack()

    # need a variable to keep track of parents of nodes
    # need variable to store nodes as well
    parents = {}

    # so we need to add the initial state into the stack
    start = problem.getStartState()
    # push the starting node , the direction state, and cost
    stack.push((start, 'Undefined', 0))

    # set the direction we just came from in the starting state to undefined
    nodesVisited[start] = 'Undefined'

    # check the case for  if the starting state is itself the goal state or not
    if problem.isGoalState(start):
        return solutionSteps

    # need a flag for goal
    goal = False

    # so until stack is not empty and goal is not reaching we loop
    while (stack.isEmpty() == False and goal != True):
        # pop a node from the start of the stack
        node = stack.pop()

        # store the element and direction
        nodesVisited[node[0]] = node[1]

        # see if the element is infact the goal
        if problem.isGoalState(node[0]):
            solutionNode = node[0]
            goal = True
            break

        # now we need to expand the node
        # check for visited successors and its parents and push it to the stack
        for elem in problem.getSuccessors(node[0]):
            # if successor has not already been visited
            if elem[0] not in nodesVisited.keys():
                # store successor and its parent
                parents[elem[0]] = node[0]
                # push successor onto stack
                stack.push(elem)

    # at last we need to find the path and store it
    while (solutionNode in parents.keys()):
        # find parent
        previousNodeSolution = parents[solutionNode]
        # prepend direction to solution
        solutionSteps.insert(0, nodesVisited[solutionNode])
        # go to previous node
        solutionNode = previousNodeSolution

    return solutionSteps

    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # need to initialize the variables

    # nodesVisited will contain nodes that are popped back from the stack,
    # it will also hold the directions from which they were obtained
    nodesVisited = {}

    # varaible solitonSteps will have the directions pacman needs to reach the goal state
    solutionSteps = []

    # Queue to hold  triplets of node, direction and cost
    queue = util.Queue()

    # need a variable to keep track of parents of nodes
    # need variable to store nodes as well
    parents = {}

    # so we need to add the initial state into the stack
    start = problem.getStartState()

    # push starting state to the queue
    queue.push((start, 'Undefined', 0))

    # set the direction we just came from in the starting state to undefined
    nodesVisited[start] = 'Undefined'

    # return if start state itself is the goal
    if problem.isGoalState(start):
        return solutionSteps

        # loop while queue is not empty and goal is not reached
    goal = False;
    while (queue.isEmpty() == False and goal != True):
        # pop from top of queue
        node = queue.pop()
        # store element , direction
        nodesVisited[node[0]] = node[1]
        # check if element is goal or not
        if problem.isGoalState(node[0]):
            solutionNode = node[0]
            goal = True
            break
        # expand node
        for elem in problem.getSuccessors(node[0]):
            # if successor has not already been visited
            if elem[0] not in nodesVisited.keys() and elem[0] not in parents.keys():
                # store successor and its parent
                parents[elem[0]] = node[0]
                # push successor onto queue
                queue.push(elem)

    # finding and storing the path
    while (solutionNode in parents.keys()):
        # find parent
        node_sol_prev = parents[solutionNode]
        # prepend direction to solution
        solutionSteps.insert(0, nodesVisited[solutionNode])
        # go to previous node
        solutionNode = node_sol_prev

    return solutionSteps

    util.raiseNotDefined()


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # initializations

    # "visited" contains nodes which have been popped from the queue,
    # and the direction from which they were obtained
    nodeVisited = {}
    # "solution" contains the sequence of directions for Pacman to get to the goal state
    solutionSteps = []
    # "queue" contains triplets of: (nodes in the fringe list, direction, cost)
    queue = util.PriorityQueue()
    # "parents" contains nodes and their parents
    parents = {}
    # "cost" contains nodes and their corresponding costs
    cost = {}

    # start state is obtained and added to the queue
    start = problem.getStartState()
    queue.push((start, 'Undefined', 0), 0)
    # the direction from which we arrived in the start state is undefined
    nodeVisited[start] = 'Undefined'
    # cost of start state is 0
    cost[start] = 0

    # return if start state itself is the goal
    if problem.isGoalState(start):
        return solutionSteps

    # loop while queue is not empty and goal is not reached
    goal = False;
    while queue.isEmpty() == False and goal == False:
        # pop from top of queue
        node = queue.pop()
        # store element and its direction
        nodeVisited[node[0]] = node[1]
        # check if element is goal
        if problem.isGoalState(node[0]):
            node_sol = node[0]
            goal = True
            break
        # expand node
        for elem in problem.getSuccessors(node[0]):
            # if successor is not visited, calculate its new cost
            if elem[0] not in nodeVisited.keys():
                priority = node[2] + elem[2]
                # if cost of successor was calculated earlier while expanding a different node,
                # if new cost is more than old cost, continue
                if elem[0] in cost.keys():
                    if cost[elem[0]] <= priority:
                        continue
                # if new cost is less than old cost, push to queue and change cost and parent
                queue.push((elem[0], elem[1], priority), priority)
                cost[elem[0]] = priority
                # store successor and its parent
                parents[elem[0]] = node[0]

    # finding and storing the path
    while (node_sol in parents.keys()):
        # find parent
        node_sol_prev = parents[node_sol]
        # prepend direction to solution
        solutionSteps.insert(0, nodeVisited[node_sol])
        # go to previous node
        node_sol = node_sol_prev

    return solutionSteps

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # initializations

    # "visited" contains nodes which have been popped from the queue,
    # and the direction from which they were obtained
    visited = {}
    # "solution" contains the sequence of directions for Pacman to get to the goal state
    solution = []
    # "queue" contains triplets of: (node in the fringe list, direction, cost)
    queue = util.PriorityQueue()
    # "parents" contains nodes and their parents
    parents = {}
    # "cost" contains nodes and their corresponding costs
    cost = {}

    # start state is obtained and added to the queue
    start = problem.getStartState()
    queue.push((start, 'Undefined', 0), 0)
    # the direction from which we arrived in the start state is undefined
    visited[start] = 'Undefined'
    # cost of start state is 0
    cost[start] = 0

    # return if start state itself is the goal
    if problem.isGoalState(start):
        return solution

    # loop while queue is not empty and goal is not reached
    goal = False;
    while queue.isEmpty() == False and goal == False:
        # pop from top of queue
        node = queue.pop()
        # store element and its direction
        visited[node[0]] = node[1]
        # check if element is goal
        if problem.isGoalState(node[0]):
            node_sol = node[0]
            goal = True
            break
        # expand node
        for elem in problem.getSuccessors(node[0]):
            # if successor is not visited, calculate its new cost
            if elem[0] not in visited.keys():
                priority = node[2] + elem[2] + heuristic(elem[0], problem)
                # if cost of successor was calculated earlier while expanding a different node,
                # if new cost is more than old cost, continue
                if elem[0] in cost.keys():
                    if cost[elem[0]] <= priority:
                        continue
                # if new cost is less than old cost, push to queue and change cost and parent
                queue.push((elem[0], elem[1], node[2] + elem[2]), priority)
                cost[elem[0]] = priority
                # store successor and its parent
                parents[elem[0]] = node[0]

    # finding and storing the path
    while node_sol in parents.keys():
        # find parent
        prev_solution = parents[node_sol]
        # prepend direction to solution
        solution.insert(0, visited[node_sol])
        # go to previous node
        node_sol = prev_solution

    return solution

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
