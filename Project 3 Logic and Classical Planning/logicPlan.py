# logicPlan.py
# ------------
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
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

from typing import Dict, List, Tuple, Callable, Generator, Any
import util
import sys
import logic
import game

from logic import conjoin, disjoin
from logic import PropSymbolExpr, Expr, to_cnf, pycoSAT, parseExpr, pl_true

import itertools
import copy

pacman_str = 'P'
food_str = 'FOOD'
wall_str = 'WALL'
pacman_wall_str = pacman_str + wall_str
DIRECTIONS = ['North', 'South', 'East', 'West']
blocked_str_map = dict([(direction, (direction + "_blocked").upper()) for direction in DIRECTIONS])
geq_num_adj_wall_str_map = dict([(num, "GEQ_{}_adj_walls".format(num)) for num in range(1, 4)])
DIR_TO_DXDY_MAP = {'North':(0, 1), 'South':(0, -1), 'East':(1, 0), 'West':(-1, 0)}


#______________________________________________________________________________
# QUESTION 1

def sentence1() -> Expr:
    """Returns a Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    A = Expr("A")
    B = Expr("B")
    C = Expr("C")

    sen_1 = A | B
    sen_2 = (~A)%((~B)| C)
    sen_3 = disjoin([(~A),(~B), C])

    return conjoin([sen_1, sen_2, sen_3])


def sentence2() -> Expr:
    """Returns a Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    A = Expr("A")
    B = Expr("B")
    C = Expr("C")
    D = Expr("D")

    sen_1 = C % disjoin([B, D])
    sen_2 = A >> conjoin([~B, ~D])
    sen_3 = ~conjoin([B, ~C]) >> A
    sen_4 = (~D) >> C

    return conjoin([sen_1, sen_2, sen_3, sen_4])


def sentence3() -> Expr:
    """Using the symbols PacmanAlive_1 PacmanAlive_0, PacmanBorn_0, and PacmanKilled_0,
    created using the PropSymbolExpr constructor, return a PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    Pacman is alive at time 1 if and only if [[Pacman was alive at time 0 and it was
    (not killed at time 0)] or it was [(not alive at time 0) and it was born at time 0]].

    Pacman cannot both be alive at time 0 and be born at time 0.

    Pacman is born at time 0.
    """
    
    PacmanAlive_0 = PropSymbolExpr('PacmanAlive', time = 0)
    PacmanAlive_1 = PropSymbolExpr('PacmanAlive', time = 1)
    PacmanBorn_0 = PropSymbolExpr("PacmanBorn", time = 0)
    PacmanKilled_0 = PropSymbolExpr("PacmanKilled", time = 0)

              

    sen_1 = PacmanAlive_1 % ((PacmanAlive_0 & ~PacmanKilled_0) | (~PacmanAlive_0 & PacmanBorn_0))
    sen_2 = ~(PacmanAlive_0 & PacmanBorn_0)
    sen_3 = PacmanBorn_0 

    return conjoin([sen_1, sen_2, sen_3])


def findModel(sentence: Expr) -> Dict[Expr, bool]:
    """Given a propositional logic sentence (i.e. a Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    cnf_sentence = to_cnf(sentence)
    return pycoSAT(cnf_sentence)

def findModelUnderstandingCheck() -> Dict[Expr, bool]:
    """Returns the result of findModel(Expr('a')) if lower cased expressions were allowed.
    You should not use findModel or Expr in this method.
    """
    a = Expr('A')
    print("a.__dict__ is:", a.__dict__) # might be helpful for getting ideas

    dict = a.__dict__ # >>> {'op': 'A', 'args': ()} 
    dict["op"] = dict["op"].lower()

    return {a: True}

def entails(premise: Expr, conclusion: Expr) -> bool:
    """Returns True if the premise entails the conclusion and False otherwise.
    """

    #print(type(findModel(premise)), ": " , findModel(premise) , "AND ", type(findModel(conclusion)), " : ", findModel(conclusion)) 

    # if type(findModel(premise)) != type(findModel(conclusion)):
    #     return False
    
    combined = premise & ~conclusion #iff alpha and not beta is false in all possible world
    
    # Find a model for the combined expression
    model = findModel(combined)
    
    return model == False

def plTrueInverse(assignments: Dict[Expr, bool], inverse_statement: Expr) -> bool:
    """Returns True if the (not inverse_statement) is True given assignments and False otherwise.
    pl_true may be useful here; see logic.py for its description.
    """

    # print(assignments, "BAHHH", inverse_statement)

    # print(assignments, "BAHHH2", inverse_statement.__dict__)

    return not pl_true(inverse_statement, assignments)


#______________________________________________________________________________
# QUESTION 2

def atLeastOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals (i.e. in the form A or ~A), return a single 
    Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals  ist is true.
    >>> A = PropSymbolExpr('A');
    >>> B = PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print(pl_true(atleast1,model1))
    False
    >>> model2 = {A:False, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    >>> model3 = {A:True, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    """
    return disjoin(literals)


def atMostOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    itertools.combinations may be useful here.
    """
    #at most one is true

    #print([_ for _ in itertools.combinations(literals, 2)])
          

    new_list = []
    for symbol1, symbol2 in itertools.combinations(literals, 2):

    #cannot both be true
        new_list.append(~symbol1 | ~symbol2) #if both is true >>> auto false

    return conjoin(new_list)



def exactlyOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    #exactly one is true

    new_list = []
    for symbol1, symbol2 in itertools.combinations(literals, 2):
        new_list.append(~symbol1 | ~symbol2) 

    return conjoin([conjoin(new_list), disjoin(literals)])

#______________________________________________________________________________
# QUESTION 3

def pacmanSuccessorAxiomSingle(x: int, y: int, time: int, walls_grid: List[List[bool]]=None) -> Expr:
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    now, last = time, time - 1
    possible_causes: List[Expr] = [] # enumerate all possible causes for P[x,y]_t
    # the if statements give a small performance boost and are required for q4 and q5 correctness
    if walls_grid[x][y+1] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x, y+1, time=last)
                            & PropSymbolExpr('South', time=last))
    if walls_grid[x][y-1] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x, y-1, time=last) 
                            & PropSymbolExpr('North', time=last))
    if walls_grid[x+1][y] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x+1, y, time=last) 
                            & PropSymbolExpr('West', time=last))
    if walls_grid[x-1][y] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x-1, y, time=last) 
                            & PropSymbolExpr('East', time=last))
    if not possible_causes:
        return None

    #A state variable X gets its value according to a successor-state axiom
    
    return PropSymbolExpr(pacman_str, x, y, time=now) % disjoin(possible_causes)
    


def SLAMSuccessorAxiomSingle(x: int, y: int, time: int, walls_grid: List[List[bool]]) -> Expr:
    """
    Similar to `pacmanSuccessorStateAxioms` but accounts for illegal actions
    where the pacman might not move timestep to timestep.
    Available actions are ['North', 'East', 'South', 'West']
    """
    now, last = time, time - 1
    moved_causes: List[Expr] = [] # enumerate all possible causes for P[x,y]_t, assuming moved to having moved
    if walls_grid[x][y+1] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x, y+1, time=last)
                            & PropSymbolExpr('South', time=last))
    if walls_grid[x][y-1] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x, y-1, time=last) 
                            & PropSymbolExpr('North', time=last))
    if walls_grid[x+1][y] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x+1, y, time=last) 
                            & PropSymbolExpr('West', time=last))
    if walls_grid[x-1][y] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x-1, y, time=last) 
                            & PropSymbolExpr('East', time=last))
    if not moved_causes:
        return None

    moved_causes_sent: Expr = conjoin([~PropSymbolExpr(pacman_str, x, y, time=last) , ~PropSymbolExpr(wall_str, x, y), disjoin(moved_causes)])

    failed_move_causes: List[Expr] = [] # using merged variables, improves speed significantly
    auxilary_expression_definitions: List[Expr] = []
    for direction in DIRECTIONS:
        dx, dy = DIR_TO_DXDY_MAP[direction]
        wall_dir_clause = PropSymbolExpr(wall_str, x + dx, y + dy) & PropSymbolExpr(direction, time=last)
        wall_dir_combined_literal = PropSymbolExpr(wall_str + direction, x + dx, y + dy, time=last)
        failed_move_causes.append(wall_dir_combined_literal)
        auxilary_expression_definitions.append(wall_dir_combined_literal % wall_dir_clause)

    failed_move_causes_sent: Expr = conjoin([
        PropSymbolExpr(pacman_str, x, y, time=last),
        disjoin(failed_move_causes)])

    return conjoin([PropSymbolExpr(pacman_str, x, y, time=now) % disjoin([moved_causes_sent, failed_move_causes_sent])] + auxilary_expression_definitions)


def pacphysicsAxioms(t: int, all_coords: List[Tuple], non_outer_wall_coords: List[Tuple], walls_grid: List[List] = None, sensorModel: Callable = None, successorAxioms: Callable = None) -> Expr:
    #generate a bunch of physics axioms, for timestep t
    """
    Given:
        t: timestep
        all_coords: list of (x, y) coordinates of the entire problem
        non_outer_wall_coords: list of (x, y) coordinates of the entire problem,
            excluding the outer border (these are the actual squares pacman can
            possibly be in)
        walls_grid: 2D array of either -1/0/1 or T/F. Used only for successorAxioms.
            Do NOT use this when making possible locations for pacman to be in.
        sensorModel(t, non_outer_wall_coords) -> Expr: function that generates
            the sensor model axioms. If None, it's not provided, so shouldn't be run.
        successorAxioms(t, walls_grid, non_outer_wall_coords) -> Expr: function that generates
            the sensor model axioms. If None, it's not provided, so shouldn't be run.
    Return a logic sentence containing all of the following:
        - for all (x, y) in all_coords:
            If a wall is at (x, y) --> Pacman is not at (x, y)
        - Pacman is at exactly one of the squares at timestep t. 
        - Pacman takes exactly one action at timestep t.
        - Results of calling sensorModel(...), unless None.
        - Results of calling successorAxioms(...), describing how Pacman can end in various
            locations on this time step. Consider edge cases. Don't call if None.
    """
    pacphysics_sentences = []

    #If a wall is at (x, y) --> Pacman is not at (x, y)
    for x, y in all_coords: #cannot both be true
        pacphysics_sentences.append(PropSymbolExpr(wall_str, x, y) >> ~PropSymbolExpr(pacman_str, x, y, time= t) )

    #-Pacman is at exactly one of the squares at timestep t
    new_list = []
    for x, y in non_outer_wall_coords: 
        new_list.append(PropSymbolExpr(pacman_str, x, y, time = t)) #all possible locations
    pacphysics_sentences.append(exactlyOne(new_list)) #only one at all possible locations

    #-Pacman takes exactly one action at timestep t.
    new_list2 = []
    for action in DIRECTIONS:
        new_list2.append(PropSymbolExpr(action, time = t)) #all possible action at time t
    pacphysics_sentences.append(exactlyOne(new_list2)) #only one possible action

    if sensorModel is not None:
        pacphysics_sentences.append(sensorModel(t, non_outer_wall_coords))

    if successorAxioms is not None and t!=0:
        pacphysics_sentences.append(successorAxioms(t, walls_grid, non_outer_wall_coords))
    
    return conjoin(pacphysics_sentences)


def checkLocationSatisfiability(x1_y1: Tuple[int, int], x0_y0: Tuple[int, int], action0, action1, problem):
    """
    Given:
        - x1_y1 = (x1, y1), a potential location at time t = 1
        - x0_y0 = (x0, y0), Pacman's location at time t = 0
        - action0 = one of the four items in DIRECTIONS, Pacman's action at time t = 0
        - action1 = to ensure match with autograder solution
        - problem = an instance of logicAgents.LocMapProblem
    Note:
        - there's no sensorModel because we know everything about the world
        - the successorAxioms should be allLegalSuccessorAxioms where needed
    Return:
        - a model where Pacman is at (x1, y1) at time t = 1
        - a model where Pacman is not at (x1, y1) at time t = 1
    """
    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))
    KB = []
    x0, y0 = x0_y0
    x1, y1 = x1_y1

    # We know which coords are walls:
    map_sent = [PropSymbolExpr(wall_str, x, y) for x, y in walls_list]
    KB.append(conjoin(map_sent))

    #Add to KB: Pacman’s current location 
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time = 0))

    #Add to KB: Pacman takes action0
    KB.append(PropSymbolExpr(action0, time = 0))
    KB.append(pacphysicsAxioms(0, all_coords, non_outer_wall_coords, walls_grid, None, allLegalSuccessorAxioms))


    #Add to KB: Pacman takes action1
    KB.append(PropSymbolExpr(action1, time = 1))
    KB.append(pacphysicsAxioms(1, all_coords, non_outer_wall_coords, walls_grid, None, allLegalSuccessorAxioms))



    #proves that it’s possible that Pacman there
    pacman_x1_y1_1 = PropSymbolExpr(pacman_str, x1, y1, time = 1)
    model1 = findModel(conjoin(KB) &  pacman_x1_y1_1)

    #possible that Pacman is not there
    model2 = findModel(conjoin(KB)  & ~pacman_x1_y1_1)

    return (model1, model2)

#______________________________________________________________________________
# QUESTION 4

def positionLogicPlan(problem) -> List:
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    Overview: add knowledge incrementally, and query for a model each timestep. Do NOT use pacphysicsAxioms.
    """
    walls_grid = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls_grid.asList()
    x0, y0 = problem.startState
    xg, yg = problem.goal
    
    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), 
            range(height + 2)))
    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = [ 'North', 'South', 'East', 'West' ]
    KB = []

    "*** BEGIN YOUR CODE HERE ***"
    #Add to KB: Initial knowledge: Pacman’s initial location at timestep 0
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time = 0))

    for t in range(50):
        #print(f"Time step: {t}")
        #can only be at exactlyOne of the locations in non_wall_coords at timestep t
        possible_positions = [PropSymbolExpr(pacman_str, x, y, time=t) for x, y in non_wall_coords]
        KB.append(exactlyOne(possible_positions))

        #Is there a satisfying assignment for the variables given the knowledge base so far
        goal_assertion = PropSymbolExpr(pacman_str, xg, yg, time=t)
        my_model = findModel(conjoin(KB) & goal_assertion)

        #return a sequence of actions from start to goal using extractActionSequence
        if my_model:
            return extractActionSequence(my_model, actions)
        #extractActionSequence(model: Dict[Expr, bool], actions: List) -> List:


        #Pacman takes exactly one action per timestep.
        #-Pacman takes exactly one action at timestep t.
        possible_actions = [PropSymbolExpr(action, time=t) for action in actions]
        KB.append(exactlyOne(possible_actions))

        #transition Model sentences: call pacmanSuccessorAxiomSingle(...) for all possible pacman positions in non_wall_coords.
        for x, y in non_wall_coords:
            KB.append(pacmanSuccessorAxiomSingle(x, y, t+1, walls_grid))

    return []

    "*** END YOUR CODE HERE ***"

#______________________________________________________________________________
# QUESTION 5

def foodLogicPlan(problem) -> List:
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    Overview: add knowledge incrementally, and query for a model each timestep. Do NOT use pacphysicsAxioms.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls.asList()
    (x0, y0), food = problem.start
    food = food.asList()

    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), range(height + 2)))

    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = [ 'North', 'South', 'East', 'West' ]

    KB = []

    "*** BEGIN YOUR CODE HERE ***"
    #Add to KB: Initial knowledge: Pacman’s initial location at timestep 0
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time = 0))

    #Initialize Food[x,y]_t 
    for x, y in all_coords:
        if (x, y) in food:
            KB.append(PropSymbolExpr(food_str, x, y, time=0))
        else:
            KB.append(~PropSymbolExpr(food_str, x, y, time=0))


    for t in range(50):
        #print(f"Time step: {t}")
        #can only be at exactlyOne of the locations in non_wall_coords at timestep t
        possible_positions = [PropSymbolExpr(pacman_str, x, y, time=t) for x, y in non_wall_coords]
        KB.append(exactlyOne(possible_positions))

        inp_food = []
        for x, y in food:
            inp_food.append(~PropSymbolExpr(food_str, x, y, time= t))

 #      #Is there a satisfying assignment for the variables given the knowledge base so far

        goal_assertion_food = conjoin([~PropSymbolExpr(food_str, x, y, time=t) for x, y in food])
        my_model = findModel(conjoin(KB) & goal_assertion_food)
        #my_model = findModel(conjoin(KB) & food_goal)

        if my_model:
#         #KB.append(~PropSymbolExpr(food_str, x, y, time=0)
            return extractActionSequence(my_model, actions)
        
         #Pacman takes exactly one action per timestep.
        possible_actions = [PropSymbolExpr(action, time=t) for action in actions]
        KB.append(exactlyOne(possible_actions))


        #transition Model sentences: call pacmanSuccessorAxiomSingle(...) for all possible pacman positions in non_wall_coords.
        for x, y in non_wall_coords:
            KB.append(pacmanSuccessorAxiomSingle(x, y, t+1, walls))

        # for x, y in food:
        #     

        for x, y in all_coords:
            if (x, y) in food:
                # If Pacman is here this timestep and there was food, then no food next timestep
                KB.append(~PropSymbolExpr(food_str, x, y, time=t+1) >> (PropSymbolExpr(pacman_str, x, y, time=t) & PropSymbolExpr(food_str, x, y, time=t))| ~PropSymbolExpr(food_str, x, y, time = t))
                # Food remains the same if not eaten by Pacman
                KB.append(PropSymbolExpr(food_str, x, y, time=t+1) >> PropSymbolExpr(food_str, x, y, time=t))

    return []
    "*** END YOUR CODE HERE ***"

#______________________________________________________________________________
# QUESTION 6

def localization(problem, agent) -> Generator:
    '''
    problem: a LocalizationProblem instance
    agent: a LocalizationLogicAgent instance
    '''
    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    KB = []

    "*** BEGIN YOUR CODE HERE ***"
    #Add to KB: where the walls are (walls_list) and aren’t (not in walls_list)
    for x, y in all_coords:
        if (x, y) in walls_list:
            KB.append(PropSymbolExpr(wall_str, x, y))
        else:
            KB.append(~PropSymbolExpr(wall_str, x, y))


    for t in range(agent.num_timesteps):
        #Add pacphysics, action, and percept information to KB.
        #Add to KB: pacphysics_axioms(...), which you wrote in q3. 
       
        KB.append(pacphysicsAxioms(t, all_coords, non_outer_wall_coords, walls_grid, None, allLegalSuccessorAxioms))
        # Use sensorAxioms and allLegalSuccessorAxioms for localization and mapping
        KB.append(sensorAxioms(t, non_outer_wall_coords))
        #Add to KB: Pacman takes action prescribed by agent.actions[t]
        KB.append(PropSymbolExpr(agent.actions[t], time=t))
        #Get the percepts by calling agent.getPercepts() and pass the percepts to fourBitPerceptRules(...) for localization and mapping,
        KB.append(fourBitPerceptRules(t, agent.getPercepts()))

        #Find possible pacman locations with updated KB.
        possible_locations = []
        for x, y in non_outer_wall_coords:
            pacman_at_xy = PropSymbolExpr(pacman_str, x, y, time=t)
 
            if entails(conjoin(KB), pacman_at_xy):
                KB.append(pacman_at_xy)
                #possible_locations.append((x, y))
    
            if entails(conjoin(KB), ~pacman_at_xy):
                KB.append(~pacman_at_xy)
            else:
                possible_locations.append((x, y))


        #Call agent.moveToNextState(action_t) on the current agent action at timestep t. yield the possible locations.
        agent.moveToNextState(agent.actions[t])
        "*** END YOUR CODE HERE ***"
      
        yield possible_locations

#______________________________________________________________________________
# QUESTION 7

def mapping(problem, agent) -> Generator:
    '''
    problem: a MappingProblem instance
    agent: a MappingLogicAgent instance
    '''
    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight()+2)] for x in range(problem.getWidth()+2)]

    # Pacman knows that the outer border of squares are all walls
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    "*** BEGIN YOUR CODE HERE ***"
    #Get initial location (pac_x_0, pac_y_0)
    KB.append(PropSymbolExpr(pacman_str, pac_x_0, pac_y_0, time = 0))
    #whether there is a wall at that location.
    for t in range(agent.num_timesteps):
        #Add pacphysics, action, and percept information to KB.       
        KB.append(pacphysicsAxioms(t, all_coords, non_outer_wall_coords, known_map, None, allLegalSuccessorAxioms))
        KB.append(sensorAxioms(t, non_outer_wall_coords))
        KB.append(PropSymbolExpr(agent.actions[t], time=t))
        KB.append(fourBitPerceptRules(t, agent.getPercepts()))

        #Find provable wall locations with updated KB
        #Iterate over non_outer_wall_coords
        for x, y in non_outer_wall_coords:
            wall_xy = PropSymbolExpr(wall_str, x, y)
            #Add to KB and update known_map: (x,y) locations where there is provably a wall.
            if entails(conjoin(KB), wall_xy):
                KB.append(wall_xy)
                known_map[x][y] = 1
            #Add to KB and update known_map: (x,y) locations where there is provably not a wall.
            if entails(conjoin(KB), ~wall_xy):
                KB.append(~wall_xy)
                known_map[x][y] = 0

        agent.moveToNextState(agent.actions[t]) 
        yield known_map

#______________________________________________________________________________
# QUESTION 8

def slam(problem, agent) -> Generator:
    '''
    problem: a SLAMProblem instance
    agent: a SLAMLogicAgent instance
    '''
    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight()+2)] for x in range(problem.getWidth()+2)]

    # We know that the outer_coords are all walls.
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    "*** BEGIN YOUR CODE HERE ***"
    KB.append(PropSymbolExpr(pacman_str, pac_x_0, pac_y_0, time = 0))


    for t in range(agent.num_timesteps):
        KB.append(pacphysicsAxioms(t, all_coords, non_outer_wall_coords, known_map, None, SLAMSuccessorAxioms))
        KB.append(SLAMSensorAxioms(t, non_outer_wall_coords))
        KB.append(PropSymbolExpr(agent.actions[t], time=t))
        KB.append(numAdjWallsPerceptRules(t, agent.getPercepts()))

        #Find provable wall locations with updated KB
        for x, y in non_outer_wall_coords:
            wall_xy = PropSymbolExpr(wall_str, x, y)
            #Add to KB and update known_map: (x,y) locations where there is provably a wall.
            if entails(conjoin(KB), wall_xy):
                KB.append(wall_xy)
                known_map[x][y] = 1
            #Add to KB and update known_map: (x,y) locations where there is provably not a wall.
            if entails(conjoin(KB), ~wall_xy):
                KB.append(~wall_xy)
                known_map[x][y] = 0

        possible_locations = []
        for x, y in non_outer_wall_coords:
            pacman_at_xy = PropSymbolExpr(pacman_str, x, y, time=t)
 
            if entails(conjoin(KB), pacman_at_xy):
                KB.append(pacman_at_xy)
                #possible_locations.append((x, y))
    
            if entails(conjoin(KB), ~pacman_at_xy):
                KB.append(~pacman_at_xy)
            else:
                possible_locations.append((x, y))
        

        agent.moveToNextState(agent.actions[t]) 
        yield (known_map, possible_locations)


# Abbreviations
plp = positionLogicPlan
loc = localization
mp = mapping
flp = foodLogicPlan
# Sometimes the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)

#______________________________________________________________________________
# Important expression generating functions, useful to read for understanding of this project.


def sensorAxioms(t: int, non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, x + dx, y + dy, time=t)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (
                PropSymbolExpr(pacman_str, x, y, time=t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], time = t)
        all_percept_exprs.append(percept_unit_clause % disjoin(percept_exprs))

    return conjoin(all_percept_exprs + combo_var_def_exprs)


def fourBitPerceptRules(t: int, percepts: List) -> Expr:
    """
    Localization and Mapping both use the 4 bit sensor, which tells us True/False whether
    a wall is to pacman's north, south, east, and west.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 4, "Percepts must be a length 4 list."

    percept_unit_clauses = []
    for wall_present, direction in zip(percepts, DIRECTIONS):
        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], time=t)
        if not wall_present:
            percept_unit_clause = ~PropSymbolExpr(blocked_str_map[direction], time=t)
        percept_unit_clauses.append(percept_unit_clause) # The actual sensor readings
    return conjoin(percept_unit_clauses)


def numAdjWallsPerceptRules(t: int, percepts: List) -> Expr:
    """
    SLAM uses a weaker numAdjWallsPerceptRules sensor, which tells us how many walls pacman is adjacent to
    in its four directions.
        000 = 0 adj walls.
        100 = 1 adj wall.
        110 = 2 adj walls.
        111 = 3 adj walls.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 3, "Percepts must be a length 3 list."

    percept_unit_clauses = []
    for i, percept in enumerate(percepts):
        n = i + 1
        percept_literal_n = PropSymbolExpr(geq_num_adj_wall_str_map[n], time=t)
        if not percept:
            percept_literal_n = ~percept_literal_n
        percept_unit_clauses.append(percept_literal_n)
    return conjoin(percept_unit_clauses)


def SLAMSensorAxioms(t: int, non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, x + dx, y + dy, time=t)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (PropSymbolExpr(pacman_str, x, y, time=t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        blocked_dir_clause = PropSymbolExpr(blocked_str_map[direction], time=t)
        all_percept_exprs.append(blocked_dir_clause % disjoin(percept_exprs))

    percept_to_blocked_sent = []
    for n in range(1, 4):
        wall_combos_size_n = itertools.combinations(blocked_str_map.values(), n)
        n_walls_blocked_sent = disjoin([
            conjoin([PropSymbolExpr(blocked_str, time=t) for blocked_str in wall_combo])
            for wall_combo in wall_combos_size_n])
        # n_walls_blocked_sent is of form: (N & S) | (N & E) | ...
        percept_to_blocked_sent.append(
            PropSymbolExpr(geq_num_adj_wall_str_map[n], time=t) % n_walls_blocked_sent)

    return conjoin(all_percept_exprs + combo_var_def_exprs + percept_to_blocked_sent)


def allLegalSuccessorAxioms(t: int, walls_grid: List[List], non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    """walls_grid can be a 2D array of ints or bools."""
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = pacmanSuccessorAxiomSingle(
            x, y, t, walls_grid)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)


def SLAMSuccessorAxioms(t: int, walls_grid: List[List], non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    """walls_grid can be a 2D array of ints or bools."""
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = SLAMSuccessorAxiomSingle(
            x, y, t, walls_grid)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)

#______________________________________________________________________________
# Various useful functions, are not needed for completing the project but may be useful for debugging


def modelToString(model: Dict[Expr, bool]) -> str:
    """Converts the model to a string for printing purposes. The keys of a model are 
    sorted before converting the model to a string.
    
    model: Either a boolean False or a dictionary of Expr symbols (keys) 
    and a corresponding assignment of True or False (values). This model is the output of 
    a call to pycoSAT.
    """
    if model == False:
        return "False" 
    else:
        # Dictionary
        modelList = sorted(model.items(), key=lambda item: str(item[0]))
        return str(modelList)


def extractActionSequence(model: Dict[Expr, bool], actions: List) -> List:
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[2]":True, "P[3,4,0]":True, "P[3,3,0]":False, "West[0]":True, "GhostScary":True, "West[2]":False, "South[1]":True, "East[0]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print(plan)
    ['West', 'South', 'North']
    """
    plan = [None for _ in range(len(model))]
    for sym, val in model.items():
        parsed = parseExpr(sym)
        if type(parsed) == tuple and parsed[0] in actions and val:
            action, _, time = parsed
            plan[time] = action
    #return list(filter(lambda x: x is not None, plan))
    return [x for x in plan if x is not None]


# Helpful Debug Method
def visualizeCoords(coords_list, problem) -> None:
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    for (x, y) in itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)):
        if (x, y) in coords_list:
            wallGrid.data[x][y] = True
    print(wallGrid)


# Helpful Debug Method
def visualizeBoolArray(bool_arr, problem) -> None:
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    wallGrid.data = copy.deepcopy(bool_arr)
    print(wallGrid)

class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()
        
    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()
