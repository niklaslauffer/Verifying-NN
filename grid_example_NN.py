from bresenham import bresenham
from itertools import product
import pickle

import numpy as np

def isNonVisible(x,y, agX, agY, obstacles, nrows, ncols):
    """ Check if an object at (x,y) can see the agent at (agX,agY)
        with the presence of obstacles
        Return True if the object can not see the agent and False otherwise
    """
    res = set(bresenham(agX, agY, x, y))
    for objPos in obstacles:
        (o1, o2) = objPos % nrows, objPos / nrows
        if (o1,o2) in res:
            return True
    return False


def nextStateAgent(currState, act, nrows, ncols):
    """ Generate the next state of an agent based on the current action
        act \in [0,1,2,3] -> North, south, west, east
    """
    currRow = currState / nrows
    currCol = currState % nrows
    if act == 0:
        return min((currRow+1), nrows-1)*nrows + currCol
    if act == 1:
        return max((currRow-1), 0)*nrows + currCol
    if act == 2:
        return currRow*nrows + max(currCol-1,0)
    if act == 3:
        return currRow*nrows + min(currCol+1,ncols-1)

def build_mdp_nn(nrows, ncols, actionList, objStates, objectTrans, obstacles):
    """ Build the MDP associated with a given combinaison of types.
        @ObjStates: dict where the key are identifiant for the objects
                    and the value is a list of the state of the objecy
        @ObjTrans: A function providing the next state and the probability of the
                   associated transitio given the its current state and the agent
                   current state. This function specifies the behavior of each
                   object -> Henceforth the current type of each object
    """
    # First, we build the product of state space of the agent and objects
    statesAg = [j+ i*nrows for i in range(nrows) for j in range(ncols)]
    listState = [statesAg]
    indexVal = 1
    dictIndex = dict()
    for objId, objState in objStates.items():
        listState.append(objState)
        dictIndex[objId] = indexVal
        indexVal+=1
    stateSpace = list(product(*listState)) # Here we buold the product
    # We iterate through the product set to build the MDP transition matrix
    dictTrans = dict()
    for state in stateSpace:
        currStateAg = state[0]
        if currStateAg in obstacles: # The agent should not go through the obstacles
            continue
        for a in actionList:
            # Get the next state of the agent given the current action a
            nextStateAg = nextStateAgent(currStateAg, a, nrows, ncols)
            if nextStateAg in obstacles:
                nextStateAg = currStateAg #No transition if next state is an obstacle
            # Obtain the next states+ prob of each objects in the environment
            transProb = objectTrans(dictIndex, state, a, objStates, nrows, ncols)
            newTransProb = {}
            for (nextObj, prob) in transProb:
                nextStateC = tuple([nextStateAg] + list(nextObj))
                newTransProb[nextStateC] = prob
            dictTrans[(state, a)] = newTransProb
    return dictTrans, stateSpace

def transType(dictIndex, state, a, objStates, nrows, ncols, distMin=1, prob=[]):
    """ Define the type and the probabilities of transitions of each objects
        This function assumes the objects move either on the horizon or vertical
        axis. The type is basically specified by the probabilities it moves left
        or right, or up and down when the agent is getting closer
        The proximity threshold can be changed with distMIn
    """
    # Get the agent state
    agState =  state[0]
    agRow =  agState / nrows
    agCol = agState % nrows
    nextStates = list()
    for objId, objState in objStates.items(): # Go through all the objects
        objStateGrid = state[dictIndex[objId]]
        objRow = objStateGrid / nrows
        objCol = objStateGrid % nrows
        # Compute the distance between the object and obstacle
        distobj = abs(objRow - agRow) + abs(objCol - agCol)
        # Compute the possible next states of the objects and check if they
        # are in the corresponding state space
        next1, next2 = min(objCol+1, ncols-1) + objRow*nrows, max(objCol-1, 0)+ objRow*nrows
        next3, next4 = objCol+min(objRow+1,nrows-1)*nrows, objCol+max(objRow-1,0)*nrows
        if next1 not in objState:
            next1, next2 = next3, next4
        # If the agent is far from the object he transists left/right or up/ down  with prob 0.5
        prob1, prob2 = 0.5, 0.5
        if distobj <= distMin: # If not the transition is given by the parameter prob
            prob1 = prob[objId]
            prob2 = 1 - prob[objId]
        # We save the possible next states and the probability of each transitions
        nextStates.append([(next1, prob1), (next2, prob2)])
    # We build the resulting transiitons prob for all possible combinaisons of
    # the next state of each object
    resultProb =  list(product(*nextStates))
    listRes = list()
    for elem in resultProb:
        probVal = 1.0
        nextL = list()
        for (n1, p1) in elem:
            probVal *= p1
            nextL.append(n1)
        listRes.append((tuple(nextL), probVal))
    return listRes

def buildSetMDPs(nrows, ncols, alphabet, objStates, typeObjects, obstacles, obserState, distMin=1):
    """ Build the set of MDP based on the observation set obserState
        typeObjects specify for each possible observation the probabilistic behavior
        of the objects used by transType to define objects' transitions
    """
    dictRes = dict()
    stateSpace = None
    for elem in obserState:
        probVal = typeObjects[elem]
        def objTrans(dictIndex, state_t, a_t, objStates_t, nr_t, nc_t):
            return transType(dictIndex, state_t, a_t, objStates_t, nr_t, nc_t, distMin, probVal)
        dictRes[elem], stateSpace = build_mdp_nn(nrows, ncols, alphabet, \
                                            objStates, objTrans, obstacles)
    return dictRes, stateSpace

def uncertainties(nrows, ncols, Sset, Z, obstacles, output='uncertain_class.txt'):
    """ Build the uncertainciies due to the distance from the object to the agent.
        The uncertaincies are saved in the file output, which is then used by the
        c++ code in order to generate output_out containing the uncertain probvabilities
        from the Neural network
    """
    f = open(output, 'w')
    mapInput = list(product(Sset, Z))
    for (state, obs) in mapInput:
        agRow = state[0] / nrows
        agCol = state[0] % nrows
        objIdCounter = 0
        for objStateGrid in state[1:]:
            objRow = objStateGrid / nrows
            objCol = objStateGrid % nrows
            nonVisObj = isNonVisible(objCol, objRow, agCol, agRow, obstacles, nrows, ncols)
            if not nonVisObj:
                dist = np.sqrt((agCol-objCol)**2 + (agRow-objRow)**2)
                # Exponentially decreasing uncertaincies with distance to agent
                # probUncert = 1.5*1e-2 * np.exp(-4.0/(0.001+dist))
                probUncert = 1e-3 * np.exp(-4.0/(0.001+dist))
                f.write('{},{},{},{},{},{}\n'.format(obs[objIdCounter], objRow, objCol, agRow, agCol, probUncert))
            objIdCounter += 1
    f.close()

def readUncertainProbs(readFile='uncertain_class_out.txt'):
    """ Read the file with the uncertain probabilities returned by the Neural Network
        Access the uncertain probabilities by calling finalDict[objId][(agPosX, agPosY)][(objPosX,objPosY)]
    """
    f = open(readFile, 'r')
    Lines = f.readlines()
    finalDict = dict()
    for line in Lines:
        lineStrip = line.strip()
        splitData = lineStrip.split(',')
        objId = int(splitData[0])
        objPosX =int(splitData[1])
        objPosY = int(splitData[2])
        agPosX =int(splitData[3])
        agPosY = int(splitData[4])
        uncerT = float(splitData[5])
        prob = list()
        for i in range(5):
            res = splitData[6+i].split(';')
            p_lb, p_ub = float(res[0]), float(res[1])
            prob.append((max(0,p_lb),min(1.0,p_ub)))
        if objId not in finalDict:
            finalDict[objId] = dict()
        if (agPosX, agPosY) not in finalDict[objId]:
            finalDict[objId][(agPosX, agPosY)] = dict()
        finalDict[objId][(agPosX, agPosY)][(objPosX,objPosY)] = \
            (prob[objId][0], prob[objId][1])
    return finalDict

def buildObservationTransitionNN(nrows, ncols, Sset, Cset, Z, obstacles, uncertainDict):
    """ Build the probabilities to transition from a joint observation
        to another observation, using the Neural networks
    """
    mapInput = list(product(Sset, Cset.keys(), Z))
    mapDict = dict()
    for (state, mdpId, obs) in mapInput:
        # print(state, mdpId, obs)
        agRow = state[0] / nrows
        agCol = state[0] % nrows
        objIdCounter = 0
        resProb = 1.0
        for objStateGrid in state[1:]:
            objRow = objStateGrid / nrows
            objCol = objStateGrid % nrows
            nonVisObj = isNonVisible(objCol, objRow, agCol, agRow, obstacles, nrows, ncols)
            probType = 0.5
            if not nonVisObj:
                # Nominal value is the middle point of the uncertain interval
                unProb = uncertainDict[obs[objIdCounter]][(agRow,agCol)][(objRow,objCol)]
                probNominal = (unProb[0] + unProb[1]) * 0.5
                probType = probNominal if obs[objIdCounter] == mdpId[objIdCounter] else (1-probNominal)
            objIdCounter += 1
            resProb *= probType
        mapDict[(state, mdpId, obs)] = resProb
    return mapInput, mapDict

def buildObservationTransition(nrows, ncols, Sset, Cset, Z, obstacles):
    """ Build the probabilities to transition from a joint observation
        to another observation
    """
    mapInput = list(product(Sset, Cset.keys(), Z))
    mapDict = dict()
    for (state, mdpId, obs) in mapInput:
        # print(state, mdpId, obs)
        agRow = state[0] / nrows
        agCol = state[0] % nrows
        objIdCounter = 0
        resProb = 1.0
        for objStateGrid in state[1:]:
            objRow = objStateGrid / nrows
            objCol = objStateGrid % nrows
            nonVisObj = isNonVisible(objCol, objRow, agCol, agRow, obstacles, nrows, ncols)
            probType = 0.5
            if not nonVisObj:
                # We compute the uncertaincy based on the distance between
                # agent and the object
                dist = np.sqrt((agCol-objCol)**2 + (agRow-objRow)**2)
                probUncert = max(0.95 - np.exp(-4.0/(0.001+dist)), 0.5)
                probType = probUncert if obs[objIdCounter] == mdpId[objIdCounter] else (1-probUncert)
            objIdCounter += 1
            resProb *= probType
        mapDict[(state, mdpId, obs)] = resProb
    return mapInput, mapDict

# Grid size
nrows = 10
ncols = 10

# Generate random obstacles
np.random.seed(201) # Make sure the generated uncertain files have the same obstacle distribution
nbObs = 20
obstacles = []
for i in range(nbObs):
    x,y = np.random.randint(0, ncols), np.random.randint(0, nrows)
    obstacles.append(x*nrows + y)

# Action set
alphabet = [0,1,2,3] # North, south, west, east

# # Object state spaces -> object 0 moves on row 6, object 1 moves on col 10, object 2 moves on row 14
# objStates = {0 : [2*nrows + j for j in range(ncols)], 1 : [j*nrows + 8 for j in range(nrows)],
#              2 : [8*nrows + j for j in range(ncols)]}
# Object state spaces -> object 0 moves on row 6, object 1 moves on col 10, object 2 moves on row 14
# objStates = {0 : [2*nrows + j for j in range(ncols)], 1 : [j*nrows + 8 for j in range(nrows)]}
objStates = {0 : [2*nrows + j for j in range(ncols)], 1 : [j*nrows + 8 for j in range(nrows)],
             2 : [0*nrows + j for j in range(ncols)]} # object 2 goes along the diagonal
typeList = [0,1] # Only three types
observation = list(product(typeList, repeat=len(objStates)))

# Construct the behavior of each object
typeProbs = {(0,0,0) : [0.8,0.8,0.8], (0,0,1) : [0.8,0.8,0.2], (0,1,0) : [0.8,0.2,0.8],
             (0,1,1) : [0.8,0.2,0.2], (1,0,0) : [0.2,0.8,0.8], (1,0,1) : [0.2,0.8,0.2],
             (1,1,0) : [0.2,0.2,0.8], (1,1,1) : [0.2,0.2,0.2]}
# typeProbs = {(0,0) : [0.8,0.8], (0,1) : [0.8,0.2], (1,0) : [0.2,0.8],(1,1) : [0.2,0.2]}


Cset, Sset = buildSetMDPs(nrows, ncols, alphabet, objStates, typeProbs, obstacles, observation, distMin=1)
Z = observation

# mapInput, mapDict = buildObservationTransition(nrows, ncols, Sset, Cset, Z, obstacles)

# for elem, val in mapDict.items():
#     print (elem, val)

######## Construct the uncertaincies due to the distance between objects and agents ########
# uncertainties(nrows, ncols, Sset, Z, obstacles, output='uncertain_class.txt')

# # Read the built uncertain probabilities
uncertainProb = readUncertainProbs(readFile='uncertain_class_out.txt')
mapInput, mapDict = buildObservationTransitionNN(nrows, ncols, Sset, Cset, Z, obstacles, uncertainProb)

with open ("Cset_wildlife_10x10.pickle", "w") as f:
    pickle.dump(Cset, f)

with open ("observation_funct_wildlife_10x10.pickle", "w") as f:
    pickle.dump(mapDict, f)

# for elem, val in mapDict.items():
#     print (elem, val)
