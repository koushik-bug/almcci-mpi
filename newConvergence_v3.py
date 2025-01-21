import random
import statistics as stat
import math
from setup_v3 import readInput

from mpi4py import MPI
subroutine = MPI.COMM_WORLD
size = subroutine.Get_size()
rank = subroutine.Get_rank()

model, nSite, subSpace, nStates, s2Target, maxItr, startSpinTargetItr, energyTola, spinTola, beta, jVal, det, Ms,  posibleDet, bondOrder,  outputfile, restart, saveBasis, multi = readInput()

def updateDeterminatList(allDetCi,  newGen, ci, dataFile, step ):
    
    
   # do the shuffle if required
   for idx, elem in enumerate(newGen):
       allDetCi[elem.bin]=ci[idx]
   items = list(allDetCi.items())

   # Shuffle the list
   random.shuffle(items)

   # Convert the shuffled list back to a dictionary
   shuffled_dict = dict(items) 
   
   with open(dataFile,"w") as fout:
        for elem in (list(shuffled_dict.items())):
            newline = elem[0].replace("1","1,").replace("0", "-1,")+str(abs(math.log10(abs(elem[1]) + 1e-16 )))+"\n"
            fout.write(newline)

   return allDetCi

def makeFitGeneration(basis, ci, newSize):
    ciOrdered = sorted(abs(ci ),  reverse =True)
    ciOrdered = ciOrdered[: newSize]
    fitness=[]
    ciFit = []
    for x in ciOrdered:
        ix = list(abs(ci)).index(x)     
        if basis[ix] not in fitness:
            fitness.append( basis[ix])
            ciFit.append(ci[ix])
            if (Ms[0] == 0):
                fitness.append( ~basis[ix])
                ixx = basis.index(~basis[ix])
                ciFit.append(ci[ixx])
    return fitness, ciFit


def convInitializer():
    # store the n'th state( target state number ) for i'th and i+1'th steps, initialized with garbage value
    targetState = [100, 101] 
    #store the diff of s^2 value of n'th state for i'th and i+1'th steps, initialized with garbage value
    s2ValDiff =  [0.0, 0.0] # 0'th should be higher than 1'th
    # to check convergence, store change in energy for i'th to (i-4)'th steps
    energyChange = [1.0, 1.0, 1.0, 1.0, 1.0]
    spinChange = [10.0, 10.0, 10.0, 10.0, 10.0]
    s2ValList = []    
    [s2ValList.append(0.0) for x in range(nStates)]    
    return targetState, s2ValList, s2ValDiff, energyChange, spinChange


def update( energy, ciCoef, basis, lenSB ):
    energySave = energy
    ciSave = ciCoef
    basisSave = basis
    return energySave, ciSave, basisSave


def checkConvergence( eMin, eNew, ciMin, ciNew, s2Min, s2New, targetState,  newGen, s2ValDiff, itr, newSize):
    fitGen = []
    if (s2ValDiff[1] - s2ValDiff[0] <= spinTola) :
        if ((eNew <=  eMin) or (random.random() <  math.exp( -( beta * (eNew - eMin) ) ))):
            eMin = eNew 
            s2Min = s2New
            fitGen, ciMin= makeFitGeneration(newGen, ciNew, newSize)
            s2ValDiff[0] = s2ValDiff[1]
            targetState[0] = targetState[1]
            energyUpdate = True
        else:
            fitGen = newGen[: newSize]
            energyUpdate = False
    else: 
        fitGen = newGen[: newSize]
        energyUpdate = False
    if rank == 0:
        newline = ("ite->\t%d ; spece->\t%d ; Energy->\t%f ; State->\t%d ; s^2 Expe Val->\t%2.4f ;\n")%((itr +1), len(newGen), round( eMin, 6), targetState[0] + 1, round(s2Min,4))
        with open(outputfile, "a") as fout:
            fout.write(newline)
    return fitGen[: int(0.8 * newSize)], eMin, ciMin, s2ValDiff, s2Min, energyUpdate

def checkFinalConv(energyChange, spinChange, eOld, eNew, spinChangeIth, convReach):
    energyChange = energyChange[1 :]+ [abs(eOld - eNew)]
    spinChange = spinChange[1 :]+ [spinChangeIth]
    if (( stat.mean(spinChange )  < spinTola)  and ( stat.mean( energyChange )  < energyTola)):
        convReach = True
    return energyChange, spinChange, convReach
