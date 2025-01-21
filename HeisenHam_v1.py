import numpy
from setup_v3 import readInput

from mpi4py import MPI
subroutine = MPI.COMM_WORLD
size = subroutine.Get_size()
rank = subroutine.Get_rank()

model, nSite, subSpace, nStates, s2Target, maxItr, startSpinTargetItr, energyTola, spinTola, beta, jVal, det, Ms,  posibleDet, bondOrder, outputfile, restart, saveBasis,  multi = readInput()

f1 = open(bondOrder)
line1=f1.readline()
bO1=[]
bO2=[]
while line1:
    values = line1.split()
    line1 = f1.readline()
    bO1.append(int(values[0])-1)
    bO2.append(int(values[1])-1)

cSz = -jVal * 0.25
cSxSy = -jVal * 0.50

def subSited(a1, a2):
    diff = 0
    for i in range(nSite):
        if a1[i] != a2[i]:
            diff += 1
    return diff

def opSz(a):
    Sz = 0.0
    for i, x in enumerate(bO1):
        if (a[bO1[ i]] == a[ bO2[ i]]):
            Sz += cSz
        else:
            Sz -= cSz
    return Sz
        
def opSxSy(a, b):
    SxSy = 0.0
    for i, x  in enumerate(bO1):
        if ((a[bO1[i]] != b[bO1[i]] and a[bO2[i]] != b[bO2[i]])  and a[bO1[i]] != a[bO2[i]] ):
            SxSy += cSxSy
    return SxSy

def Hamiltonian(A):
    lenA = len(A)
    Hsub = numpy.zeros((lenA,lenA))

    """------------------------------------------------------------| Parallel Region |------------------------------------------------------------"""    
    chunks = int(lenA // size)
    start = rank * chunks
    if rank < (size - 1):
        end = start + chunks
    else:
        end = lenA
    
    # start_time = MPI.Wtime()
    # for idx, x in enumerate(A) :
    for idx in range(start, end) :
        for idy, y in enumerate(A) :
            siteDiff = subSited(A[idx], y)
            if siteDiff == 0:
                Hsub[idx][idy] = opSz(A[idx].bin)
            if siteDiff == 2:
                Hsub[idx][idy] = opSxSy(A[idx].bin, y.bin)
    """-------------------------------------------------------------------------------------------------------------------------------------------"""
    # end_time = MPI.Wtime() - start_time
    # print(f"Hemelton time {end_time:.4f} seconds.")
    return Hsub

f1.close()
