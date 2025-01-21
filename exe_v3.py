import math
import numpy
import time
import sys
import random
import net_nstates
import array
import os
#import gc
from mpi4py import MPI

subroutine = MPI.COMM_WORLD
size = subroutine.Get_size()
rank = subroutine.Get_rank()

from setup_v3 import readInput
from MCCI_v3 import performMCCI

start = time.time()

model, nSite, subSpace, nStates, s2Target, maxItr, startSpinTargetItr, energyTola, spinTola, beta, jVal, det, Ms,  posibleDet, bondOrder, outputfile, restart, saveBasis, multi  = readInput()

newline1 = '''\n
  A    L           M   M   CCCC   CCCC   II
 A A   L           MM MM  C      C       II
AAAAA  L     ###   M M M C      C        II
A   A  L           M   M  C      C       II
A   A  LLLLL       M   M   CCCC   CCCC   II\n'''


status = ""
# Ensure MPI synchronization
subroutine.Barrier()
if rank == 0:
    with open(outputfile, "a", buffering=1) as fout:
        fout.write(newline1)
        fout.flush()
    
    newline = ("\nTotal Posible Determinats are %d .\nBreakup are [Ms, No of Determinants] - ")% (sum(posibleDet))
    with open (outputfile, "a") as fout:
        fout.write(newline)
        fout.flush()
    for i in range(len(Ms)):
        newline = ("\t[%d, %d]")%(Ms[i], posibleDet[i])
        with open(outputfile, "a") as fout:
            fout.write(newline)
            fout.flush()
            if (i+1 == len(Ms)):
                fout.write("\n\n")
                fout.flush()
    if ( subSpace > (sum(posibleDet) *0.8)):
        status = 'exit'
        newline = ("\nSub-Space size is more than 80 % of total determinants space. Make Sub-Space size smaller and run it again.\n ")
        print(newline)
        with open (outputfile, "a") as fout:
            fout.write(newline)
            fout.flush()

subroutine.Barrier()
status = subroutine.bcast(status,root=0)

if status == 'exit':
        sys.exit()

performMCCI()
if rank == 0:
    newline = ("Total Time Taken in MCCI Calculation is %f sec.")%( time.time() - start )
    with open(outputfile, "a") as fout:
        fout.write(newline)
        fout.flush()
        fout.close()
