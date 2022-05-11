#!/usr/bin/env python

import korali
from mpi4py import MPI
import numpy as np
from scipy.special import factorial

def multi_rank_function(ks):
    """
    A dumb way to compute log(2) by maximizing -(exp(x)-2)**2
    exp(x) is here computed through a taylor expansion in parallel.
    This example is just here to illustrate the use of MPI with korali.
    """
    comm = korali.getWorkerMPIComm()
    rank = comm.Get_rank()
    size = comm.Get_size()

    x = ks["Parameters"][0]

    ntot = 20

    # split the work among ranks
    work_per_rank = (ntot + size - 1) // size
    start = rank * work_per_rank
    end = min([start + work_per_rank, ntot])

    n = np.arange(start, end)
    local_expx = np.array([np.sum(x**n / factorial(n))])

    # gather the result with MPI
    expx = np.array([0.0])
    comm.Allreduce(local_expx, expx, op=MPI.SUM)

    ks["F(x)"] = -(expx[0] - 2)**2


if __name__ == '__main__':

    # The optimization problem is described in a korali Experiment object
    e = korali.Experiment()

    e["Problem"]["Type"] = "Optimization"
    e["Problem"]["Objective Function"] = multi_rank_function

    # Defining the problem's variables.
    e["Variables"][0]["Name"] = "x"
    e["Variables"][0]["Initial Value"] = 0.0
    e["Variables"][0]["Initial Standard Deviation"] = 1.0

    # Configuring CMA-ES parameters
    e["Solver"]["Type"] = "Optimizer/CMAES"
    e["Solver"]["Population Size"] = 16
    e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-8
    e["Solver"]["Termination Criteria"]["Max Generations"] = 100

    # Experiments need a Korali Engine object to be executed
    k = korali.Engine()
    k.setMPIComm(MPI.COMM_WORLD)
    k["Conduit"]["Type"] = "Distributed"
    k["Conduit"]["Ranks Per Worker"] = 2


    # Run the optimization
    k.run(e)
