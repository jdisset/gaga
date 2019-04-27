import sys
sys.path.insert(0,'..')
from gagaworker import GAGAWorker
import json
import numpy as np

def maxSum(indList):
    for i, _ in enumerate(indList):
        val = json.loads(indList[i]['dna'])['values']
        indList[i]['fitnesses']['sum'] = sum(val)
        indList[i]['footprint'] = val;
    return indList

def euclidianDistance(x, y):
    a = np.array(x)
    b = np.array(y)
    return np.linalg.norm(a-b)

worker = GAGAWorker("tcp://localhost:4321", evaluationFunc = maxSum, evalBatchSize = 2, distanceFunc = euclidianDistance, distanceBatchSize = 100000, compression = False)
worker.start()
