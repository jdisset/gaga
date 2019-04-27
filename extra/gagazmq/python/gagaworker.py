from __future__ import print_function
import time
import random
from random import randint
import zmq
import json

class GAGAWorker:

    def __init__(self,serverAddress, evaluationFunc, distanceFunc = lambda x,y:0, evalBatchSize = 1, distanceBatchSize = 10, identity = u"", compression = False):
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.REQ)
        self.identity = identity + u"-%04x-%04x" % (randint(0, 0x10000), randint(0, 0x10000))
        self.evalBatchSize = evalBatchSize
        self.distanceBatchSize = distanceBatchSize
        self.evaluationFunc = evaluationFunc
        self.distanceFunc = distanceFunc
        self.compression = compression
        self.socket.setsockopt_string(zmq.IDENTITY, self.identity)
        self.socket.connect(serverAddress)

    def start(self):
        while True:
            req = {'req':'READY', 'EVAL_batchSize':self.evalBatchSize, 'DISTANCE_batchSize':self.distanceBatchSize}
            self.socket.send(bytes(json.dumps(req),'utf8'))
            strRep = self.socket.recv()
            rep = json.loads(strRep)
            #print("sent READY request")

            if rep['req'] == 'EVAL':
                dnaList = [i['dna'] for i in rep['tasks']]
                results = self.evaluationFunc(rep['tasks'])
                assert len(results) == len(rep['tasks'])
                reply = {'req':'RESULT', 'individuals':results}
                self.socket.send(bytes(json.dumps(reply), 'utf-8'))
                self.socket.recv() #ACK

            elif rep['req'] == 'DISTANCE':
                footprints = [i['footprint'] for i in rep['extra']['archive']]
                distances = [i for i in rep['tasks']]
                #print('computing', len(distances), 'distances from', len(footprints), 'footprints')
                distances = [[i[0],i[1],self.distanceFunc(footprints[i[0]], footprints[i[1]])] for i in distances]
                reply = {'req':'RESULT', 'distances':distances}
                self.socket.send(bytes(json.dumps(reply), 'utf-8'))
                self.socket.recv() #ACK

            elif rep['req'] == 'STOP':
                print("Received STOP request, exiting")
                break;

            else :
                print("[WARNING] Received unknown request:",strRep.decode('utf-8'))


