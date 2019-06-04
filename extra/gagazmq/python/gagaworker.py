from __future__ import print_function
import time
import random
from random import randint
import zmq
import json
import msgpack

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

    def encodeMsg(self, msg):
        if self.compression:
            return msgpack.packb(msg)
        else:
            return bytes(json.dumps(msg), 'utf-8')

    def decodeMsg(self, msg):
        if self.compression:
            return msgpack.unpackb(msg, raw=False)
        else:
            return json.loads(msg)



    def start(self):
        while True:

            # send a READY request and wait for a reply
            print("sending ready");
            self.socket.send(self.encodeMsg({'req':'READY', 'EVAL_batchSize':self.evalBatchSize, 'DISTANCE_batchSize':self.distanceBatchSize}))
            rep = self.decodeMsg(self.socket.recv())
            print("received");

            if rep['req'] == 'EVAL': # Evaluation of individuals
                #dnaList = [i['dna'] for i in rep['tasks']]
                results = self.evaluationFunc(rep['tasks'], rep['extra'])
                assert len(results) == len(rep['tasks'])
                reply = self.encodeMsg({'req':'RESULT', 'individuals':results})
                self.socket.send(reply)
                self.socket.recv() #ACK

            elif rep['req'] == 'DISTANCE': # Distance computations for novelty
                footprints = [i['footprint'] for i in rep['extra']['archive']]
                distances = [i for i in rep['tasks']]
                print('computing', len(distances), 'distances from', len(footprints), 'footprints')
                distances = [[i[0],i[1],self.distanceFunc(footprints[i[0]], footprints[i[1]])] for i in distances]
                print('distances = ', distances)
                reply = self.encodeMsg({'req':'RESULT', 'distances':distances})
                self.socket.send(reply)
                self.socket.recv() #ACK

            elif rep['req'] == 'STOP':
                print("Received STOP request, exiting")
                break;

            else :
                print("[WARNING] Received unknown request:",strRep.decode('utf-8'))


