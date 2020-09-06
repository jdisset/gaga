from __future__ import print_function
import time
import datetime
import random
from random import randint
import zmq
import json
import msgpack
import logging

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
            return bytes(json.dumps(msg), encoding='utf-8')

    def decodeMsg(self, msg):
        if self.compression:
            return msgpack.unpackb(msg, raw=False)
        else:
            return json.loads(msg)

    def start(self):
        while True:
            # send a READY request and wait for a reply
            self.socket.send(self.encodeMsg({'req':'READY', 'EVAL_batchSize':self.evalBatchSize, 'DISTANCE_batchSize':self.distanceBatchSize}))
            rep = self.decodeMsg(self.socket.recv())

            if rep['req'] == 'EVAL': # Evaluation of individuals
                logging.info("[WORKER %s] - Received %d EVAL tasks", self.identity, len(rep['tasks']));
                results = self.evaluationFunc(rep['tasks'], rep['extra'])
                assert len(results) == len(rep['tasks'])
                reply = self.encodeMsg({'req':'RESULT', 'individuals':results})
                self.socket.send(reply)
                self.socket.recv() #ACK

            elif rep['req'] == 'DISTANCE': # Distance computations for novelty
                signatures = [i['signature'] for i in rep['extra']['archive']]
                distances = [i for i in rep['tasks']]
                logging.info("[WORKER %s] - Received %d distances computations for %d signatures", self.identity, len(rep['tasks']), len(signatures));
                logging.debug("[WORKER %s] - Signatures : %s", self.identity, json.dumps(signatures));
                ta = datetime.datetime.now()
                distances = [[i[0],i[1],self.distanceFunc(signatures[i[0]], signatures[i[1]])] for i in distances]
                delta = datetime.datetime.now() - ta
                logging.info("[WORKER %s] - Computed %d distances in %d ms", self.identity, len(rep['tasks']), int(delta.total_seconds()*1000));
                reply = self.encodeMsg({'req':'RESULT', 'distances':distances})
                self.socket.send(reply)
                self.socket.recv() #ACK

            elif rep['req'] == 'STOP':
                logging.info("Received STOP request, exiting")
                break;

            else :
                logging.warning("Received unknown request:",strRep.decode('utf-8'))
