from __future__ import annotations 
import sys
import os
import re
import json
import datetime
import random
import numpy as np
import uuid
from langchain.embeddings.base import Embeddings
from langchain.embeddings import FakeEmbeddings
from langchain.vectorstores import FAISS
from collections import Counter
import pickle

from sklearn.model_selection import StratifiedKFold
from VectorizerConnector import VectorizerConnector
import confusionVisualizer


def ts():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
def log(msg, end="\n"):
    print(f"{ts()} | {msg}", end=end, flush=True)
def logSmall(msg):
    print(f"{msg}", end="", flush=True)

from collections import Counter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)
from pathlib import Path
import time


      
class IntDetTrainer(object):
    """class responsible for intent detection model training"""
    def __init__(self, ft, modelName, trainData, action):
        self.ft = ft
        self.modelName = modelName
        self.trainData = trainData
        vect_end = 50 if action == "train" else 20
        self.logger = ProgressLogger(f"{modelName}.{action}.status", 40, 0.5, vect_end, 50, 100)

    def doVectorizing(self):
        self.logger.clearFile()                       
        self.questions, self.answers = map(list, zip(*((d["text"], d["intent"]) for d in self.trainData)))
        self.x, self.y, self.answerDict = IntDetUtils.prepareSentenceXY(self.ft, self.questions, self.answers, self.logger)
        if len(self.x)>0:
            self.revDict = IntDetUtils.reverseDict(self.answerDict)
            IntDetUtils.writeDict(f"{self.modelName}.intentmap.json", self.answerDict)

    def doTraining(self):
        self.model = IntDetUtils.runTrainSequence(self.modelName, self.x, self.y, self.questions, self.answers, self.revDict, self.answerDict, self.logger)
        with open(f"{self.modelName}.train.done", "w") as done_file:
            pass

    def doXVal(self):
        confMatrix, badExamples = IntDetUtils.runKFoldXValSequence(self.x, self.y, self.questions, self.answers, self.revDict, self.answerDict, self.logger)
        with open(f"{self.modelName}.confusion.txt", "w", encoding="utf-8") as confMatrFile:
            confMatrFile.write("\n".join("\t".join(str(x) for x in y) for y in confMatrix) + "\n")
        with open(f"{self.modelName}.confusionExamples.txt", "w", encoding="utf-8") as badExamplesFile:
            json.dump(badExamples, badExamplesFile, ensure_ascii=False, indent=2)
        confusionVisualizer.processModel(self.modelName)
        with open(f"{self.modelName}.xval.done", "w") as done_xval_file:
            pass


class IntDetServer(object):
    """class responsible for intent detection model serving"""

    def __init__(self, ft, modelName):
        self.ft = ft
        self.modelName = modelName
        
        self.model = FAISS.load_local(self.modelName, FakeEmbeddings(size=100))

        self.answerDict = IntDetUtils.readDict(f"{self.modelName}.intentmap.json")
        self.answerCounter = 0

    def get_intents(self, question):
        self.answerCounter += 1
        logSmall(".")
        if self.answerCounter % 100 == 0:
            log(f"Answered {self.answerCounter} queries")
        rr = IntDetUtils.query(self.ft, self.model, question,5)
        returnValue = {}
        returnValue['text'] = question
        if len(rr)>0:
            returnValue['intents'] = IntDetUtils.resultToDict(rr)
        else:
            returnValue['intents'] = []
        return returnValue

class ProgressLogger():
    def __init__(self, logfile, epochs, vect_stepsize, vect_end, train_start, end):
        self.logfile = logfile
        os.makedirs(os.path.dirname(self.logfile), exist_ok=True)
        self.epochs = epochs if epochs > 0 else 1
        self.end = end
        self.vect_stepsize = vect_stepsize
        self.vect_end = vect_end
        self.train_start = train_start
        self.vect_curr_perc = self.vect_stepsize

    def clearFile(self):
        with open(self.logfile, "w", encoding="utf-8") as f:
            pass
    def getPerc(self, start, end, current, all):
        return start + (end - start) * (current + 1) / all
    def writePerc(self, perc):
        try:
            with open(self.logfile, "a", encoding="utf-8") as f:
                f.write(f"{ts()} | {perc:0.2f}\n")
        except (PermissionError, BlockingIOError):
            pass
    def on_vect_progress(self, wordNr, wordCount):
        while (self.vect_end - 2) * (wordNr+1) / wordCount >= self.vect_curr_perc:
            perc = self.vect_curr_perc
            self.writePerc(perc)
            self.vect_curr_perc += self.vect_stepsize
    def on_epoch_end(self, epoch, logs=None):
        perc = self.getPerc(self.train_start, self.end, epoch, self.epochs)
        self.writePerc(perc)
    def reconfigureXVal(self, currentFold, foldCount):
        self.train_start = self.vect_end + (100 - self.vect_end) * currentFold / foldCount
        self.end = self.vect_end + (100 - self.vect_end) * (currentFold+1) / foldCount
    def nextfold(self):
        self.writePerc(self.end)



class IntDetUtils(object):
    """useful functions for intent detection"""

    @staticmethod
    def getSentenceVector(ft, question):
        if len(question.strip()) == 0:
            question = 'a'
        vec=ft.vectorize(question)
        if len(vec)==1: #vector containing embeddings for a single question, returning only embeddings for that question
            return vec[0]
        else:
            return vec

    @staticmethod
    def prepareSentenceXY(ft, questions, answers, logger=None):
    
        if answers == None:
            sentenceVectors = []
            for i, q in enumerate(questions):
                sentvec=IntDetUtils.getSentenceVector(ft, q)
                if len(sentvec)==0:
                    return [], None, None
                sentenceVectors.append(sentvec)

            x = np.array(sentenceVectors)
            return x, None, None
        
        else:
            if logger:
                logger.writePerc(0)
            sentenceVectors = []
            num_q=len(questions)

            for i in range(0,num_q,256):
                subarr = questions[i:min(i+256,num_q)]
                sentvec=ft.vectorize(json.dumps(subarr))
                if logger:
                    logger.on_vect_progress(i, len(questions))
                if len(sentvec)==0:
                    return [], None, None

                sentenceVectors.extend(sentvec)

            x = np.array(sentenceVectors)
            answerDict = { ans : i for i, ans in enumerate(sorted(set(answers))) }
            
            if logger:
                logger.writePerc(logger.vect_end - 1)
                logger.writePerc(logger.vect_end)

            answerDict = { ans : i for i, ans in enumerate(sorted(set(answers))) }          

        return x, answers, answerDict

    @staticmethod
    def runTrainSequence(modelName, x, y, questions, answers, revDict, answerDict, logger):
        log("Training process started")
        
        model = IntDetUtils.simpleTrain(x, y, modelName)
        if logger:
            logger.writePerc(80)         
        model.save_local(modelName)
            
        log("Training process finished")
        if logger:
            logger.writePerc(100)
        return model

    @staticmethod
    def writeDict(filename, dd):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(dd, f, ensure_ascii=False, indent=2)

    @staticmethod
    def readDict(filename):
        with open(filename, "r", encoding="utf-8") as f:
            jj = json.load(f)
        return jj

    @staticmethod
    def runKFoldXValSequence(unshuffledX, unshuffledY, unshuffledQuestions, unshuffledAnswers, revDict, answerDict, xval_logger=None, k=5):
        minCount = np.unique(unshuffledAnswers, return_counts=True)[1].min()
        badExamples = []
        if minCount < k:
            log(f"At least {k} test examples for each intent are needed to perform cross-validation (currently minimum is {minCount})")
            return np.zeros(0, dtype=int), badExamples
        X, y, que, ans, kf = IntDetUtils.getKFold(unshuffledX, unshuffledY, unshuffledQuestions, unshuffledAnswers, k=k)
        confMatrix = np.zeros((len(answerDict), len(answerDict)), dtype=int)

        for foldNr, (train_index, test_index) in enumerate(kf):
            log(f"Fold {foldNr}:")
            X_train, X_test = X[train_index], X[test_index]
            Que_test = []
            for i in test_index:
                Que_test.append(que[i])
            y_train, y_test = y[train_index], y[test_index]
            if xval_logger:
                xval_logger.reconfigureXVal(foldNr, k)    
            
            model = IntDetUtils.simpleTrain(X_train, y_train)
                
            correctConfidences, topIntent, topConf = IntDetUtils.getConfidences(model, X_test, y_test, answerDict)
            correctIntent = [answerDict[item] for item in y_test]
            for j in range(0, len(topIntent)):
                confMatrix[correctIntent[j], topIntent[j]] += 1
                if correctIntent[j] != topIntent[j]:
                    badExamples.append({
                        "question": Que_test[j],
                        "topIntent": revDict[topIntent[j]],
                        "topIntentProbability": float(topConf[j]),
                        "correctIntent": revDict[correctIntent[j]],
                        "correctIntentProbability": float(correctConfidences[j]), 
                        })
            if xval_logger:
                xval_logger.nextfold()
        return confMatrix, badExamples
    
    @staticmethod
    def getKFold(X, y, questions, answers, k=10, randomSeed=1):
        assert(len(X) == len(y))
        n = len(y)
        r = list(range(n))
        random.seed(randomSeed)
        random.shuffle(r)

        #own shuffle
        shX = []; shY = []; ans = []; que = [];
        for pos in r:
            shX.append(X[pos])
            shY.append(y[pos])
            ans.append(answers[pos])
            que.append(questions[pos])
        
        skf = StratifiedKFold(k, shuffle=True, random_state=randomSeed)
        res = skf.split(X, ans)
        return np.array(shX), np.array(shY), que, ans, res

    @staticmethod
    def simpleTrain(x, y, modelName=None):
        if modelName:
            metadatas = [{"source": modelName.split('/')[-1]}] * len(y)
        else:
            metadatas = None
        text_embedding_pairs = [(txt, emb) for txt, emb in zip(y, x)]
        return FAISS.from_embeddings(text_embedding_pairs, embedding=FakeEmbeddings(size=100),metadatas=metadatas)

    @staticmethod
    def reverseDict(answerDict):
        revDict = {value : key for key, value in answerDict.items()}
        return revDict

    @staticmethod
    def processResult(res, answerDict, k=5):
        assert(len(res) == len(answerDict))
        rr = sorted([(item, res[nr]) for item, nr in answerDict.items()],
                    key=lambda tuple:tuple[1], reverse=True)
        rr = rr[:k]
        return rr
     
    @staticmethod
    def resultToDict(res):
        retval = [{"intentid": r[0], "confidence": r[1], "source": r[2]} if len(r) == 3 else {"intentid": r[0], "confidence": r[1]} for r in res]
        return retval
        
    @staticmethod
    def getTransformSimilarity(sim):
        return 1.0 - min(1.0, sim)
    
    @staticmethod
    def getConfidences(model, x, y, answerDict):
    
        correctConfidences = np.zeros(len(y))
        topIntent = np.zeros(len(y), dtype=int)
        topConf = np.zeros(len(y))

        for i, singleitem in enumerate(x):
            result = model.similarity_search_with_score_by_vector(singleitem,k=10)
            # scores in result are in ascending order, less is better
            intents=[]
            intents_scores = {}
            
            topConf[i] = IntDetUtils.getTransformSimilarity(result[0][1]) #the best transformed similarity score
            topIntent[i] = answerDict[result[0][0].page_content]
            
            for d in result:
                intent_idx = answerDict[d[0].page_content]
                intents.append(intent_idx)
                sim = IntDetUtils.getTransformSimilarity(d[1])
                if intent_idx not in intents_scores:
                    intents_scores[intent_idx] = sim # saving result intent's (1 - similarity) (best) result
                    
                if d[0].page_content == y[i] and correctConfidences[i] < sim:
                    correctConfidences[i]= sim # saving required intent's (1 - similarity) (best) result
                    
        return correctConfidences, topIntent, topConf
    
    @staticmethod
    def writeBadConf(fileName, questions, answers, correctConfidences, topIntent, topConf, answerDict, threshold=0.5):
        badIndices = (correctConfidences < threshold).nonzero()
        badQuestions = np.array(questions)[badIndices]
        badAnswers = np.array(answers)[badIndices]
        badConfidences = correctConfidences[badIndices]
        badTopIntent = topIntent[badIndices]
        badTopIntent = [answerDict[i] for i in badTopIntent]
        badTopConf = topConf[badIndices]
        with open(fileName, 'w', encoding='utf-8') as f:
            json.dump([{
                    "question": bq,
                    "topIntent": bti,
                    "topIntentProbability": float(btc),
                    "correctIntent": ba,
                    "correctIntentProbability": float(bc),
                } for bq, ba, bc, bti, btc in zip(badQuestions, badAnswers, badConfidences, badTopIntent, badTopConf)],
                      f, indent=2, ensure_ascii=False)

    @staticmethod
    def query(ft, model, question, items=5):
        x, _, _ = IntDetUtils.prepareSentenceXY(ft, [question], None)
        output=[]
            
        if len(x)>0:
            result = model.similarity_search_with_score_by_vector(x[0],k=10)
            intents=set()

            for d in result:
                if "source" in d[0].metadata:
                    if (d[0].page_content,d[0].metadata["source"]) not in intents:
                        intents.add((d[0].page_content,d[0].metadata["source"]))
                        output.append((d[0].page_content, IntDetUtils.getTransformSimilarity(d[1]),d[0].metadata["source"]))
                        items = items - 1
                        if items==0:
                            break
                elif d[0].page_content not in intents:
                    intents.add(d[0].page_content)
                    output.append((d[0].page_content, IntDetUtils.getTransformSimilarity(d[1])))            
                
                    items = items - 1
                    if items==0:
                        break
                                    
        return output
