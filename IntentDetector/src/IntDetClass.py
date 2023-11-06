from __future__ import annotations 
import tornado.web
import sys
import os
import re
import json
import datetime
import random
import numpy as np
from langchain.embeddings import FakeEmbeddings
from langchain.vectorstores import FAISS
import pickle
import os
from sklearn.model_selection import StratifiedKFold
from VectorizerConnector import VectorizerConnector

def ts():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
def log(msg, end="\n"):
    print(f"{ts()} | {msg}", end=end, flush=True)

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
      
class IntDetServerTrainer(object):
    """class responsible for intent detection model training"""
    def __init__(self, ft, modelName, trainData=None, action=""):
        self.ft = ft
        self.model=None
        self.modelName = modelName
        self.trainData = trainData
        vect_end = 50 if action == "train" else 20
        self.logger = ProgressLogger(f"{modelName}.{action}.status", 40, 0.5, vect_end, 50, 100)

        if os.path.isfile(f"{self.modelName}/index.faiss"):
            self.model = FAISS.load_local(self.modelName, FakeEmbeddings(size=1))
            print("Model loaded",flush=True)
            
        if os.path.isfile(f"{self.modelName}.intentmap.json"):
            self.answerDict = IntDetUtils.readDict(f"{self.modelName}.intentmap.json")

    def doVectorizing(self,trainData="",lang=""):
        self.logger.clearFile()                  
        if trainData != "":
            self.trainData=trainData
            
        self.questions, self.answers = map(list, zip(*((d["text"], d["intent"]) for d in self.trainData)))
        self.x, self.y, self.answerDict = IntDetUtils.prepareSentenceXY(self.ft, self.questions, self.answers, lang, self.logger)
        
        if len(self.x)>0:
            IntDetUtils.writeDict(f"{self.modelName}.intentmap.json", self.answerDict)

    def doTraining(self, source='', new='1'):
        if source=='':
            model = IntDetUtils.simpleTrain(self.x, self.y, self.modelName)
        else:
            model = IntDetUtils.simpleTrain(self.x, self.y, source)
            
        if self.logger:
            self.logger.writePerc(100)

        retval=self.mergeModel(model,new)
        print("Model trained",flush=True)
        with open(f"{self.modelName}.train.done", "w") as done_file:
            pass
            
        return retval
            
    def mergeModel(self,newmodel,new='0'):
        try:
            if (new=='0'):
                self.model.merge_from(newmodel)
            else:
                del self.model
                self.model=newmodel
            self.model.save_local(self.modelName)
            
            return ({"status": f"SUCCESS"})
        except:
            return ({"status": f"ERROR: Failed to merge models!"})
        
    def doXVal(self):
        revDict = {value : key for key, value in self.answerDict.items()}
        
        confMatrix, badExamples = IntDetUtils.runKFoldXValSequence(self.x, self.y, self.questions, self.answers, revDict, self.answerDict, self.logger)
        
        with open(f"{self.modelName}.confusionExamples.txt", "w", encoding="utf-8") as badExamplesFile:
            json.dump(badExamples, badExamplesFile, ensure_ascii=False, indent=2)
        
        statFile = f"{self.modelName}.stats.json"
        intentStatFile = f"{self.modelName}.intentstats.json"
   
        F1, precision, recall, TP, FP, FN = IntDetUtils.calcMeasures(confMatrix)
        globalAccuracy, globalPrecision, globalRecall, globalF1, globalOtherF1 = IntDetUtils.calcGlobalMeasures(F1, precision, recall, TP, FP, FN)

        with open(intentStatFile, 'w', encoding='utf-8') as f:
            json.dump([{
                "intent": revDict[i],
                "precision": prec,
                "recall": rec,
                "F1": f1,
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
            } for i, (prec, rec, f1, tp, fp, fn) in enumerate(zip(precision, recall, F1, TP, FP, FN))], f, indent=2, ensure_ascii=False)

        with open(statFile, 'w', encoding='utf-8') as f:
            json.dump({
                "accuracy": globalAccuracy,
                "precision": globalPrecision,
                "recall": globalRecall,
                "MicroF1": globalF1,
                "MacroF1": globalOtherF1,
            }, f, indent=2)
            
        with open(f"{self.modelName}.xval.done", "w") as done_xval_file:
            pass
    
    def get_intents(self, question,lang=""):
        returnValue = {}
        returnValue['text'] = question
        returnValue['intents'] = []
        
        if self.model != None: 
            rr = IntDetUtils.query(self.ft, self.model, question,lang,5)
        
            if len(rr)>0:
                returnValue['intents'] =[{"intentid": r[0], "confidence": r[1], "source": r[2]} if len(r) == 3 else {"intentid": r[0], "confidence": r[1]} for r in rr]
            
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

    def reconfigureXVal(self, currentFold, foldCount):
        self.train_start = self.vect_end + (100 - self.vect_end) * currentFold / foldCount
        self.end = self.vect_end + (100 - self.vect_end) * (currentFold+1) / foldCount
        
    def nextfold(self):
        self.writePerc(self.end)



class IntDetUtils(object):
    """useful functions for intent detection"""

    @staticmethod
    def getSentenceVector(ft, question, lang):
        if len(question.strip()) == 0:
            question = 'a'
            
        vec=ft.vectorize(question, lang)
        
        if len(vec)==1: #vector containing embeddings for a single question, returning only embeddings for that question
            return vec[0]
        else:
            return vec

    @staticmethod
    def prepareSentenceXY(ft, questions, answers, lang, logger=None):
        if answers == None:
            sentenceVectors = []

            for i, q in enumerate(questions):
                sentvec=IntDetUtils.getSentenceVector(ft, q, lang)
                
                if len(sentvec)==0:
                    return [], None, None
                    
                sentenceVectors.append(sentvec)

            x = np.array(sentenceVectors)
            return x, None, None
        
        else:
            if logger:
                logger.writePerc(0)
            print(answers)    
            sentenceVectors = []
            num_q=len(questions)

            for i in range(0,num_q,256):
                subarr = questions[i:min(i+256,num_q)]
                sentvec=ft.vectorize(json.dumps(subarr),lang)
                
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
            print(np.unique(unshuffledAnswers, return_counts=True))
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
        return FAISS.from_embeddings(text_embedding_pairs, embedding=FakeEmbeddings(size=1),metadatas=metadatas)

    @staticmethod
    def processResult(res, answerDict, k=5):
        assert(len(res) == len(answerDict))
        rr = sorted([(item, res[nr]) for item, nr in answerDict.items()],
                    key=lambda tuple:tuple[1], reverse=True)
        rr = rr[:k]
        return rr
            
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
    def query(ft, model, question, lang, items=5):
        print(f"queryLanguage: {lang}")
        x, _, _ = IntDetUtils.prepareSentenceXY(ft, [question], None, lang, None)
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
        
    @staticmethod
    def calcGlobalMeasures(F1, precision, recall, TP, FP, FN):
        if len(F1) == 0: return 0,0,0,0,0
        globalTP = np.sum(TP)
        globalFP = np.sum(FP)
        globalFN = np.sum(FN)
        globalAccuracy = globalTP / (globalTP + globalFP)

        globalPrecision = globalTP/(globalTP+globalFP)
        globalRecall = globalTP/(globalTP+globalFN)
        globalPR = globalPrecision * globalRecall
        globalF1 = 2*globalPR / (globalPrecision + globalRecall) if globalPR != 0 else 0

        globalOtherF1 = np.mean(F1)

        return globalAccuracy, globalPrecision, globalRecall, globalF1, globalOtherF1

    @staticmethod
    def calcMeasures(matrix):
        if len(matrix) == 0: return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        TP = np.diag(matrix)
        FP = np.sum(matrix, axis=0) - TP
        FN = np.sum(matrix, axis=1) - TP
        precision = np.divide(TP, TP+FP, out=np.zeros_like(TP, dtype=float), where=(TP+FP)!=0)
        recall = np.divide(TP, TP+FN, out=np.zeros_like(TP, dtype=float), where=(TP+FN)!=0)
        PR = precision * recall
        F1 = np.where(PR != 0, 2*PR / (precision + recall), 0)
        
        return F1, precision, recall, TP, FP, FN

    @staticmethod        
    def remove_metadata_records(model: FAISS, target_metadata: dict ):
        id_to_remove = []
        
        for _id, doc in model.docstore._dict.items():
            to_remove = True
            
            for k, v in target_metadata.items():
                if doc.metadata[k] != v:
                    to_remove = False
                    break
                    
            if to_remove:
                id_to_remove.append(_id)
                
        docstore_id_to_index = {v: k for k, v in model.index_to_docstore_id.items()}
        
        n_removed = len(id_to_remove)
        n_total = model.index.ntotal
        
        model.index.remove_ids(np.array(id_to_remove, dtype=np.int64))
        
        for _id in id_to_remove:
            # remove the document from the docstore
            del model.docstore._dict[_id]
            
            ind = docstore_id_to_index[_id]
            
            # remove the index to docstore id mapping
            del model.index_to_docstore_id[ind] 
       
        # reorder the mapping
        model.index_to_docstore_id = {i: _id for i, _id in enumerate(model.index_to_docstore_id.values())}
        
        return n_removed, n_total