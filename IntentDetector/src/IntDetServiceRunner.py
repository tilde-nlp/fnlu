import sys
import os
import argparse
import json
import tornado.ioloop
import tornado.web
import asyncio
import traceback
from IntDetClass import IntDetServerTrainer
from IntDetClass import log
from VectorizerConnector import VectorizerConnector
from urllib.parse import urlparse
import requests
import pickle
from langchain.embeddings import FakeEmbeddings
from langchain.vectorstores import FAISS

class IntDetHandler(tornado.web.RequestHandler):
    def initialize(self, detector : IntDetServerTrainer):
        self.detector = detector

    async def get(self):
        q = self.get_query_argument("q", "", False)
        #q - user's utterance
        
        lang = self.get_query_argument("lang", "", False)
        #est_Latn - Estonian

        returnValue = await tornado.ioloop.IOLoop.current().run_in_executor(None, self.detector.get_intents,q, lang)
        returnText = json.dumps(returnValue, indent=2, ensure_ascii=False)
        self.write(returnText)

class IntDetTrainHandler(tornado.web.RequestHandler):
    def initialize(self, detector : IntDetServerTrainer):
        self.detector = detector

    async def get(self):
        data = self.get_query_argument("data", "", False)
        if len(data)==0:
            self.write('{"status": "ERROR: No data received by Web Service!"}')
        else:
            lang = self.get_argument("lang", "", False)
            #est_Latn - Estonian
            
            source = self.get_query_argument("name", "", False)
            #dataset name
            
            new = self.get_query_argument("newmodel", "1", False)
            #new == '1' - create new model
            #new == '0' - merge data to the same model

            parsed_url = urlparse(data)
            if bool(parsed_url.scheme):
            #'data' is url, get local model's parameters (from the model that is trained localy)
                response = requests.get(f"{data}/getlocalparameters")
                
                if response.status_code == 200:
                    binary_data = response.content
                    model=FAISS.deserialize_from_bytes(embeddings=FakeEmbeddings(size=1), serialized=binary_data)
                    
                    returnValue = await tornado.ioloop.IOLoop.current().run_in_executor(None, self.detector.mergeModel,model,new)
                    returnText = json.dumps(returnValue, indent=2, ensure_ascii=False)
                    self.write(returnText)
                else:
                    print(f"Failed to retrieve parameters from {data}/getlocalparameters. Status code: {response.status_code}")
                    self.write('{"status": "ERROR: Failed to retrieve parameters from {data}/getlocalparameters."}')
            else:
            #'data' is text training data in .json format
                try:
                    trainData = json.loads(data)

                    returnValue = await tornado.ioloop.IOLoop.current().run_in_executor(None, self.detector.doVectorizing,trainData,lang)
                    returnValue = await tornado.ioloop.IOLoop.current().run_in_executor(None, self.detector.doTraining,source,new)

                    returnText = json.dumps(returnValue, indent=2, ensure_ascii=False)
                    self.write(returnText)
                except:
                    self.write('{"status": "ERROR: Failed to train with new data, probably wrong format."}')

    async def post(self):
        data = self.get_argument("data", "", False)
        if len(data)==0:
            self.write('{"status": "ERROR: No data received by Web Service!"}')
        else:
            lang = self.get_argument("lang", "", False)
            #est_Latn - Estonian
            
            source = self.get_argument("name", "", False)
            #dataset name

            new = self.get_argument("newmodel", "1", False)
            #new == '1' - create new model
            #new == '0' - merge data to the same model

            parsed_url = urlparse(data)
            if bool(parsed_url.scheme):
            #'data' is url, get local model's parameters (from the model that is trained localy)
                response = requests.get(f"{data}/getlocalparameters")
                
                if response.status_code == 200:
                    binary_data = response.content
                    model=FAISS.deserialize_from_bytes(embeddings=FakeEmbeddings(size=1), serialized=binary_data)
                    
                    returnValue = await tornado.ioloop.IOLoop.current().run_in_executor(None, self.detector.mergeModel,model,new)
                    returnText = json.dumps(returnValue, indent=2, ensure_ascii=False)
                    self.write(returnText)
                else:
                    print(f"Failed to retrieve parameters from {data}/getlocalparameters. Status code: {response.status_code}")
                    self.write("{\"status\": \"ERROR: Failed to retrieve parameters from {data}/getlocalparameters.\"}")
            else:
            #'data' is text training data in .json format
                try:
                    trainData = json.loads(data)

                    returnValue = await tornado.ioloop.IOLoop.current().run_in_executor(None, self.detector.doVectorizing,trainData,lang)
                    returnValue = await tornado.ioloop.IOLoop.current().run_in_executor(None, self.detector.doTraining,source,new)
                    
                    returnText = json.dumps(returnValue, indent=2, ensure_ascii=False)
                    self.write(returnText)
                except:
                    self.write('{"status": "ERROR: Failed to train with new data, probably wrong format."}')
                
class GetParamsHandler(tornado.web.RequestHandler):
    def initialize(self, detector : IntDetServerTrainer):
        self.detector = detector

    async def get(self):
        self.set_header('Content-Type', 'application/octet-stream')
        if self.detector.model != None:
            self.write(self.detector.model.serialize_to_bytes())
        self.finish()
        
def make_app(detector):
    return tornado.web.Application([
        (r"/intents", IntDetHandler, {"detector": detector}),
        (r"/train", IntDetTrainHandler, {"detector": detector}),
        (r"/getlocalparameters", GetParamsHandler, {"detector": detector}),
    ])


async def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('data', help='training data file')
    parser_train.add_argument('model_prefix', help='model name to output')
    parser_train.add_argument('vect_address', help='vectorizer service address')
    parser_train.add_argument('vect_port', help='vectorizer service port', type=int)
    parser_train.add_argument('lang', help='vectorization language')
    parser_train.add_argument('--xval', action='store_true', help='whether to do cross-validation')
    
    parser_serve = subparsers.add_parser('serve')
    parser_serve.add_argument('model_prefix', help='model name to use')
    parser_serve.add_argument('serving_port', help='port on which to serve')
    parser_serve.add_argument('vect_address', help='vectorizer service address')
    parser_serve.add_argument('vect_port', help='vectorizer service port', type=int)
    parser_serve.add_argument('lang', help='vectorization language')
    
    args = parser.parse_args()
    log(args)

    ft = VectorizerConnector(args.vect_address, args.vect_port, args.lang)

    modelName = args.model_prefix
    
    if args.command == "train":
        try:
            action = "xval" if args.xval else "train"
            with open(args.data, "r", encoding="utf-8") as f:
                trainData = json.load(f)
            
            if "train" in trainData and "examples" in trainData["train"]:
                idt = IntDetServerTrainer(ft, modelName, trainData["train"]["examples"], action)
            else:
                idt = IntDetServerTrainer(ft, modelName, trainData, action)
                
            idt.doVectorizing()
            
            if len(idt.x)>0:
                if args.xval:
                    idt.doXVal()
                else:
                    idt.doTraining()
            else:
                raise ValueError(f"Vectorization service failed!")
                
        except Exception as e:
            os.makedirs(os.path.dirname(modelName), exist_ok=True)
            with open(f"{modelName}.{action}.error", "w", encoding="utf-8") as f:
                f.write(f"Error {e.__class__}:\n{e}\n\nStack Trace:\n{traceback.format_exc()}")  
                
    elif args.command == "serve":
        singleIntDet = IntDetServerTrainer(ft, modelName)
        
        app = make_app(singleIntDet)
        app.listen(args.serving_port)

        print("Server started!")
        await asyncio.Event().wait()
    else:
        raise ValueError(f"Bad command: {args.command}")


if __name__ == "__main__":
    asyncio.run(main())



