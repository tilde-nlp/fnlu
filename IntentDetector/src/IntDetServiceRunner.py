print("Starting 1", flush=True)
import sys
import os
import argparse
import json
import tornado.ioloop
import tornado.web
import asyncio
import traceback
from IntDetClass import IntDetServer, IntDetTrainer
from IntDetClass import log
from VectorizerConnector import VectorizerConnector

class IntDetHandler(tornado.web.RequestHandler):
    def initialize(self, detector : IntDetServer):
        self.detector = detector

    async def get(self):
        q = self.get_query_argument("q", "", False)
        returnValue = await tornado.ioloop.IOLoop.current().run_in_executor(None, self.detector.get_intents,q)
        returnText = json.dumps(returnValue, indent=2, ensure_ascii=False)
        self.write(returnText)

def make_app(detector):
    return tornado.web.Application([
        (r"/intents", IntDetHandler, {"detector": detector}),
    ])


async def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('data', help='training data file')
    parser_train.add_argument('out_model_prefix', help='model name to output')
    parser_train.add_argument('vect_address', help='vectorizer service address')
    parser_train.add_argument('vect_port', help='vectorizer service port', type=int)
    parser_train.add_argument('--xval', action='store_true', help='whether to do cross-validation')
    
    parser_serve = subparsers.add_parser('serve')
    parser_serve.add_argument('model_prefix', help='model name to use')
    parser_serve.add_argument('serving_port', help='port on which to serve')
    parser_serve.add_argument('vect_address', help='vectorizer service address')
    parser_serve.add_argument('vect_port', help='vectorizer service port', type=int)

    args = parser.parse_args()
    log(args)

    if args.command == "train":
        try:
            modelName = args.out_model_prefix
            action = "xval" if args.xval else "train"
            with open(args.data, "r", encoding="utf-8") as f:
                trainData = json.load(f)
            ft = VectorizerConnector(args.vect_address, args.vect_port,args.lang)
            
            if "train" in trainData and "examples" in trainData["train"]:
                idt = IntDetTrainer(ft, modelName, trainData["train"]["examples"], action)
            else:
                idt = IntDetTrainer(ft, modelName, trainData, action)
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
        ft = VectorizerConnector(args.vect_address, args.vect_port)

        modelName = args.model_prefix
        port = args.serving_port
        singleIntDet = IntDetServer(ft, modelName)
        app = make_app(singleIntDet)
        app.listen(port)

        print("Server started!")
        await asyncio.Event().wait()
    else:
        raise ValueError(f"Bad command: {args.command}")


if __name__ == "__main__":
    asyncio.run(main())



