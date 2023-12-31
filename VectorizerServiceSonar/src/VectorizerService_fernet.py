#!/usr/bin/python
##############################################################################
# Sentence vectorization Web Service using SONAR                             #
# Created by D.Deksne, @Tilde, 2023                                          #
##############################################################################
#   Copyright 2023 Tilde SIA

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import sys
import json
import datetime
import argparse
import os

import tornado.ioloop
import tornado.web
import asyncio
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from threading import Thread
import numpy as np
from datetime import datetime
import logging
from cryptography.fernet import Fernet

key = Fernet.generate_key()

logging.basicConfig(level=logging.INFO)
#est_Latn - Estonian, lvs_Latn - Latvian, lit_Latn - Lithuanian



def encrypt(listemb, encrypt):
    if encrypt:
        f = Fernet(key)
        return [f.encrypt(i) for i in listemb]
    else:
        return listemb

class CustomThread(Thread):
    def __init__(self, model, q, lang, encrypt):
        Thread.__init__(self)
        self.model = model
        self.q = q
        self.lang = lang
        self.encrypt = encrypt
        self.response = {'sentence': '', 'vector': []}
        print(key)
   
    def run(self):
        embeddings = self.model.predict(self.q, source_lang=self.lang)
        print(embeddings)
        listemb= [encrypt(item.tolist(), self.encrypt) for item in embeddings]
        print(listemb)
        self.response = {'sentence': self.q, 'vector': listemb}
        
class VectorizeRequestHandler(tornado.web.RequestHandler):
    def initialize(self, model):
        self.model = model
        self.__logger = logging.getLogger(self.__class__.__name__)

    async def get(self):
        q = self.get_query_argument("q", "", False)
        lang = self.get_query_argument("lang", "est_Latn", False)
        encrypt = self.get_query_argument("encrypt", False, False)
        
        if lang not in {"est_Latn","lvs_Latn","lit_Latn","eng_Latn"}:
            lang="est_Latn"
            
        try:
            qlist = json.loads(q)
        except:
            qlist = [q]

        thread = CustomThread(self.model, qlist, lang, encrypt)
        thread.start()
        thread.join()
        response = thread.response
        self.write(json.dumps(response, ensure_ascii=False))

class ParamsRequestHandler(tornado.web.RequestHandler):
    def initialize(self, model):
        self.model = model
        self.model_file_name = "SONAR"

    def get(self):
        embeddings = self.model.predict(["Text size"], source_lang="eng_Latn")
        response = {
            'type': 'transformer',
            'dim': len(embeddings.tolist()[0]),
            'model_file_name': self.model_file_name,
        }
        self.write(json.dumps(response, indent=2))

def make_app(model):
    return tornado.web.Application([
        (r"/vectorize", VectorizeRequestHandler, {"model": model}),
        (r"/params", ParamsRequestHandler, {"model": model}),
    ],debug=True)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=False, help="port to listen", type=int, default=8888)

    args = parser.parse_args()

    print("Starting server:")
    print(f"port: {args.port}")
    print("Please wait while vectorization model is being set up!", flush=True)
    model = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder")

    app = make_app(model)
    app.listen(args.port)

    print("Server started!")
    shutdown_event = asyncio.Event()
    await shutdown_event.wait()


if __name__ == "__main__":
    asyncio.run(main())

