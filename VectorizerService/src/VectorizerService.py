#!/usr/bin/python
##############################################################################
# Sentence vectorization Web Service using transformer embeddings            #
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
from sentence_transformers import SentenceTransformer
from threading import Thread
import numpy as np
from datetime import datetime

class CustomThread(Thread):
    def __init__(self, model, q):
        Thread.__init__(self)
        self.model = model
        self.q = q
        self.response = {'sentence': [], 'vector': []}
        
    def run(self):
        embeddings = self.model.encode(self.q)
        listemb= [item.tolist() for item in embeddings]
        self.response = {'sentence': self.q, 'vector': listemb}
        
class VectorizeRequestHandler(tornado.web.RequestHandler):
    def initialize(self, model):
        self.model = model

    async def get(self):
        q = self.get_query_argument("q", "", False)
        #print(q,flush=True)
        try:
            qlist = json.loads(q)
        except:
            qlist = [q]
        thread = CustomThread(self.model, qlist)
        thread.start()
        thread.join()
        response = thread.response
        self.write(json.dumps(response, ensure_ascii=False))

 
def make_app(model):
    return tornado.web.Application([
        (r"/vectorize", VectorizeRequestHandler, {"model": model}),
    ],debug=True)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=False, help="port to listen", type=int, default=8888)

    args = parser.parse_args()

    print("Starting:")
    print(f"port: {args.port}")

    model = SentenceTransformer('sentence-transformers/LaBSE')
    app = make_app(model)
    app.listen(args.port)

    print("Server started!")
    shutdown_event = asyncio.Event()
    await shutdown_event.wait()


if __name__ == "__main__":
    asyncio.run(main())

