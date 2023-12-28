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

from __future__ import annotations
from typing import Any, Dict, List, Optional, Text, Tuple, Union, TypeVar, Type

from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message

import os
import numpy as np
import tensorflow as tf
import json
import requests
#from requests_toolbelt import MultipartEncoder
@DefaultV1Recipe.register(DefaultV1Recipe.ComponentType.INTENT_CLASSIFIER, is_trainable=True)
class FederatedIntentDetector(IntentClassifier, GraphComponent):
    name = "fedintdet"
    provides = ["intent"]
    requires = ["text"]
    lanuage_list = ["et"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {}
        
    def __init__(self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        intent2id: Optional[Dict[Text,int]] = {},
        id2intent: Optional[Dict[int,Text]] = {},
        model: Optional[Any] = None,
        vectorizer_model: Optional[SentenceTransformer] = None,) -> None:
        
        self.component_config = config
        self._model_storage = model_storage
        self._resource = resource


    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,) -> GraphComponent:
        return cls(config, model_storage, resource, execution_context)
      
    def _transform_data(self, data):
        training_data = []
        
        for message in data.training_examples:
            if "text" in message.data and "intent" in message.data:
                training_data.append({"text":message.data["text"], "intent":message.data["intent"]})
        
        return training_data

    def train(self, training_data: TrainingData) -> Resource:
        training_data = self._transform_data(training_data)

        payload = {
            'lang':'est_Latn',
            'newmodel':'1',
            'name':self.component_config['modelname'],
            'data':json.dumps(training_data)}
            
        url=self.component_config['intdeturl']
        response = requests.post(f"{url}/train", data=payload, headers={'Accept':'text/json'})

        if response.status_code == 200:
            print(response.content)
        else:
            print("Call to Web Service failed!")
            
        return self._resource

    def _predict(self, text) -> Tuple[Dict[Text, Any], List[Dict[Text, Any]]]:
        """Predicts the intent of the provided message."""
        label: Dict[Text, Any] = {"name": None, "confidence": 0.0}
        label_ranking: List[Dict[Text, Any]] = []
        
        payload = {
            'lang':'est_Latn',
            'q':text}
        
        #using general intent detector from the Server
        url=self.component_config['mainintdeturl'] 
        
        #for using local intent detection model
        #url=self.component_config['intdeturl']
        
        response = requests.get(f"{url}/intents", params=payload, headers={'Accept':'text/json'})
            
        if response.status_code == 200:
            results=json.loads(response.content)
            
            for item in results['intents']:
                if item['source'] == self.component_config['modelname']:
                    #intent can be handled by this virtual assistant
                    label_ranking.append({'name':item['intentid'], 'confidence':item['confidence']})
                else:
                    #intent belongs to a different virtual assistant
                    #re-routing can be implemented here ...
                    print(f"Detected intent of {item['source']} virtual assistant")
                
            if len(label_ranking)>0:
                label=label_ranking[0]
        else:
            print("Call to Web Service failed!")
            
        return label, label_ranking

    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            text = message.data["text"]
            toplabel, label_ranking = self._predict(text)

            message.set("intent", toplabel, add_to_output=True)
            message.set("intent_ranking", label_ranking, add_to_output=True)
  
        return messages

        
