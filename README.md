# fnlu
Software for Federated NLU; Project: EKTB78 Liitõppe rakendamise võimalused dialoogiandmete põhjal

## Description
Nowadays, many companies use virtual assistants to relieve the work of customer support professionals and ensure continued communication with the company beyond business hours as well.

Reliable, high-quality virtual assistant can not be developed without a good NLU model, particulary an intent detector.

Intent detection models are trained using a large, representative amount of example data. Depending on institution type the data might be sensitive or scattered among many institutions.

The solution in such cases are federated learning that involves training separate intent detection modules on each data holder premisses and sharing model parameters with the Server.

Using parameters from all the involved data holders the Server builds a single common intent detection model.

In such way data holders data in text form does not leave premisses, shared are only parameters in a binary format.
It allows preserving privacy of the data and reducing data exchange load.

Bellow are the architecture of the federated system with the Server and several remote modules.

![Architecture of the FL sytem](Federated_learning.jpg)

## Content
This repository contains 6 directories.

### VectorizerService

Directory [VectorizerService](VectorizerService) contains container code for the vectorizer based on LaBSE embedding model.

### VectorizerServiceSonar

Directory [VectorizerServiceSonar](VectorizerServiceSonar) contains container code for the vectorizer based on SONAR embedding model.

### IntentDetector

Directory [IntentDetector](IntentDetector) contains container code for the intent detector.

### rasa

Directory [rasa](rasa) contains example Rasa bot project with custom intent detector trained/accessed through the Web Service.

### Other

- Contains several training data fails in .json format for training separate models (vector stores).
- Contains script file *MergeFaiss.py* for merging several vector stores. Argument 'vec_stores' contains file with vector stores' names to merge, and argument 'out_model' contains name of the merged vector store.

### Prototype

Directory [Prototype](Prototype) contains setup instructions how to set up the system with several client nodes and one server node.
