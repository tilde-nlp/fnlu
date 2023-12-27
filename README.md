# fnlu
Software for Federated NLU; Project: EKTB78 Liitõppe rakendamise võimalused dialoogiandmete põhjal

## Description
Nowadays, many companies and institutions use virtual assistants to relieve the work of customer support professionals and ensure continued communication with the organization beyond business hours.

A reliable, high-quality virtual assistant can not be developed without a good NLU model, particularly an intent detector.

Intent detection models are trained using a large, representative amount of example data. Depending on institution type the data might be sensitive or scattered among many institutions.

The solution in such cases is federated learning that involves training separate intent detection modules on each data holder premisses and sharing model parameters with the Server.

Using parameters from all the involved data holders the Server builds a single common intent detection model.

In such a way, the data holder's data in text form does not leave premisses, shared are only parameters in a binary format.
It allows for the preservation of the privacy of the data and reduces data exchange load.

Below is the architecture of the federated system with the Server and several remote modules.

![Architecture of the FL sytem](Federated_learning.jpg)

## Content
This repository contains 6 directories.

### VectorizerService

Directory [VectorizerService](VectorizerService) contains container code for the vectorizer based on the LaBSE embedding model.

### VectorizerServiceSonar

Directory [VectorizerServiceSonar](VectorizerServiceSonar) contains container code for the vectorizer based on the SONAR embedding model.

### IntentDetector

Directory [IntentDetector](IntentDetector) contains the container code for the intent detector.

### rasa

Directory [rasa](rasa) contains an example Rasa bot project with a custom intent detector trained/accessed through the Web Service.

### Other

- Contains several training data fails in .json format for training separate models (vector stores).
- Contains script file *MergeFaiss.py* for merging several vector stores. Argument 'vec_stores' contains a file with vector stores' names to merge, and argument 'out_model' contains the name of the merged vector store.

### Prototype

Directory [Prototype](Prototype) contains setup instructions on how to set up the system with several client nodes and one server node.
