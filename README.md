# fnlu
Software for Federated NLU; Project: EKTB78 Liitõppe rakendamise võimalused dialoogiandmete põhjal

## Content
This repository contains 4 directories.

### VectorizerService

Directory [VectorizerService](VectorizerService) contains container code for the vectorizer based on LaBSE embedding model.

### VectorizerServiceSonar

Directory [VectorizerServiceSonar](VectorizerServiceSonar) contains container code for the vectorizer based on SONAR embedding model.

### IntentDetector

Directory [IntentDetector](IntentDetector) contains container code for the intent detector.

### Rasa

Directory [Rasa](Rasa) contains example Rasa bot project with custom intent detector trained/accessed through the Web Service.

### Other

- Contains several training data fails in .json format for training separate models (vector stores).
- Contains script file *MergeFaiss.py* for merging several vector stores. Argument 'vec_stores' contains file with vector stores' names to merge, and argument 'out_model' contains name of the merged vector store.
