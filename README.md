# fnlu
Software for Federated NLU; Project: EKTB78 Liitõppe rakendamise võimalused dialoogiandmete põhjal

## Content
This repository contains 3 directories.

### VectorizerService

Directory [VectorizerService](VectorizerService) contains container code for the vectorizer.

### IntentDetector

Directory [IntentDetector](IntentDetector) contains container code for the intent detector.

### Other

- Contains training data *rahvusraamatukogu.json* file prepared based on Rahvus raamatukogu data.
- Contains script file *MergeFaiss.py* for merging several vector stores. Argument 'vec_stores' contains file with vector stores' names to merge, and argument 'out_model' contains name of the merged vector store.
