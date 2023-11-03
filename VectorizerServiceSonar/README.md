# SONAR vectorizer service
## Introduction
Returns sentence embeddings for the input text.

SONAR stands for Sentence-level multimOdal and laNguage-Agnostic Representations. https://github.com/facebookresearch/SONAR

## Getting Started
To set up the Docker container perform the following steps:

1. Get the code of the container from the source controle.

2. Build the container with the following command:

`docker build -t sonarvector_app .`

## Functionality
 
To start the vectorizer Web Service start the container with the following command:

`docker run -it -p 12345:12345 --name vec --rm sonarvector_app`

## Web Service API

1. Method *vectorize* returns sentence vector of the input string.
- Parameter *q* contains input string.
- Parameter *lang* contains text language. For Estonian it should be *est_Latn*.

Example of the API call:

`http://.../vectorize?q=kas%20teil%20on%20ajakirja%20pööning&lang=est_Latn`

Output:

`{"sentence": "kas teil on ajakirja pööning", "vector": [0.0064210835844278336, -0.11738108098506927, ... ]}`

2. Method *params* returns an information about the pre-trained embedding model.

Example of the API call:

`http://.../params`

Output:

{ "type": "transformer", "dim": 1024, "model_file_name": "SONAR" }

@Tilde, 2023