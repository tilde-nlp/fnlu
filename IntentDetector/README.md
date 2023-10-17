# Multi-lingual intent detector
## Introduction
Detects intents in text.

## Getting Started
To set up the Docker container perform the following steps:

1. Get the code of the container from the source controle.

2. Change the directory to the one containing the source.

3. Build the container with the following command:

`docker build -t intdet .`

4. Prepare directory which will be mounted to the docker container. It should contain:

-- directory with training data (in .json format) if container will be used for training a new model;

Example of training data:

`[ {"text":"kas ajalehti saab", "intent":"Ajalehed/Ajakirjad"}, {"text":"Vajan animatsiooni kohta materjale", "intent":"Filmiteave"}, ...]`

5. Multi-lingual intent detection model uses LaBSE vectorizer Web Service.

 
## Functionality
Multi-lingual intent detector is a Docker container with several functions that are specified using environment variable ACT.

1. To train the model start the container with the following command:

`docker run -it --rm -p 22222:8888 -v <host_dir>:<container_dir> -e ACT="train" -e DATAFILE=<PATH_TO_JSON_FILE> -e MODEL_PREFIX=<MODEL_PREFIX_WITH_PATH> -e DO_XVAL="0" -e VECTORIZER_ADDRESS=<VECTORIZER_ADDRESS> -e VECTORIZER_PORT=<VECTORIZER_PORT> intdet`

2. To start the multi-lingual intent detector Web Service start the container with the following command:

`docker run -it --rm -p 22222:8888 -v <host_dir>:<container_dir> -e ACT="serve" -e MODEL_PREFIX=<MODEL_PREFIX_WITH_PATH> -e VECTORIZER_ADDRESS=<VECTORIZER_ADDRESS> -e VECTORIZER_PORT=<VECTORIZER_PORT> intdet`

Parameter *-v* links the host directory with the directory in container.

## Web Service API
Example of API call:

`http://.../intents?q=Kas%20originaalkunstist%20tohib%20teha%20koopiaid`

Output:
`{ "text": "Kas originaalkunstist tohib teha koopiaid", "intents": [ { "intentid": "Infopäring", "confidence": 0.9439612239599228, "source": "raamatukoguFaiss" }, { "intentid": "Filmiteave", "confidence": 0.3423901915550232, "source": "raamatukoguFaiss" }, ... ] }`


@Tilde, 2023