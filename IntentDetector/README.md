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

5. Intent detection model uses vectorizer Web Service. It should be started prior to intent detector.

 
## Functionality
Intent detector is a Docker container with several functions that are specified using environment variable *ACT*. The values is eaither *train* or *serve*.

1. To train the model without starting the Web Service start the container with the following command:

`docker run -it --rm -p 22222:8888 -v <host_dir>:<container_dir> -e ACT="train" -e DATAFILE=<PATH_TO_JSON_FILE> -e MODEL_PREFIX=<MODEL_PREFIX_WITH_PATH> -e DO_XVAL="0" -e VECTORIZER_ADDRESS=<VECTORIZER_ADDRESS> -e VECTORIZER_PORT=<VECTORIZER_PORT> -e LANG=est_Latn intdet`

This option is used if you wish 

- to perform 5 fold x-validation on your training data (in .json format), then environment variable DO_XVAL="1". Only results of x-validation are saved to the disk.
- to train new initial model, then environment variable DO_XVAL="0". The trained model is saved to the disk in the directory specified by *MODEL_PREFIX* environment variable.

2. To start the intent detector Web Service start the container with the following command:

`docker run -it --rm -p 22222:8888 -v <host_dir>:<container_dir> -e ACT="serve" -e MODEL_PREFIX=<MODEL_PREFIX_WITH_PATH> -e VECTORIZER_ADDRESS=<VECTORIZER_ADDRESS> -e VECTORIZER_PORT=<VECTORIZER_PORT> -e LANG=est_Latn intdet`

Parameter *-v* links the host directory with the directory in container.

## Web Service API

The Web Service has several methods:

1. Method *intents* return a list of intents with confidence scores and source information for the provided text
	- parameter *q*, text (user question) 
	- parameter *lang*, text language , *est_Latn* for Estonian

Example of API call:

`http://.../intents?q=Kas%20originaalkunstist%20tohib%20teha%20koopiaid&lang=est_Latn`

Output:
`{ "text": "Kas originaalkunstist tohib teha koopiaid", "intents": [ { "intentid": "Infopäring", "confidence": 0.9439612239599228, "source": "raamatukogu" }, { "intentid": "Filmiteave", "confidence": 0.3423901915550232, "source": "raamatukogu" }, ... ] }`

2. Method *train* trains and/or merges models
	- parameter *data*
		- contains URL to the site from which to get allready trained model, this value is used by the Server that collects and merges models from the local users
		- contains data in .json format for model training
	- parameter *newmodel*, value *1* for training a new model, value *0* for merging models
	- parameter *lang*, model's text language , *est_Latn* for Estonian
	- parameter *name*, model's name, will be returned by *intents* method as a *source* of intent.
	
POST and GET requests available

Example of API call:
http://localhost:22221/train?newmodel=0&name=sotsiaalkindlustusamet&data=http::\/localhost:22222&lang=est_Latn

Output: 
`{"status": "SUCCESS"}`
 or 
`{"status": "ERROR: Failed to retrieve parameters from {data}/getlocalparameters."}`

3. Method *getlocalparameters* requests the parameters from the local users models, this method is called by the Server. Server merges local models with the central model.

Example of API call:

`http://.../getlocalparameters`

Output: model parameters in binary format
	
@Tilde, 2023