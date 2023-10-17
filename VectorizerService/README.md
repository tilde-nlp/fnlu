# Transformer vectorizer service

## Introduction
Returns sentence embeddings for the input text.

## Getting Started
To set up the Docker container perform the following steps:

1. Get the code of the container from the source controle.

2. Change the directory to the one containing the source.

3. Build the container with the following command:

`docker build -t vector_app .`

## Functionality
 
To start the vectorizer Web Service start the container with the following command:

`docker run -it -p 12345:12345 --name vec --rm vector_app`


## Web Service API

Method *vectorize* with parameter *q* returns sentence vector of the input string.

Example of the API call:

`http://.../vectorize?q=kas%20ajalehti%20saab`

Output:

`{"sentence": ["kas ajalehti saab"], "vector": [[0.0064210835844278336, -0.11738108098506927, ... ]]}`


@Tilde, 2023