# Prototype
This is the prototype of the intent detector trained using federated learning approach. Instructions bellow describe steps to set up the network with several client nodes and one server node.

## Use cases

### Case 1 - local intent detection systems

There are 3 institutions with private data. To train the intent detector for use of the institution's Virtual Assistant particular institution must perform the following steps:

- start the Vectorizer Web Service on institution premisses, the code is distributed as a container. You can use either [../VectorizerService](../VectorizerService) or [../VectorizerServiceSonar](../VectorizerServiceSonar).
- start the Intent Detection Web Service on institution premisses specifying command ACT="serve" and Vectorizer Service's URL and port.

- if institution uses VA implemented in rasa (see example [../rasa](../rasa) project):
    - specify your intents and examples in *data/nlu.yml* and responses to the intents in *domain.yml* as usually for the rasa project
	- in *config.yml* specify pipline including custom intent detection class *CustomIntentDetector.FederatedIntentDetector* and parameter *intdeturl* with link to the started Intent Detection Web Service and parameter *modelname* with the name of this VA
	- train the VA from the command line using command 'rasa train nlu'. 
	
- if institution uses custom VA that calls intent detector through the API:
    - prepare training data in the .json format. See files *kriisijuhtimine.json*, *rahvusraamatukogu.json*, *sotsiaalkindlustusamet.json* in the folder [../Other](../Other).
	- train the Intent Detection model by calling the Intent Detection Web Service's method *train* and passing
	    - the content of your .json file in *data* parameter,
		- the name of your dataset in *name* parameter,
		- 1 in *newmodel* parameter,
		- and 'est_Latn' in *lang* parameter.
		
	`http://.../train?newmodel=1&name=sotsiaalkindlustusamet&lang=est_Latn&data=[{"text":"puude taotlemine kus on", "intent":"puude taotlemine"},{"text":"Kuidas pikendada kaarti?", "intent":"puudega isiku kaart"},...]`

	- detect the user utterance's intent using the locally trained model by calling the Intent Detection Web Service's method *intents* and passing
	    - utterance in *q* parameter,
		- and 'est_Latn' in *lang* parameter.
		
	`http://.../intents?lang=est_Latn&data=[{"text":"puude taotlemine kus on", "intent":"puude taotlemine"},{"text":"Kuidas pikendada kaarti?", "intent":"puudega isiku kaart"},...]`


### Case 2 - single common intent detection system trained in federated way

- start the Vectorizer Web Service on the Server, the code is distributed as a container. You can use either [../VectorizerService](../VectorizerService) or [../VectorizerServiceSonar](../VectorizerServiceSonar).
- start the Intent Detection Web Service on the Server specifying command ACT="serve" and Vectorizer Service's URL and port.

- train the Intent Detection model that recognizes intents of general nature like 'greetings', 'thank you', etc. (to better suit your needs, please add more examples to this file) by calling the Intent Detection Web Service's method *train* and passing
	- the content of [../Other/general.json](../general.json) file in *data* parameter,
	- value 'general' in *name* parameter,
    - 1 in *newmodel* parameter,
	- and 'est_Latn' in *lang* parameter.

`http://.../train?newmodel=1&name=general&lang=est_Latn&data=[{"text":"Tere","intent":"tervitus"},{"text":"Terekest seal","intent":"tervitus"},...]`

- to add parameters of each institution to the general model on the Server call the Intent Detection Web Service's method *train* and pass
	- the institution's  Intent Detection Web Service's URL in *data* parameter,
	- institution's dataset name in *name* parameter,
    - 0 in *newmodel* parameter,
	- and 'est_Latn' in *lang* parameter.
	
`http://.../train?newmodel=0&name=kriisijuhtimine&lang=est_Latn&data=http://kriisijuhtimine_intitution:port`

`http://.../train?newmodel=0&name=rahvusraamatukogu&lang=est_Latn&data=http://rahvusraamatukogu_intitution:port`

`http://.../train?newmodel=0&name=sotsiaalkindlustusamet&lang=est_Latn&data=http://sotsiaalkindlustusamet_intitution:port`

Now the Intent Detection Web Service on the Server recognizes general intents and intents of each involved institution.

If you wish to use this Intent Detection WS for the VA implemented in rasa, in *config.yml* specify pipline including custom intent detection class *CustomIntentDetector.FederatedIntentDetector* and parameter *mainintdeturl* with the link to the Intent Detection Web Service on the Server. Also add functionality of passing control to the Virtual Assistant whose intent has been recognized. Module [../rasa/CustomIntentDetector.py](../rasa/CustomIntentDetector.py) starting from Line 106.