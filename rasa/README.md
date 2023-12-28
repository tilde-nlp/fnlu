# RASA project
Example Rasa project that uses custom intent detector

## Setup
1. Install Rasa using instructions [https://rasa.com/docs/rasa/installation/environment-set-up/](https://rasa.com/docs/rasa/installation/environment-set-up/)

2. Start the Intent detection Web Service's container. 

3. Open [config.yml](config.yml) file. Rasa project pipline contains custom intent classifier *CustomIntentDetector.FederatedIntentDetector*. Specify Web Service's URL in *intdeturl*. Specify model's name in *modelname* variable.

4. Train intent detection model through Rasa by executing command 'rasa train nlu'.

5. In case of success execute command 'rasa shell' and converse with the bot. Answers will contain the name of the recognized intent. Answers are specified in the file [domain.yml](domain.yml).

Custom intent detector is implemented in file [CustomIntentDetector.py](CustomIntentDetector.py).

