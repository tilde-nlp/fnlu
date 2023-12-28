# Federated NLU
This repository hosts both the software for federated learning and a prototype implementation for the project:: EKTB78 Liitõppe rakendamise võimalused dialoogiandmete põhjal

## Description
Nowadays, many companies and institutions use virtual assistants to relieve the work of customer support professionals and ensure continued communication with the organization beyond business hours. A reliable, high-quality virtual assistant can not be developed without a good Natural Language Understanding (NLU) model, particularly an intent detector.

This project is dedicated to addressing challenges associated with the development of sophisticated bots designed to cater to the diverse needs of multiple organizations. Our objective involves constructing independent bots for each organization while concurrently creating a unified bot capable of serving the collective requirements of all participating entities. This approach empowers us to meticulously develop and assess bots tailored to the unique demands of individual organizations. Subsequently, these individualized bots can seamlessly integrate into a unified bot, ensuring a cohesive user experience.

The rationale behind the creation of a unified bot stems from the realization that end users often lack awareness of, and interest in, the specific bot they are interacting with. 

To illustrate this solution architecture, consider the following depiction:

![Architecture of the FL sytem](Federated_learning.jpg)

The project encompasses various remote bot training sites where bot trainers autonomously develop and test their respective bots. These trainers manage their private training data, training local NLU models that can be employed within their specific remote bots as needed.

In addition to the remote training sites, a central training site plays a pivotal role in the federated learning process. At this central location, a singular NLU model is trained using a federated training approach. The central federated training process aggregates NLU model parameters from the remote training sites, consolidating them into a cohesive federated NLU model. Notably, the federated training process not only acquires parameters from the remote nodes but may also incorporate its own training data.

The outcome of this federated training process is a central bot equipped with the federated NLU model. This central bot possesses the capability to identify intents irrespective of their origin, as the federated NLU model recognizes intents defined in shared training data and any of the remote sites. This innovative approach ensures a versatile and inclusive bot system that effectively addresses the intricate needs of diverse organizations.

The process of training intent detection models traditionally involves utilizing a substantial and representative dataset. However, when dealing with data from diverse organizations, it becomes imperative to acknowledge potential sensitivities, with data owners being understandably hesitant to disclose the content of their data due to various reasons such as data quality, personal or classified information, or sensitive user queries posed to the bot.

The federated learning approach implemented in this project offers an effective solution to this privacy challenge. By employing federated learning, the central federated NLU model is trained without the need for sensitive data to leave the remote training sites. At no point does the central federated training process directly access or utilize the training data. Instead, the training data is vectorized on the remote site, and these vectors are utilized to compute the parameters of the local NLU model. Crucially, the raw data itself is never stored within the model.

Upon completion of the federated training process, only the model parameters and vectors are disclosed to the central training site. This meticulous approach ensures that the data, in its textual form, remains securely within the premises of the data holder. Only binary representations of parameters are shared, safeguarding the privacy of the data and significantly reducing the load associated with data exchange. This not only mitigates privacy concerns but also optimizes the efficiency of the overall federated learning system.

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
