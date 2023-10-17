import requests
import datetime

class VectorizerConnector(object):
    """A class used to connect to a vectorizer service"""

    def __init__(self, address, port,lang="-"):
        self.address = address
        self.port = port
        if (self.address.startswith("http")):
            self.urlPrefix = f"{self.address}:{self.port}/"
        else:
            self.urlPrefix = f"http://{self.address}:{self.port}/"	
        self.lang=lang
        pass

    def vectorize(self, utterance):
        if self.lang != "-":
            payload = {"q": utterance, "lang": self.lang}
        else:
            payload = {"q": utterance}
        try:
            r = requests.get(self.urlPrefix + "vectorize", params=payload)
            jsonret=r.json()
            return jsonret["vector"]
        except (requests.exceptions.HTTPError, requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')} | Failed to communicate with vectorization service!", flush=True)
            return []
        



