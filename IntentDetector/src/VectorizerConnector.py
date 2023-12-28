#   Copyright 2023 Tilde SIA

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

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

    def vectorize(self, utterance,otherlang=""):

        if otherlang != "":
            payload = {"q": utterance, "lang": otherlang}
        elif self.lang != "-":
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
        



