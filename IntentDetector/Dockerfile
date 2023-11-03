FROM python:3.10-buster

WORKDIR /app
#RUN apt-get update && apt-get install -y python3-dev
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY src src

COPY runIntDet.sh .
RUN chmod +x runIntDet.sh

ENV ACT="train"
ENV DO_XVAL="0"
ENV DATAFILE=""
ENV MODEL_PREFIX="/data/NonoModel/testmodel"
ENV VECTORIZER_ADDRESS="vec"
ENV VECTORIZER_PORT="12345"
ENV SERVING_PORT="8888"
ENV LANG="eng_Latn"

ENTRYPOINT ["./runIntDet.sh"]