FROM python:3.8-buster

WORKDIR /app
RUN apt-get update && apt-get install -y libsndfile-dev
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app

ENV port="12345"

ENTRYPOINT ["sh", "-c", "python src/VectorizerService.py ${transformer_dir} --port ${port}"]
#ENTRYPOINT ["sh", "-c", "python Test.py"]
