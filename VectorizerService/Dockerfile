FROM python:3.10-buster

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app

ENV port="12345"

ENTRYPOINT ["sh", "-c", "python src/VectorizerService.py --port ${port}"]
#ENTRYPOINT ["sh", "-c", "python Test.py"]
