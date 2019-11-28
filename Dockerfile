FROM python:3.7

ADD . /pygrams
WORKDIR /pygrams

RUN apt-get install unixodbc-dev

RUN pip install -r requirements.txt && \
    python -m nltk.downloader punkt averaged_perceptron_tagger wordnet

CMD [ "python", "./pygrams/pygrams.py" ]