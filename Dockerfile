FROM python:3.7

ADD . /pygrams
WORKDIR /pygrams

RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt averaged_perceptron_tagger wordnet

ENTRYPOINT [ "python", "./pygrams.py" ]