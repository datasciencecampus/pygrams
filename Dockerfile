FROM python:3.7

ADD . /pygrams
WORKDIR /pygrams

RUN pip install -e .

ENTRYPOINT [ "python", "./pygrams.py" ]