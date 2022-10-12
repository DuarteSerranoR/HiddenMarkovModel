FROM continuumio/miniconda3:4.10.3-alpine

RUN apk update
RUN apk upgrade

RUN conda install numpy pandas
RUN conda install -c conda-forge uvicorn fastapi
RUN conda install -c anaconda nltk

COPY ./lib/download_nltk.py /download_nltk.py
RUN python /download_nltk.py
RUN rm /download_nltk.py

WORKDIR /app
COPY ./api.py /app/api.py
COPY ./model /app/model
COPY ./lib /app/lib
COPY ./decoders /app/decoders
COPY ./hmm /app/hmm

ENTRYPOINT [ "python", "api.py" ]
