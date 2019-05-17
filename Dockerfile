FROM ubuntu:xenial

RUN apt -y update
RUN apt -y install git python python-setuptools gcc python-dev make

COPY . /pyzfp

WORKDIR /pyzfp

RUN python setup.py install
