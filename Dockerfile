FROM ubuntu:xenial

RUN apt -y update
RUN apt -y install git python python-setuptools gcc python-dev make python-pip


RUN adduser --disabled-password --gecos "" pyzfp

COPY . /home/pyzfp

RUN chown -R pyzfp /home/pyzfp

USER pyzfp
WORKDIR /home/pyzfp

RUN pip install -e .
