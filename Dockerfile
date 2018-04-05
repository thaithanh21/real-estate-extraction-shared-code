FROM ubuntu:16.04

# Install Python.
RUN \
  apt-get update && \
  apt-get install -y python3 python3-dev python3-pip python3-virtualenv && \
  rm -rf /var/lib/apt/lists/*

RUN mkdir /real-estate-model
COPY ./data_utils /real-estate-model/data_utils
COPY ./model /real-estate-model/model
COPY ./server /real-estate-model/server
COPY ./requirements.txt /real-estate-model
WORKDIR /real-estate-model
RUN pip3 install -r requirements.txt
COPY ./run-prediction-service.sh /real-estate-model
ENV PYTHONPATH=/real-estate-model
EXPOSE 5000
ENTRYPOINT ["bash"]
CMD ["run-prediction-service.sh"]
