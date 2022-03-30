FROM python:3.8.12-slim-buster

RUN apt-get update
RUN pip install gdown
RUN apt install -y tar
COPY  scripts /scripts
RUN chmod 777 scripts/*
RUN /bin/bash /scripts/get_data.sh
COPY models /models
COPY  src /src
COPY app.py /app.py
RUN apt install -y  curl
COPY requirements.txt /requirements.txt
RUN apt install -y libgl1-mesa-glx
RUN apt-get -y install libgl1
RUN apt-get -y install libglib2.0-0
RUN pip install -r requirements.txt
RUN pip uninstall -y werkzeug
RUN pip install -v https://github.com/pallets/werkzeug/archive/refs/tags/2.0.1.tar.gz


ENV PYTHONPATH=.
EXPOSE 8000

WORKDIR .


CMD ["gunicorn", "--timeout", "100", "-b 0.0.0.0:8000","app:server"]
                                        
