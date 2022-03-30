FROM python:3.8.12-slim-buster

COPY requirements.txt /requirements.txt
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN apt-get -y install libgl1
RUN apt-get -y install libglib2.0-0
RUN pip install -r requirements.txt
RUN pip uninstall -y werkzeug
RUN pip install -v https://github.com/pallets/werkzeug/archive/refs/tags/2.0.1.tar.gz
COPY  data /data
COPY models /models
COPY  src /src
COPY app.py /app.py
RUN apt install -y  curl


ENV PYTHONPATH=.
EXPOSE 8000

WORKDIR .


CMD ["gunicorn", "--timeout", "100", "-b 0.0.0.0:8000","app:server"]
                                        
