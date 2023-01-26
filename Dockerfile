FROM python:3.8

EXPOSE 8501
#/tcp

RUN apt-get update

RUN pip install --upgrade pip

# RUN pip install --upgrade cython

# RUN pip install --upgrade numpy

RUN mkdir -p /opt/pinsage

WORKDIR /opt/pinsage

COPY . .

COPY requirements.txt .

COPY validation-kdeep.py .

COPY training-kdeep.py .

COPY ./KData /KData

COPY ./graph_data /graph_data

COPY ./model /model

RUN pip install -r requirements.txt

CMD ["python", "./validation-kdeep.py"]