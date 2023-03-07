FROM python:3.9-slim-buster

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
run pip3 install flask
COPY . .

WORKDIR /src/predict

CMD [ "python3", "predictor.py", "--host=0.0.0.0"]