FROM python:3.9

ENV APP_HOME /app
WORKDIR $APP_HOME
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python3", "app2.py"]