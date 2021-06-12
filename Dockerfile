FROM python:3.9-slim

# To reduce lag in log-aggregation
ENV PYTHONUNBUFFERED True

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN pip install -r requirements.txt

# threads should be changed to # of CPU cores
# timeout 0 to allow cluster manager to do fast instance scaling
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
