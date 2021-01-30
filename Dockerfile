FROM python:3.7-slim-buster

# Install dependencies required for psycopg2 python package
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y netcat-openbsd gcc python3-dev musl-dev libpq-dev libboost-all-dev && \
    apt-get clean

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
COPY . .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


# run entrypoint.sh
ENTRYPOINT ["/usr/src/app/entrypoint.sh"]

