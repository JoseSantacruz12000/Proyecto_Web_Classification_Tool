FROM python:3.10 as build-stage
WORKDIR /tmp
RUN pip install poetry
COPY ./pyproject.toml ./poetry.lock* /tmp/
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes


FROM python:3.10 as production-stage
RUN apt-get update \
    && pip install mlflow

WORKDIR /code
COPY --from=build-stage /tmp/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . .

CMD ["mlflow", "server", "-h", "0.0.0.0", "-p", "4000"]
EXPOSE 4000

#mlflow server -h 127.0.0.1 -p 4000