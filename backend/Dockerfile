FROM python:3.10 as build-stage
WORKDIR /tmp
RUN pip install poetry
COPY ./pyproject.toml ./poetry.lock* /tmp/
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM python:3.10
RUN apt-get update \
    && pip install mlflow \
    && pip install tensorflow

WORKDIR /code
COPY --from=build-stage /tmp/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY . .
RUN python -m spacy download en_core_web_sm

#CMD ["python", "-m","spacy","download","en_core_web_sm"]
EXPOSE 4040
CMD ["python", "main.py"]

