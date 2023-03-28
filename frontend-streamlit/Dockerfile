FROM python:3.10 as build-stage
WORKDIR /tmp
RUN pip install poetry
COPY ./pyproject.toml ./poetry.lock* /tmp/
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM python:3.10-slim as production-stage
RUN pip install streamlit \
    && pip install requests \
    && pip install python-decouple


WORKDIR /CODE
COPY --from=build-stage /tmp/requirements.txt /CODE/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /CODE/requirements.txt
COPY . .

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "home.py", "--server.port=8501", "--server.address=0.0.0.0"]
#CMD [streamlit run home.py --server.port=8501 --server.address=127.0.0.1]
